"""
SILVER LAYER — Cleaned & Validated Flight Data
===============================================

This module reads raw flight data from the bronze layer (ingest_flights),
applies data quality rules, cleans string fields, filters invalid
geographic coordinates, deduplicates near-identical events, and adds
processing metadata. The result is a streaming table of trustworthy,
analysis-ready flight records for the gold layer.

Architecture decisions
----------------------
1. STREAMING TABLE (not Materialized View):
   Silver reads from a bronze streaming table using spark.readStream,
   which preserves incremental processing semantics. Each pipeline update
   processes only newly arrived bronze data rather than re-scanning the
   full history. A materialized view would recompute all cleaning
   transformations on every run.
   Alternative: If the upstream bronze layer used Auto CDC (with updates
   or deletes) we would need skipChangeCommits. Bronze is append-only,
   so no special handling is needed.

2. DEDUPLICATION WITH WATERMARK (not global dropDuplicates):
   dropDuplicatesWithinWatermark limits state to the watermark window
   (10 minutes) instead of tracking all keys indefinitely. This is
   essential for high-volume streaming — global dropDuplicates would
   accumulate unbounded state and eventually cause out-of-memory errors.
   The watermark uses time_position (the actual observation timestamp)
   rather than time_ingest (when we received it) because duplicate events
   share the same observation time.
   Alternative: If exact global deduplication were required we would use
   dropDuplicates without a watermark, but this is not sustainable for
   long-running streams.

3. EXPECTATIONS — strict enforcement (drop, not just warn):
   Unlike bronze's lenient warn-only expectations, silver uses
   expect_or_drop for critical fields — rows with null icao24 or invalid
   coordinates are dropped because they cannot be used for any
   meaningful flight analysis. expect (warn) is used for softer checks
   (e.g., null callsign) where the record is still useful.
   Alternative: expect_or_fail would halt the pipeline on any bad row,
   which is too aggressive for a live data feed where occasional bad
   records are expected.

4. CLUSTERING KEY (icao24):
   Liquid clustering on icao24 optimises queries that filter by aircraft
   identifier — the most common access pattern for flight data analysis.
   Alternative: Partitioning by date could help time-based queries, but
   flight data is queried more often by aircraft ID than by time range.

5. LOGGING:
   Module-level logging provides visibility into the cleaning process.
   Log statements do not affect data flow.

6. NO TRY/CATCH IN DATASET DEFINITIONS:
   SDP handles failures automatically. Wrapping transformations in
   try/catch would hide errors from the framework's retry logic.
"""

import logging

from pyspark import pipelines as dp
from pyspark.sql.functions import col, trim, current_timestamp

# Module-level logger for the silver layer.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SILVER STREAMING TABLE: cleaned_flights
# ---------------------------------------------------------------------------
# Reads the bronze streaming table, applies cleaning transformations, and
# materialises the result as a new streaming table.
#
# Cleaning steps:
#   1. Trim whitespace from string columns (callsign, origin_country)
#   2. Drop rows where icao24 is null (via expectation)
#   3. Drop rows with invalid geographic coordinates (via expectation)
#   4. Filter out negative velocity values
#   5. Deduplicate within a 10-minute watermark window
#   6. Add a processing timestamp (_silver_processed_at)
#
# Published to workspace.default.cleaned_flights.
# ---------------------------------------------------------------------------
@dp.table(
    comment=(
        "Silver layer: cleaned and validated flight data from OpenSky. "
        "Streaming table with deduplication, coordinate validation, "
        "and whitespace trimming."
    ),
    cluster_by=["icao24"],
)
# Strict expectations: drop rows that cannot be used for analysis.
@dp.expect_or_drop("icao24_not_null", "icao24 IS NOT NULL")
@dp.expect_or_drop("valid_latitude", "latitude IS NULL OR latitude BETWEEN -90 AND 90")
@dp.expect_or_drop("valid_longitude", "longitude IS NULL OR longitude BETWEEN -180 AND 180")
# Lenient expectations: warn but keep the row.
@dp.expect("callsign_present", "callsign IS NOT NULL")
@dp.expect("valid_velocity", "velocity IS NULL OR velocity >= 0")
def cleaned_flights():
    """
    Clean and validate raw flight data from the bronze layer.

    Reads the bronze streaming table (ingest_flights) and applies:
    - Whitespace trimming on string columns (callsign, origin_country)
    - Negative velocity filtering (keep nulls — may indicate ground status)
    - Deduplication within a 10-minute watermark window on (icao24, time_position)
    - Processing timestamp for lineage tracking (_silver_processed_at)

    Returns a streaming DataFrame materialised as a Delta table.
    """
    logger.info("Starting silver layer cleaning transformation.")

    return (
        spark.readStream.table("ingest_flights")
        # Watermark on time_position bounds state for deduplication.
        # 10 minutes tolerates late-arriving data while keeping state small.
        .withWatermark("time_position", "10 minutes")
        # Deduplicate within the watermark window. Key on icao24 + time_position
        # to drop redundant position reports for the same aircraft at the same
        # observation time. OpenSky can emit the same record multiple times.
        .dropDuplicatesWithinWatermark(["icao24", "time_position"])
        # Trim whitespace from string columns. OpenSky sometimes pads callsigns
        # and country names with trailing spaces (e.g., "UAL419  ").
        .withColumn("callsign", trim(col("callsign")))
        .withColumn("origin_country", trim(col("origin_country")))
        # Filter out negative velocity — physically impossible for speed.
        # We keep null velocity (may indicate the aircraft is on the ground).
        .filter((col("velocity").isNull()) | (col("velocity") >= 0))
        # Add processing timestamp for lineage — records when the silver layer
        # processed this row. This is distinct from time_ingest (bronze arrival
        # time) and time_position (OpenSky observation time).
        .withColumn("_silver_processed_at", current_timestamp())
    )