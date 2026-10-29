"""
BRONZE LAYER — Raw Flight Data Ingestion
=========================================

This module ingests live flight tracking data from the OpenSky Network
via a custom Spark data source. The bronze layer's primary responsibility
is to land raw data as-is, with minimal transformation, so that
downstream silver and gold layers can clean and aggregate it.

Architecture decisions
----------------------
1. STREAMING TABLE (not Materialized View):
   The OpenSky data source emits a continuous stream of flight position
   updates. A streaming table (@dp.table with spark.readStream) enables
   incremental ingestion — only new data is processed each run, which is
   far more efficient than re-reading the full source every time.
   Alternative: A materialized view (spark.read) would re-scan the entire
   source on every update — wasteful for a continuously growing stream.

2. CUSTOM DATA SOURCE (not Auto Loader):
   OpenSky provides a programmatic API, not files in cloud storage.
   Auto Loader is designed for file-based ingestion (CSV, JSON, Parquet
   in S3/ADLS/GCS) and is not applicable to an API-based source.
   Alternative: If OpenSky data were landed as files, we would use Auto
   Loader with cloudFiles format for schema inference and evolution.

3. EXPECTATIONS — lenient (warn, not drop/fail):
   Bronze expectations intentionally warn rather than drop or fail so
   that ALL raw data is captured, including imperfect records, for
   full auditability. Silver enforces stricter quality rules.
   Alternative: expect_or_drop would silently discard bad rows before
   they are landed, making debugging harder. expect_or_fail would halt
   ingestion on any quality issue.

4. LOGGING:
   Python's logging module is used for informational purposes. In SDP,
   dataset functions may be evaluated multiple times during planning,
   so log statements may appear more than once — this is acceptable and
   does not affect data processing.

5. NO TRY/CATCH IN DATASET DEFINITIONS:
   SDP handles failure recovery automatically (retries failed updates,
   restarts clusters for recoverable errors). Wrapping dataset logic in
   try/catch would mask errors from the framework's retry mechanism.
   Try/catch is only used in the foreach_batch_sink handler (gold layer)
   where custom SQL execution benefits from explicit error handling.
"""

import logging

from pyspark import pipelines as dp
try:
    from pyspark_datasources import OpenSkyDataSource
except ImportError:
    OpenSkyDataSource = None

# Module-level logger. Using __name__ ensures the logger is named after
# the module, making it easy to filter logs by pipeline component.
logger = logging.getLogger(__name__)

# Register the custom OpenSky data source with Spark. This must happen
# before any readStream call references the "opensky" format. The
# pyspark-data-sources environment dependency (in pipeline settings)
# provides the OpenSkyDataSource class.
# The import is wrapped in a try/except so that the module can be loaded
# in test environments where pyspark-data-sources is not installed.
# The bronze flow is not exercised during silver/gold unit tests.
if OpenSkyDataSource is not None:
    spark.dataSource.register(OpenSkyDataSource)
    logger.info("OpenSky data source registered successfully.")
else:
    logger.warning("pyspark-data-sources not available; OpenSky data source not registered.")


# ---------------------------------------------------------------------------
# BRONZE STREAMING TABLE: ingest_flights
# ---------------------------------------------------------------------------
# A streaming table ingests flight position updates incrementally.
# The data is stored as-is (raw) — no cleaning or transformation is applied
# at this layer. This preserves the full audit trail of what OpenSky
# reported, even if some records are malformed or incomplete.
#
# Published to workspace.default.ingest_flights (per pipeline catalog/schema).
#
# Alternative considered: Using a materialized view with spark.read would
# re-fetch the entire OpenSky dataset on every pipeline update. For a
# continuously growing stream this would be prohibitively expensive and
# could lose data between runs. The streaming table ensures exactly-once
# processing semantics with automatic checkpointing.
# ---------------------------------------------------------------------------
@dp.table(
    comment=(
        "Bronze layer: raw flight position data from OpenSky Network. "
        "Streaming table for incremental ingestion. No cleaning applied."
    ),
)
# Expectations are warn-only so that ALL raw data is captured for audit.
# Silver layer enforces stricter rules (drop/fail).
@dp.expect("icao24_present", "icao24 IS NOT NULL")
@dp.expect("valid_velocity", "velocity IS NULL OR velocity >= 0")
@dp.expect("valid_latitude", "latitude IS NULL OR latitude BETWEEN -90 AND 90")
@dp.expect("valid_longitude", "longitude IS NULL OR longitude BETWEEN -180 AND 180")
def ingest_flights():
    """
    Stream raw flight data from the OpenSky custom data source.

    Returns a streaming DataFrame materialised as a Delta table.
    Expectations log (warn) data quality issues without dropping records,
    preserving the full raw dataset for downstream cleaning in silver.
    """
    logger.info("Starting OpenSky stream ingestion for bronze layer.")
    return spark.readStream.format("opensky").load()


# ---------------------------------------------------------------------------
# ONE-TIME TEST DATA INJECTION FLOW
# ---------------------------------------------------------------------------
# A one-time append flow that injects rows violating each bronze
# expectation so that the data quality checks have something to report.
# Runs only once per pipeline lifetime (re-runs on full refresh).
# Remove this flow once live OpenSky data naturally triggers the checks.
# ---------------------------------------------------------------------------
@dp.append_flow(
    target="ingest_flights",
    name="test_data_injection",
    once=True,
    comment=(
        "One-time injection of test rows that violate bronze expectations "
        "for data quality demonstration."
    ),
)
def test_data_injection():
    """
    Inject test rows that violate each bronze expectation (warn-only).

    Rows are crafted to trigger every expectation at least once:
      - null icao24          → icao24_present
      - negative velocity    → valid_velocity
      - latitude > 90       → valid_latitude
      - longitude > 180     → valid_longitude
    """
    from datetime import datetime
    from pyspark.sql.types import (
        StructType, StructField, TimestampType, StringType,
        DoubleType, BooleanType, IntegerType, ArrayType,
    )

    schema = StructType([
        StructField("time_ingest", TimestampType(), True),
        StructField("icao24", StringType(), True),
        StructField("callsign", StringType(), True),
        StructField("origin_country", StringType(), True),
        StructField("time_position", TimestampType(), True),
        StructField("last_contact", TimestampType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("geo_altitude", DoubleType(), True),
        StructField("on_ground", BooleanType(), True),
        StructField("velocity", DoubleType(), True),
        StructField("true_track", DoubleType(), True),
        StructField("vertical_rate", DoubleType(), True),
        StructField("sensors", ArrayType(IntegerType()), True),
        StructField("baro_altitude", DoubleType(), True),
        StructField("squawk", StringType(), True),
        StructField("spi", BooleanType(), True),
        StructField("category", IntegerType(), True),
    ])

    now = datetime.now()

    test_rows = [
        # Violates icao24_present (null icao24)
        (now, None, "BAD001", "XX", now, now,
         10.0, 45.0, 10000.0, False, 250.0, 180.0, 0.0, None,
         10000.0, "1234", False, 1),
        # Violates valid_velocity (negative velocity)
        (now, "abc123", "BAD002", "US", now, now,
         -74.0, 40.0, 9000.0, False, -50.0, 90.0, 0.0, None,
         9000.0, "5678", False, 1),
        # Violates valid_latitude (latitude > 90)
        (now, "def456", "BAD003", "DE", now, now,
         13.0, 95.5, 11000.0, False, 300.0, 45.0, 0.0, None,
         11000.0, "9999", False, 1),
        # Violates valid_longitude (longitude > 180)
        (now, "ghi789", "BAD004", "FR", now, now,
         200.0, 48.0, 8000.0, False, 280.0, 270.0, 0.0, None,
         8000.0, "4321", False, 1),
        # Violates multiple: null icao24 + negative velocity + bad lat + bad lon
        (now, None, "BAD005", None, now, now,
         185.0, -95.0, 12000.0, False, -10.0, 0.0, 0.0, None,
         12000.0, None, False, 1),
    ]

    logger.info("Injecting %d test rows that violate bronze expectations.", len(test_rows))
    return spark.createDataFrame(test_rows, schema=schema)