"""
GOLD LAYER — Flight Statistics & History
========================================

This module computes aggregate flight statistics from the silver layer
(cleaned_flights) and maintains a historical record of each recompute in
a separate Delta table via a foreach_batch sink.

Architecture decisions
----------------------
1. MATERIALIZED VIEW (not Streaming Table):
   flight_stats is an aggregation (COUNT, COUNT DISTINCT, MAX) over the
   full cleaned_flights dataset. Materialized views use batch reads
   (spark.read) and support incremental refresh on serverless pipelines —
   only the delta changes since the last refresh are processed. A
   streaming table would require a streaming aggregation with watermarks
   and would produce intermediate windowed results rather than a single
   cumulative snapshot.
   Alternative: A streaming table with outputMode("complete") and a
   watermark would be more complex and is not suited for a simple stats
   table that is recomputed each run.

2. FOREACH_BATCH SINK FOR HISTORY:
   The flight_stats MV is overwritten on each update, so it cannot
   accumulate history. To track how stats evolve over time, a
   foreach_batch sink appends a new row (same stats + timestamp) to an
   external Delta table (workspace.default.flight_stats_history) every
   time new data is processed.
   Alternative: An append flow with once=True would run only once per
   pipeline lifetime (not per update) — unsuitable for per-run snapshot
   tracking. A streaming table cannot accept appends into a materialized
   view's published UC object (it is a view, not a table).

3. SINK READS FROM SILVER (not Bronze):
   Both the MV and the history sink read from cleaned_flights (silver),
   not ingest_flights (bronze). This ensures statistics are computed on
   validated, deduplicated data, not raw data that may contain nulls,
   duplicates, or invalid coordinates.
   Alternative: Reading from bronze would inflate counts (duplicates) and
   include records with null icao24, making distinct_aircraft unreliable.

4. TRY/CATCH IN SINK HANDLER:
   Unlike dataset definitions (where SDP handles failures), the
   foreach_batch_sink handler runs custom SQL (CREATE TABLE, INSERT).
   These SQL operations can fail for reasons the framework does not
   catch (e.g., permission issues, schema drift). A try/catch block
   with logging ensures errors are visible in pipeline logs and re-raised
   so the framework can retry the batch.

5. LOGGING:
   Module-level and sink-handler-level logging provides end-to-end
   visibility into gold layer execution.

6. NO TRY/CATCH IN MV DEFINITION:
   The materialized view definition is a pure DataFrame transformation.
   SDP handles failures and retries automatically. Adding try/catch
   here would mask errors from the framework.
"""

import logging

from pyspark import pipelines as dp
from pyspark.sql.functions import count, countDistinct, max, current_timestamp

# Module-level logger for the gold layer.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GOLD MATERIALIZED VIEW: flight_stats
# ---------------------------------------------------------------------------
# Recomputed on every pipeline run. Provides the current cumulative flight
# statistics across all cleaned data:
#   - num_events: total flight position records
#   - distinct_aircraft: unique aircraft (by icao24 transponder code)
#   - max_velocity: highest recorded velocity
#
# Published to workspace.default.flight_stats.
#
# On serverless pipelines this MV supports incremental refresh — only
# changes in cleaned_flights since the last update are processed, rather
# than recomputing the full aggregation every time.
# ---------------------------------------------------------------------------
@dp.materialized_view(
    comment=(
        "Gold layer: cumulative flight statistics from cleaned silver data. "
        "Materialized view recomputed on each update with incremental refresh."
    ),
)
def flight_stats():
    """
    Compute aggregate flight statistics from the silver layer.

    Reads cleaned_flights (batch read) and returns a single-row DataFrame
    with total event count, distinct aircraft count, and max velocity.
    """
    logger.info("Computing gold layer flight statistics from silver data.")
    df = spark.read.table("cleaned_flights")
    return (
        df.agg(
            count("*").alias("num_events"),
            countDistinct("icao24").alias("distinct_aircraft"),
            max("velocity").alias("max_velocity"),
        )
    )


# ---------------------------------------------------------------------------
# HISTORY TRACKING: append flow + foreach_batch sink
# ---------------------------------------------------------------------------
# The append_flow feeds new silver data into the sink, triggering the
# foreach_batch handler on each micro-batch. The handler recomputes the
# same statistics and appends them (with a UTC timestamp) to the external
# history table. This creates a time-series of cumulative snapshots.
# ---------------------------------------------------------------------------

# The append flow streams from the silver layer. Each micro-batch of new
# cleaned data triggers the sink handler, which appends a fresh stats row.
# Alternative: streaming from bronze would trigger on raw (uncleaned) data;
# we read silver so that history rows reflect the same data quality as
# flight_stats.
@dp.append_flow(target="flight_stats_history_writer")
def flight_stats_history_stream():
    """Stream cleaned flight data to trigger history snapshots on each batch."""
    logger.info("Starting history append flow from silver layer.")
    return spark.readStream.table("cleaned_flights")


# The foreach_batch sink handler. For each micro-batch of new data:
#   1. Recompute cumulative stats from the full cleaned_flights table
#   2. Append the stats with a UTC timestamp to the history table
#
# The sink runtime on serverless blocks DataFrameWriter calls (Py4J security
# whitelist), so writes are done via SQL (CREATE TABLE IF NOT EXISTS + INSERT).
#
# Try/catch is used here because the handler executes custom SQL that can
# fail for reasons outside SDP's automatic retry (e.g., schema drift,
# permissions). The error is logged and re-raised so the framework can retry.
@dp.foreach_batch_sink(name="flight_stats_history_writer")
def flight_stats_history_writer(df, batch_id):
    """
    Append a cumulative stats snapshot to the history table on each batch.

    Parameters:
        df: micro-batch DataFrame from the append flow (not used directly —
            we recompute stats from the full silver table to get cumulative
            values, not just the batch's incremental values).
        batch_id: integer micro-batch ID (used for logging).
    """
    spark_session = df.sparkSession
    logger.info(f"Processing history snapshot for batch_id={batch_id}.")

    try:
        # Recompute cumulative statistics from the FULL silver table, not
        # just the current micro-batch. This ensures each history row is a
        # complete snapshot, not an incremental delta.
        # Alternative: Using the batch DataFrame (df) would give per-batch
        # stats, which are less useful for trend analysis.
        stats = (
            spark_session.read.table("cleaned_flights")
            .agg(
                count("*").alias("num_events"),
                countDistinct("icao24").alias("distinct_aircraft"),
                max("velocity").alias("max_velocity"),
            )
            .withColumn("added_at", current_timestamp())
        )
        stats.createOrReplaceTempView("flight_stats_batch")

        # Create the history table if it does not yet exist.
        # CREATE TABLE IF NOT EXISTS is idempotent — safe on every batch.
        # Timestamps are stored in plain UTC per project convention.
        spark_session.sql(
            "CREATE TABLE IF NOT EXISTS workspace.default.flight_stats_history "
            "(num_events BIGINT, distinct_aircraft BIGINT, "
            "max_velocity DOUBLE, added_at TIMESTAMP)"
        )

        # Insert the current cumulative snapshot as a new row.
        # This is an append-only operation — historical rows are never
        # modified or deleted, preserving the full timeline of stats.
        spark_session.sql(
            "INSERT INTO workspace.default.flight_stats_history "
            "SELECT num_events, distinct_aircraft, max_velocity, added_at "
            "FROM flight_stats_batch"
        )
        logger.info(
            f"History snapshot written successfully for batch_id={batch_id}."
        )

    except Exception as e:
        # Log the full error and re-raised so SDP can retry the batch.
        # Swallowing the error would cause silent data loss in the history
        # table with no indication of failure.
        logger.error(
            f"Failed to write history snapshot for batch_id={batch_id}: {e}"
        )
        raise