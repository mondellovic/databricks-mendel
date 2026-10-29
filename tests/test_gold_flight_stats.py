"""
Unit tests for the gold layer (flight_stats materialized view).

Tests verify that the gold MV correctly computes:
- num_events: total row count from cleaned_flights
- distinct_aircraft: count of distinct icao24 values
- max_velocity: maximum velocity value
- Handles empty input gracefully

Dummy data is seeded inline in each test. Tests seed bronze and refresh
silver + gold together: refreshing flight_stats alone triggers the
foreach_batch sink setup, which is not supported under test schema
redirection.
"""

from pyspark.pipelines.testing import TestPipeline, test_spark

test_pipeline = TestPipeline.active()

# Schema for the bronze ingest_flights table (mocked as a regular table in tests).
INGEST_SCHEMA = """
    time_ingest TIMESTAMP, icao24 STRING, callsign STRING,
    origin_country STRING, time_position TIMESTAMP, last_contact TIMESTAMP,
    longitude DOUBLE, latitude DOUBLE, geo_altitude DOUBLE,
    on_ground BOOLEAN, velocity DOUBLE, true_track DOUBLE,
    vertical_rate DOUBLE, sensors ARRAY<INT>,
    baro_altitude DOUBLE, squawk STRING, spi BOOLEAN, category INT
"""


def _seed_ingest_flights(test_spark, rows_sql=""):
    """Create and populate the mock bronze table."""
    test_spark.sql(
        f"CREATE OR REPLACE TABLE workspace.default.ingest_flights ({INGEST_SCHEMA})"
    )
    if rows_sql:
        test_spark.sql(
            f"INSERT INTO workspace.default.ingest_flights VALUES {rows_sql}"
        )


def test_gold_counts_events(test_spark):
    """num_events reflects the total row count from cleaned_flights."""
    _seed_ingest_flights(test_spark, """
        ('2026-01-01T00:00:00', 'abc123', 'UAL100', 'US', '2026-01-01T00:00:00', '2026-01-01T00:00:00',
         -74.0, 40.0, 10000.0, false, 250.0, 180.0, 0.0, NULL, 10000.0, '1234', false, 1),
        ('2026-01-01T00:01:00', 'def456', 'UAL200', 'US', '2026-01-01T00:01:00', '2026-01-01T00:01:00',
         -87.0, 41.0, 9000.0, false, 300.0, 90.0, 0.0, NULL, 9000.0, '5678', false, 1)
    """)

    status = test_pipeline.run(test_spark, set(["cleaned_flights", "flight_stats"]))
    assert status.is_success, f"Pipeline run failed: {status.error_message}"

    result = test_spark.sql("SELECT num_events FROM workspace.default.flight_stats")
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0].num_events == 2


def test_gold_counts_distinct_aircraft(test_spark):
    """distinct_aircraft counts unique icao24 values."""
    _seed_ingest_flights(test_spark, """
        ('2026-01-01T00:00:00', 'abc123', 'UAL100', 'US', '2026-01-01T00:00:00', '2026-01-01T00:00:00',
         -74.0, 40.0, 10000.0, false, 250.0, 180.0, 0.0, NULL, 10000.0, '1234', false, 1),
        ('2026-01-01T00:01:00', 'abc123', 'UAL101', 'US', '2026-01-01T00:01:00', '2026-01-01T00:01:00',
         -74.0, 40.0, 10000.0, false, 260.0, 180.0, 0.0, NULL, 10000.0, '1234', false, 1),
        ('2026-01-01T00:02:00', 'def456', 'UAL200', 'US', '2026-01-01T00:02:00', '2026-01-01T00:02:00',
         -87.0, 41.0, 9000.0, false, 300.0, 90.0, 0.0, NULL, 9000.0, '5678', false, 1)
    """)

    status = test_pipeline.run(test_spark, set(["cleaned_flights", "flight_stats"]))
    assert status.is_success, f"Pipeline run failed: {status.error_message}"

    result = test_spark.sql("SELECT distinct_aircraft FROM workspace.default.flight_stats")
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0].distinct_aircraft == 2


def test_gold_max_velocity(test_spark):
    """max_velocity returns the highest velocity value."""
    _seed_ingest_flights(test_spark, """
        ('2026-01-01T00:00:00', 'abc123', 'UAL100', 'US', '2026-01-01T00:00:00', '2026-01-01T00:00:00',
         -74.0, 40.0, 10000.0, false, 250.0, 180.0, 0.0, NULL, 10000.0, '1234', false, 1),
        ('2026-01-01T00:01:00', 'def456', 'UAL200', 'US', '2026-01-01T00:01:00', '2026-01-01T00:01:00',
         -87.0, 41.0, 9000.0, false, 300.0, 90.0, 0.0, NULL, 9000.0, '5678', false, 1)
    """)

    status = test_pipeline.run(test_spark, set(["cleaned_flights", "flight_stats"]))
    assert status.is_success, f"Pipeline run failed: {status.error_message}"

    result = test_spark.sql("SELECT max_velocity FROM workspace.default.flight_stats")
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0].max_velocity == 300.0


def test_gold_empty_table(test_spark):
    """Gold MV handles an empty cleaned_flights table gracefully."""
    _seed_ingest_flights(test_spark)  # no rows

    status = test_pipeline.run(test_spark, set(["cleaned_flights", "flight_stats"]))
    assert status.is_success, f"Pipeline run failed: {status.error_message}"

    result = test_spark.sql(
        "SELECT num_events, distinct_aircraft, max_velocity FROM workspace.default.flight_stats"
    )
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0].num_events == 0
    assert rows[0].distinct_aircraft == 0
    assert rows[0].max_velocity is None