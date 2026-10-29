"""
Unit tests for the silver layer (cleaned_flights).

Tests verify that the silver streaming table:
- Drops rows with null icao24 (expect_or_drop)
- Drops rows with invalid coordinates (expect_or_drop)
- Filters out negative velocity
- Keeps rows with null velocity (ground status)
- Trims whitespace from callsign and origin_country
- Adds _silver_processed_at timestamp
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


def test_silver_drops_null_icao24(test_spark):
    """Rows with null icao24 are dropped by the icao24_not_null expectation."""
    _seed_ingest_flights(test_spark, """
        ('2026-01-01T00:00:00', 'abc123', 'UAL100', 'US', '2026-01-01T00:00:00', '2026-01-01T00:00:00',
         -74.0, 40.0, 10000.0, false, 250.0, 180.0, 0.0, NULL, 10000.0, '1234', false, 1),
        ('2026-01-01T00:01:00', NULL, 'BAD001', 'XX', '2026-01-01T00:01:00', '2026-01-01T00:01:00',
         10.0, 45.0, 10000.0, false, 200.0, 90.0, 0.0, NULL, 10000.0, '5678', false, 1)
    """)

    status = test_pipeline.run(test_spark, set(["cleaned_flights"]))
    assert status.is_success, f"Pipeline run failed: {status.error_message}"

    result = test_spark.sql("SELECT icao24 FROM workspace.default.cleaned_flights ORDER BY icao24")
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0].icao24 == "abc123"


def test_silver_drops_invalid_coordinates(test_spark):
    """Rows with invalid latitude or longitude are dropped by expectations."""
    _seed_ingest_flights(test_spark, """
        ('2026-01-01T00:00:00', 'abc123', 'UAL100', 'US', '2026-01-01T00:00:00', '2026-01-01T00:00:00',
         -74.0, 40.0, 10000.0, false, 250.0, 180.0, 0.0, NULL, 10000.0, '1234', false, 1),
        ('2026-01-01T00:01:00', 'def456', 'BAD002', 'DE', '2026-01-01T00:01:00', '2026-01-01T00:01:00',
         13.0, 95.5, 11000.0, false, 300.0, 45.0, 0.0, NULL, 11000.0, '9999', false, 1),
        ('2026-01-01T00:02:00', 'ghi789', 'BAD003', 'FR', '2026-01-01T00:02:00', '2026-01-01T00:02:00',
         200.0, 48.0, 8000.0, false, 280.0, 270.0, 0.0, NULL, 8000.0, '4321', false, 1)
    """)

    status = test_pipeline.run(test_spark, set(["cleaned_flights"]))
    assert status.is_success, f"Pipeline run failed: {status.error_message}"

    result = test_spark.sql("SELECT icao24 FROM workspace.default.cleaned_flights ORDER BY icao24")
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0].icao24 == "abc123"


def test_silver_filters_negative_velocity(test_spark):
    """Rows with negative velocity are filtered out."""
    _seed_ingest_flights(test_spark, """
        ('2026-01-01T00:00:00', 'abc123', 'UAL100', 'US', '2026-01-01T00:00:00', '2026-01-01T00:00:00',
         -74.0, 40.0, 10000.0, false, 250.0, 180.0, 0.0, NULL, 10000.0, '1234', false, 1),
        ('2026-01-01T00:01:00', 'def456', 'UAL200', 'US', '2026-01-01T00:01:00', '2026-01-01T00:01:00',
         -87.0, 41.0, 9000.0, false, -50.0, 90.0, 0.0, NULL, 9000.0, '5678', false, 1)
    """)

    status = test_pipeline.run(test_spark, set(["cleaned_flights"]))
    assert status.is_success, f"Pipeline run failed: {status.error_message}"

    result = test_spark.sql("SELECT icao24, velocity FROM workspace.default.cleaned_flights ORDER BY icao24")
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0].icao24 == "abc123"
    assert rows[0].velocity == 250.0


def test_silver_keeps_null_velocity(test_spark):
    """Rows with null velocity are kept (may indicate aircraft on the ground)."""
    _seed_ingest_flights(test_spark, """
        ('2026-01-01T00:00:00', 'abc123', 'UAL100', 'US', '2026-01-01T00:00:00', '2026-01-01T00:00:00',
         -74.0, 40.0, 10000.0, false, NULL, 180.0, 0.0, NULL, 10000.0, '1234', false, 1)
    """)

    status = test_pipeline.run(test_spark, set(["cleaned_flights"]))
    assert status.is_success, f"Pipeline run failed: {status.error_message}"

    result = test_spark.sql("SELECT icao24, velocity FROM workspace.default.cleaned_flights")
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0].velocity is None


def test_silver_trims_whitespace(test_spark):
    """Whitespace in callsign and origin_country is trimmed."""
    _seed_ingest_flights(test_spark, """
        ('2026-01-01T00:00:00', 'abc123', '  UAL100  ', '  US  ', '2026-01-01T00:00:00', '2026-01-01T00:00:00',
         -74.0, 40.0, 10000.0, false, 250.0, 180.0, 0.0, NULL, 10000.0, '1234', false, 1)
    """)

    status = test_pipeline.run(test_spark, set(["cleaned_flights"]))
    assert status.is_success, f"Pipeline run failed: {status.error_message}"

    result = test_spark.sql("SELECT callsign, origin_country FROM workspace.default.cleaned_flights")
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0].callsign == "UAL100"
    assert rows[0].origin_country == "US"


def test_silver_adds_processing_timestamp(test_spark):
    """The _silver_processed_at column is added with a non-null value."""
    _seed_ingest_flights(test_spark, """
        ('2026-01-01T00:00:00', 'abc123', 'UAL100', 'US', '2026-01-01T00:00:00', '2026-01-01T00:00:00',
         -74.0, 40.0, 10000.0, false, 250.0, 180.0, 0.0, NULL, 10000.0, '1234', false, 1)
    """)

    status = test_pipeline.run(test_spark, set(["cleaned_flights"]))
    assert status.is_success, f"Pipeline run failed: {status.error_message}"

    result = test_spark.sql("SELECT _silver_processed_at FROM workspace.default.cleaned_flights")
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0]._silver_processed_at is not None