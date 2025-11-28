import sys
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exogena_loader import _build_dummy_series, _to_date


class DateNormalizationTests(unittest.TestCase):
    def test_datetime_converts_to_date(self):
        dt = datetime(2024, 6, 3, 12, 30)
        self.assertEqual(_to_date(dt), date(2024, 6, 3))

    def test_date_passed_through(self):
        raw_date = date(2024, 6, 4)
        self.assertEqual(_to_date(raw_date), raw_date)

    def test_timestamp_converts_to_date(self):
        timestamp = pd.Timestamp("2024-06-05T00:00:00")
        self.assertEqual(_to_date(timestamp), date(2024, 6, 5))

    def test_string_parses_correctly(self):
        self.assertEqual(_to_date("2024-06-06"), date(2024, 6, 6))

    def test_invalid_string_returns_none(self):
        self.assertIsNone(_to_date("nope"))


class DateMappingTests(unittest.TestCase):
    def test_mapping_matches_date_key(self):
        source_values = {datetime(2024, 6, 7, 9, 0).date(): 42}
        idx = pd.date_range("2024-06-05", "2024-06-09")
        mapped = [source_values.get(_to_date(ts), 0) for ts in idx]
        self.assertEqual(mapped[2], 42)

    def test_mapping_defaults_to_zero(self):
        source_values: dict[date, int] = {}
        idx = pd.date_range("2024-06-01", "2024-06-03")
        mapped = [source_values.get(_to_date(ts), 0) for ts in idx]
        self.assertListEqual(mapped, [0, 0, 0])


class DummyRangesTests(unittest.TestCase):
    def test_multiple_ranges_are_combined(self):
        idx = pd.date_range("2024-01-01", "2024-01-07")
        series = _build_dummy_series(
            idx,
            [
                (date(2024, 1, 2), date(2024, 1, 3)),
                (date(2024, 1, 5), date(2024, 1, 6)),
            ],
        )
        self.assertListEqual(series.tolist(), [0, 1, 1, 0, 1, 1, 0])

    def test_invalid_ranges_keep_default_zero(self):
        idx = pd.date_range("2024-01-01", "2024-01-03")
        series = _build_dummy_series(idx, [(None, None)])
        self.assertListEqual(series.tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
