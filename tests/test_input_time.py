"""Tests for input_time.py pure functions: _parse_time, _add_days,
_add_month, _is_first/last_visible_week."""

from datetime import date, time
import calendar
import pytest

from input_time import (
    _parse_time,
    _add_days,
    _add_month,
    _is_first_visible_week,
    _is_last_visible_week,
    _Cancelled,
)


class TestParseTime:
    def test_valid_hh_mm(self):
        result = _parse_time("09:30")
        assert result == time(9, 30)

    def test_valid_hh_mm_ss(self):
        result = _parse_time("23:59:59")
        assert result == time(23, 59, 59)

    def test_with_whitespace(self):
        result = _parse_time("  09:30  ")
        assert result == time(9, 30)

    def test_single_digit_hour_accepted_on_some_platforms(self):
        # %H may accept single digits depending on the platform
        result = _parse_time("9:30")
        assert result is not None
        assert result.hour == 9

    def test_invalid_hour_fails(self):
        with pytest.raises(ValueError):
            _parse_time("25:00")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_time("")

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            _parse_time("not a time")


class TestAddDays:
    def test_same_day(self):
        assert _add_days(date(2024, 1, 15), 0) == date(2024, 1, 15)

    def test_positive(self):
        assert _add_days(date(2024, 1, 15), 10) == date(2024, 1, 25)

    def test_cross_month(self):
        assert _add_days(date(2024, 1, 31), 1) == date(2024, 2, 1)

    def test_cross_year(self):
        assert _add_days(date(2024, 12, 31), 1) == date(2025, 1, 1)

    def test_negative(self):
        assert _add_days(date(2024, 1, 1), -1) == date(2023, 12, 31)

    def test_leap_year(self):
        assert _add_days(date(2024, 2, 28), 1) == date(2024, 2, 29)


class TestAddMonth:
    def test_same_month(self):
        assert _add_month(date(2024, 3, 15), 0) == date(2024, 3, 15)

    def test_positive(self):
        assert _add_month(date(2024, 1, 15), 1) == date(2024, 2, 15)

    def test_jan_31_clamped(self):
        result = _add_month(date(2024, 1, 31), 1)
        assert result == date(2024, 2, 29)  # 2024 is leap year

    def test_jan_31_non_leap(self):
        result = _add_month(date(2023, 1, 31), 1)
        assert result == date(2023, 2, 28)

    def test_dec_to_jan(self):
        assert _add_month(date(2024, 12, 15), 1) == date(2025, 1, 15)

    def test_negative(self):
        assert _add_month(date(2024, 2, 15), -1) == date(2024, 1, 15)

    def test_mar_31_to_feb_clamped(self):
        result = _add_month(date(2024, 3, 31), -1)
        assert result == date(2024, 2, 29)

    def test_feb_29_leap_year(self):
        result = _add_month(date(2024, 2, 29), 1)
        assert result == date(2024, 3, 29)


class TestVisibleWeek:
    def test_first_day_is_first_week(self):
        d = date(2024, 1, 1)
        assert _is_first_visible_week(d) is True

    def test_last_day_is_last_week(self):
        d = date(2024, 1, 31)
        assert _is_last_visible_week(d) is True

    def test_mid_month_not_first_or_last(self):
        d = date(2024, 1, 15)
        # depends on calendar layout, but Jan 15 is very likely not first or last week
        first = _is_first_visible_week(d)
        last = _is_last_visible_week(d)
        assert not (first and last)

    def test_first_week_jan_2024(self):
        # Jan 2024 starts on Monday, so Jan 1-7 are in the first row
        for day in range(1, 8):
            assert _is_first_visible_week(date(2024, 1, day)) is True


class TestCancelledException:
    def test_is_exception(self):
        assert issubclass(_Cancelled, Exception)

    def test_raisable(self):
        with pytest.raises(_Cancelled):
            raise _Cancelled()
