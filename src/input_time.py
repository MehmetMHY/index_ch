"""
Interactive UTC calendar-based time range picker for the terminal.

Public API:
    pick_time_range() -> (start, end) | None

Usage:
    from input_time import pick_time_range
    result = pick_time_range()
    if result:
        start, end = result
        print(start, end)
"""

from __future__ import annotations

import calendar
import curses
from datetime import date, datetime, time, timezone
from typing import Optional, Tuple

# Layout constants
CAL_WIDTH = 21
CONTROL_ROW = "◀ Month ▶   ◀ Year ▶"
ARROW_POSITIONS = [i for i, c in enumerate(CONTROL_ROW) if c in "◀▶"]
ARROW_DELTAS = (-1, 1, -12, 12)
STATUS_WIDTH = len("0000-00-00 00:00 UTC")

TITLE_Y = 3
CONTROL_Y = 4
WEEKDAY_Y = 5
MIN_HEIGHT = 16
MIN_WIDTH = 24


# Exceptions
class _Cancelled(Exception):
    """Internal: user cancelled via Ctrl-C / Ctrl-D / Q."""


# Public API
def pick_time_range() -> Optional[Tuple[datetime, datetime]]:
    """Open an interactive UTC calendar picker in the terminal.

    Returns (start, end) as timezone-aware UTC datetimes, or None if the
    user cancelled. Safe to call multiple times. All terminal state is
    restored even on error.
    """
    try:
        return curses.wrapper(_picker)
    except (curses.error, _Cancelled):
        return None


# Internal helpers
def _safe_addstr(
    stdscr, y: int, x: int, text: str, attr: int = curses.A_NORMAL
) -> None:
    """addstr that silently skips out-of-bounds writes (small terminals)."""
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h or x < 0 or x >= w:
        return
    stdscr.addstr(y, x, text[: max(0, w - x - 1)], attr)


def _parse_time(value: str) -> time:
    value = value.strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            pass
    raise ValueError("Use HH:MM, e.g. 09:30")


def _prompt(stdscr, y: int, text: str, x: int = 0) -> str:
    h, w = stdscr.getmaxyx()
    curses.echo()
    curses.curs_set(1)
    try:
        stdscr.move(y, 0)
        stdscr.clrtoeol()
        stdscr.addstr(y, x, text)
        stdscr.refresh()
        value = stdscr.getstr(y, x + len(text), 20).decode().strip()
    except (KeyboardInterrupt, EOFError) as exc:
        raise _Cancelled from exc
    finally:
        curses.noecho()
        curses.curs_set(0)

    if "\x03" in value or "\x04" in value:
        raise _Cancelled

    return value


def _add_days(d: date, days: int) -> date:
    return date.fromordinal(d.toordinal() + days)


def _add_month(d: date, delta: int) -> date:
    month = d.month + delta
    year = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))


def _is_first_visible_week(d: date) -> bool:
    return any(d.day in week for week in calendar.monthcalendar(d.year, d.month)[:1])


def _is_last_visible_week(d: date) -> bool:
    return any(d.day in week for week in calendar.monthcalendar(d.year, d.month)[-1:])


def _origin_x(stdscr) -> int:
    _, w = stdscr.getmaxyx()
    content_w = max(
        len(datetime.now(timezone.utc).strftime("%A, %B %d %Y")),
        len(CONTROL_ROW),
        CAL_WIDTH,
        STATUS_WIDTH,
    )
    return max(0, (w - content_w) // 2)


def _ask_time(stdscr, default: time, ox: int = 0) -> time:
    h, _ = stdscr.getmaxyx()

    while True:
        try:
            value = _prompt(stdscr, h - 1, "Time [HH:MM]: ", ox)
            if not value:
                return default
            return _parse_time(value)
        except ValueError as exc:
            _prompt(stdscr, h - 1, f"{exc}. Press Enter to retry.", ox)


def _confirm(stdscr) -> bool:
    """Return True to confirm, False to re-edit, or raise _Cancelled to exit."""
    h, _ = stdscr.getmaxyx()
    msg = "Confirm range? [Y/n]:"
    x = _origin_x(stdscr)
    curses.curs_set(1)
    stdscr.move(h - 1, 0)
    stdscr.clrtoeol()
    stdscr.addstr(h - 1, x, msg)
    stdscr.refresh()
    try:
        key = stdscr.getch()
    except (KeyboardInterrupt, EOFError) as exc:
        raise _Cancelled from exc
    finally:
        curses.curs_set(0)

    if key in (ord("q"), ord("Q"), 27, 3, 4):
        raise _Cancelled
    return key in (10, 13, curses.KEY_ENTER, ord("y"), ord("Y"))


# Drawing
def _draw(
    stdscr,
    shown: date,
    cursor: date,
    start_dt,
    end_dt,
    focus: str,
    control_idx: int,
) -> None:
    stdscr.clear()
    h, w = stdscr.getmaxyx()

    if h < MIN_HEIGHT or w < MIN_WIDTH:
        _safe_addstr(
            stdscr,
            0,
            0,
            f"Terminal too small (need {MIN_WIDTH}x{MIN_HEIGHT})",
            curses.A_BOLD,
        )
        stdscr.refresh()
        return

    now_utc = datetime.now(timezone.utc)
    today = now_utc.strftime("%A, %B %d %Y")
    tz_label = now_utc.strftime("%H:%M UTC")
    cal = calendar.monthcalendar(shown.year, shown.month)
    content_w = max(len(today), len(CONTROL_ROW), CAL_WIDTH, STATUS_WIDTH)
    content_h = WEEKDAY_Y + 1 + len(cal) + 1 + 2 + 1
    ox = max(0, (w - content_w) // 2)
    oy = max(0, (h - content_h) // 2)

    _safe_addstr(stdscr, oy, ox, today, curses.A_BOLD)
    _safe_addstr(stdscr, oy + 1, ox, tz_label, curses.A_DIM)

    title = shown.strftime("%B %Y")
    title_x = ox + max(0, (CAL_WIDTH - len(title)) // 2)
    _safe_addstr(stdscr, oy + TITLE_Y, title_x, title, curses.A_BOLD)

    _safe_addstr(stdscr, oy + CONTROL_Y, ox, CONTROL_ROW)
    if focus == "controls":
        pos = ARROW_POSITIONS[control_idx]
        _safe_addstr(
            stdscr, oy + CONTROL_Y, ox + pos, CONTROL_ROW[pos], curses.A_REVERSE
        )

    _safe_addstr(stdscr, oy + WEEKDAY_Y, ox, "Mo Tu We Th Fr Sa Su", curses.A_BOLD)

    for row_i, week in enumerate(cal):
        y = oy + WEEKDAY_Y + 1 + row_i
        for col_i, day in enumerate(week):
            x = ox + col_i * 3
            if day == 0:
                _safe_addstr(stdscr, y, x, "  ")
                continue

            current = date(shown.year, shown.month, day)
            attr = curses.A_NORMAL

            if current == cursor and focus == "calendar":
                attr |= curses.A_REVERSE
            if start_dt and current == start_dt.date():
                attr |= curses.A_BOLD
            if end_dt and current == end_dt.date():
                attr |= curses.A_UNDERLINE
            if start_dt and end_dt and start_dt.date() <= current <= end_dt.date():
                attr |= curses.A_DIM

            _safe_addstr(stdscr, y, x, f"{day:2}", attr)

    status_y = oy + WEEKDAY_Y + 1 + len(cal) + 1
    start_val = (
        start_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if start_dt
        else "missing start time"
    )
    _safe_addstr(stdscr, status_y, ox, start_val)
    end_val = (
        end_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if end_dt
        else "missing end time"
    )
    _safe_addstr(stdscr, status_y + 1, ox, end_val)

    stdscr.refresh()


# Main picker loop
def _picker(stdscr):
    curses.curs_set(0)
    stdscr.keypad(True)

    cursor = datetime.now(timezone.utc).date()
    shown = cursor.replace(day=1)
    focus = "calendar"
    control_idx = 0

    start_dt = None
    end_dt = None

    while True:
        _draw(stdscr, shown, cursor, start_dt, end_dt, focus, control_idx)
        try:
            key = stdscr.getch()
        except (KeyboardInterrupt, EOFError):
            return None

        if key in (ord("q"), ord("Q"), 27, 3, 4):
            return None

        if key in (9, ord("\t")):
            focus = "calendar" if focus == "controls" else "controls"
            if focus == "controls":
                control_idx = 0

        if focus == "controls":
            if key == curses.KEY_LEFT:
                control_idx = max(0, control_idx - 1)
            elif key == curses.KEY_RIGHT:
                control_idx = min(len(ARROW_DELTAS) - 1, control_idx + 1)
            elif key in (10, 13, curses.KEY_ENTER):
                cursor = _add_month(cursor, ARROW_DELTAS[control_idx])
        else:
            if key == curses.KEY_LEFT:
                cursor = _add_days(cursor, -1)
            elif key == curses.KEY_RIGHT:
                cursor = _add_days(cursor, 1)
            elif key == curses.KEY_UP:
                if not _is_first_visible_week(cursor):
                    cursor = _add_days(cursor, -7)
            elif key == curses.KEY_DOWN:
                if not _is_last_visible_week(cursor):
                    cursor = _add_days(cursor, 7)
            elif key in (10, 13, curses.KEY_ENTER):
                ox = _origin_x(stdscr)
                if start_dt is None or end_dt is not None:
                    t = _ask_time(stdscr, time(0, 0), ox)
                    start_dt = datetime.combine(cursor, t, tzinfo=timezone.utc)
                    end_dt = None
                else:
                    t = _ask_time(stdscr, time(23, 59), ox)
                    candidate = datetime.combine(cursor, t, tzinfo=timezone.utc)

                    if candidate < start_dt:
                        h, _ = stdscr.getmaxyx()
                        _prompt(
                            stdscr, h - 1, "End cannot be before start. Press Enter."
                        )
                    else:
                        end_dt = candidate
                        _draw(
                            stdscr, shown, cursor, start_dt, end_dt, focus, control_idx
                        )
                        if _confirm(stdscr):
                            return start_dt, end_dt

        shown = cursor.replace(day=1)


# CLI demo
if __name__ == "__main__":
    result = pick_time_range()

    if result is None:
        print("Cancelled")
    else:
        start, end = result
        print(f"Start: {start}")
        print(f"End:   {end}")
