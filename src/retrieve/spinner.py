import itertools
import re
import shutil
import sys
import threading
import time

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visible_len(s):
    """Length of *s* with ANSI color escapes stripped."""
    return len(_ANSI_RE.sub("", s))


def erase_wrapped_input(prompt, text):
    """Erase the prompt + input line(s) a long query occupied on screen.

    The spinner resets the cursor with \\r, which only reaches the start of
    the current line. When a query wraps across several terminal lines the
    wrapped fragments above the cursor are never cleared, so they sit on top
    of the spinner animation and corrupt the display. When the input wrapped
    we move the cursor back to the first line of the prompt and clear to the
    end of the screen so the spinner starts on a clean line. A single-line
    (non-wrapped) input is left untouched to keep the familiar prompt + query
    line visible above the spinner. No-op when stdout is not a TTY.
    """
    if not sys.stdout.isatty():
        return
    try:
        cols = shutil.get_terminal_size().columns or 80
    except OSError:
        cols = 80
    if cols <= 0:
        return
    up = (_visible_len(prompt) + len(text) + cols - 1) // cols
    if up <= 1:
        return
    sys.stdout.write(f"\033[{up}A\r\033[J")
    sys.stdout.flush()


class Spinner:
    """Animated one-line spinner that erases itself; no-op when not a terminal."""

    def __init__(self, message=""):
        self.message = message
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        yellow = "\033[93m"
        reset = "\033[0m"
        for ch in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            if self.message:
                sys.stdout.write(f"\r{yellow}{ch} {self.message}{reset}")
            else:
                sys.stdout.write(f"\r{yellow}{ch}{reset} ")
            sys.stdout.flush()
            time.sleep(0.1)

    def __enter__(self):
        if sys.stdout.isatty():
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
            width = len(self.message) + 12 if self.message else 1
            sys.stdout.write("\r" + " " * width + "\r")
            sys.stdout.flush()


# started before the slow imports (numpy, openai, httpx, pydantic) so the user
# sees feedback immediately; stopped by cli.main() after DB setup completes.
_startup_spinner = None


def start_startup_spinner(message="Starting search..."):
    global _startup_spinner
    _startup_spinner = Spinner(message)
    _startup_spinner.__enter__()


def stop_startup_spinner():
    global _startup_spinner
    if _startup_spinner is not None:
        _startup_spinner.__exit__(None, None, None)
        _startup_spinner = None
