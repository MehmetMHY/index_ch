import itertools
import sys
import threading
import time


class Spinner:
    """Animated one-line spinner that erases itself; no-op when not a terminal."""

    def __init__(self, message=""):
        self.message = message
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        for ch in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{ch} {self.message}" if self.message else f"\r{ch} ")
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
