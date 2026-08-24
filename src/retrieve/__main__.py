import sys

# start the spinner BEFORE importing cli (which triggers the slow numpy, openai,
# httpx, pydantic imports) so the user sees feedback immediately
from .spinner import start_startup_spinner, stop_startup_spinner

start_startup_spinner()

from .cli import main

try:
    sys.exit(main())
finally:
    stop_startup_spinner()
