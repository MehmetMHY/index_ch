import sys

# start the spinner BEFORE importing cli (which triggers the slow numpy, openai,
# httpx, pydantic imports) so the user sees feedback immediately
from .spinner import start_startup_spinner, stop_startup_spinner

start_startup_spinner()

try:
    from .cli import main

    sys.exit(main())
except KeyboardInterrupt:
    pass
finally:
    stop_startup_spinner()
