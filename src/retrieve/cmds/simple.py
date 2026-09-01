import shutil
import subprocess
from datetime import datetime, timezone

from config import TIME_RANGES
from input_time import pick_time_range

from ..display import time_filter_desc, TIME_LABELS
from ..pickers import resolve_pick, confirm_return
from ..actions import view_chat, copy_chat, run_chat
from ..state import Session
from .. import color


def handle_view(session: Session, args):
    cid = resolve_pick(args, session.last_results, session.meta, "/view")
    if cid is not None:
        view_chat(session.conn, cid, session.meta)


def handle_copy(session: Session, args):
    cid = resolve_pick(args, session.last_results, session.meta, "/copy")
    if cid is not None:
        copy_chat(cid, session.meta)


def handle_run(session: Session, args):
    cid = resolve_pick(args, session.last_results, session.meta, "/run")
    if cid is None:
        return True
    if shutil.which("ch") is None:
        print(color.red("ch not found on PATH - https://github.com/MehmetMHY/ch"))
        return True
    ret = confirm_return("return to search? > ")
    if ret is None:
        return True
    run_chat(cid, session.meta, reprint_prompt=ret)
    return ret


# /time: scope searches to a time window
TIME_USAGE = f"/time <{'|'.join(TIME_RANGES)}|all|custom>"


def parse_time_token(token):
    """Map a token to (action, value). action is 'set' (value is a key or None),
    'custom' (launch the picker), or 'error' (unknown token)."""
    t = token.lower()
    if t in ("all", "off", "none"):
        return "set", None
    if t == "custom":
        return "custom", None
    if t in TIME_RANGES:
        return "set", t
    return "error", None


def pick_time_with_fzf():
    """fzf-pick a time window. Returns (action, value) like parse_time_token,
    plus 'cancel' when fzf is missing or the pick was cancelled."""
    if shutil.which("fzf") is None:
        print(color.red(f"fzf not found on PATH - install it, or use '{TIME_USAGE}'."))
        return "cancel", None

    # ordered (value, label); None = all time, "custom" opens the calendar picker
    options = [(None, "All time")]
    options += [(k, TIME_LABELS.get(k, k).capitalize()) for k in TIME_RANGES]
    options.append(("custom", "Custom"))

    label_to_value = {label: value for value, label in options}
    proc = subprocess.run(
        ["fzf", "--prompt=time> ", "--cycle"],
        input="\n".join(label for _, label in options),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return "cancel", None

    value = label_to_value.get(proc.stdout.strip())
    if value == "custom":
        return "custom", None
    return "set", value


def run_custom_picker():
    """Open the calendar picker; return an absolute (start_epoch, end_epoch)
    tuple, or None if the user cancelled."""
    result = pick_time_range()
    if result is None:
        return None
    start, end = result
    return (start.timestamp(), end.timestamp())


def handle_time(session: Session, args):
    """Set session.time_filter after a /time command."""
    if args:
        action, value = parse_time_token(args[0])
        if action == "error":
            print(color.yellow(f"Usage: {TIME_USAGE}"))
            return
    else:
        action, value = pick_time_with_fzf()

    if action == "cancel":
        return
    if action == "custom":
        value = run_custom_picker()
        if value is None:
            return

    print(color.green(f"Time filter set to {time_filter_desc(value)}."))
    session.time_filter = value


# /len: how many results to show per search
RESULT_LEN_MIN = 1
RESULT_LEN_MAX = 25
LEN_USAGE = f"/len <{RESULT_LEN_MIN}-{RESULT_LEN_MAX}>"


def handle_len(session: Session, args):
    """Set session.result_len after a /len command."""
    current = session.result_len
    if not args:
        print(
            color.yellow(
                f"Showing {current} result{'s' if current != 1 else ''}. Usage: {LEN_USAGE}"
            )
        )
        return

    try:
        n = int(args[0])
    except ValueError:
        print(color.yellow(f"Usage: {LEN_USAGE}"))
        return
    if not (RESULT_LEN_MIN <= n <= RESULT_LEN_MAX):
        print(color.yellow(f"Usage: {LEN_USAGE}"))
        return

    print(color.green(f"Result count set to {n}."))
    session.result_len = n
