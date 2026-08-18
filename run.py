import subprocess
import shutil
import time
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CH_DIR = os.path.join(os.path.expanduser("~"), ".ch")
CHATS_SOURCE_DIR = os.path.join(CH_DIR, "tmp")

SRC_DIR = os.path.join(ROOT_DIR, "src")
BUILD_SCRIPT = os.path.join(SRC_DIR, "build.py")
PROCESS_SCRIPT = os.path.join(SRC_DIR, "process.py")
GET_SCRIPT = os.path.join(SRC_DIR, "retrieve.py")

PY_CALL = os.path.join(ROOT_DIR, "env/bin/python3")
if not os.path.isfile(PY_CALL):
    PY_CALL = "python3"

RUN_ACTIONS = [
    ("Browse Chats", "ls"),
    ("Smart Search", "retrieve"),
    ("Update Cache", "update"),
    ("Exit Session", "exit"),
]

SCRIPT_LABELS = {
    BUILD_SCRIPT: "Scanning Ch exports...",
    PROCESS_SCRIPT: "Processing pending chats...",
    GET_SCRIPT: "Opening smart search...",
}


def require_ch_dirs():
    if not os.path.isdir(CH_DIR):
        print(
            "error: ~/.ch/ does not exist. Install Ch and configure local mode first: "
            "https://github.com/MehmetMHY/ch",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.isdir(CHATS_SOURCE_DIR):
        print(
            "error: ~/.ch/tmp/ does not exist. Install Ch and configure local mode first: "
            "https://github.com/MehmetMHY/ch",
            file=sys.stderr,
        )
        sys.exit(1)


def pick_action():
    if shutil.which("fzf") is None:
        print("fzf not found on PATH - running update + retrieve.")
        return "update_retrieve"

    label_to_action = {label: action for label, action in RUN_ACTIONS}
    proc = subprocess.run(
        ["fzf", "--cycle"],
        input="\n".join(label for label, _ in RUN_ACTIONS),
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0 or not proc.stdout.strip():
        return "exit"

    return label_to_action.get(proc.stdout.strip(), "exit")


def run_scripts(scripts):
    for i, script in enumerate(scripts):
        label = SCRIPT_LABELS.get(script)
        if label:
            print(label, flush=True)
        cmd = f"{PY_CALL} {script}"
        status = os.system(cmd)
        if status != 0:
            print(f"error: '{cmd}' command failed with status {status}")
            sys.exit(1)
        if i != len(scripts) - 1:
            time.sleep(0.1)
            print()


if __name__ == "__main__":
    require_ch_dirs()
    action = pick_action()
    if action == "retrieve":
        run_scripts([GET_SCRIPT])
    elif action == "ls":
        # launch retrieve.py with a startup command so it runs /ls on launch
        print("Opening chat browser...", flush=True)
        cmd = f"{PY_CALL} {GET_SCRIPT} ls"
        status = os.system(cmd)
        if status != 0:
            print(f"error: '{cmd}' command failed with status {status}")
            sys.exit(1)
    elif action == "update":
        run_scripts([BUILD_SCRIPT, PROCESS_SCRIPT])
    elif action == "update_retrieve":
        run_scripts([BUILD_SCRIPT, PROCESS_SCRIPT, GET_SCRIPT])
    sys.exit(0)
