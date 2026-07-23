import subprocess
import shutil
import time
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.join(ROOT_DIR, "src")
BUILD_SCRIPT = os.path.join(SRC_DIR, "build.py")
PROCESS_SCRIPT = os.path.join(SRC_DIR, "process.py")
GET_SCRIPT = os.path.join(SRC_DIR, "retrieve.py")

PY_CALL = os.path.join(ROOT_DIR, "env/bin/python3")
if not os.path.isfile(PY_CALL):
    PY_CALL = "python3"

RUN_ACTIONS = [
    ("Just Retrieve", "retrieve"),
    ("Update Cache", "update"),
    ("Update & Retrieve", "both"),
    ("Exit", "exit"),
]


def pick_action():
    if shutil.which("fzf") is None:
        print("fzf not found on PATH - running the full pipeline (update + retrieve).")
        return "both"

    label_to_action = {label: action for label, action in RUN_ACTIONS}
    proc = subprocess.run(
        ["fzf", "--prompt=run> ", "--cycle"],
        input="\n".join(label for label, _ in RUN_ACTIONS),
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0 or not proc.stdout.strip():
        return "exit"

    return label_to_action.get(proc.stdout.strip(), "exit")


def run_scripts(scripts):
    for i, script in enumerate(scripts):
        cmd = f"{PY_CALL} {script}"
        status = os.system(cmd)
        if status != 0:
            print(f"error: '{cmd}' command failed with status {status}")
            sys.exit(1)
        if i != len(scripts) - 1:
            time.sleep(0.1)
            print()


if __name__ == "__main__":
    action = pick_action()
    if action == "retrieve":
        run_scripts([GET_SCRIPT])
    elif action == "update":
        run_scripts([BUILD_SCRIPT, PROCESS_SCRIPT])
    elif action == "both":
        run_scripts([BUILD_SCRIPT, PROCESS_SCRIPT, GET_SCRIPT])
    sys.exit(0)
