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
# retrieve is a package (src/retrieve/), invoked via -m with src/ on PYTHONPATH
RETRIEVE_MODULE = "retrieve"

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
    RETRIEVE_MODULE: "Opening smart search...",
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


# "No" first so a bare Enter is safe, matching /purge's confirmation gate.
RETURN_NO = "No"
RETURN_YES = "Yes"


def confirm_return_to_menu():
    """Ask whether to go back to the main fzf menu after an action. Returns
    True for Yes, False for No, cancel, or when fzf is missing."""
    if shutil.which("fzf") is None:
        return False
    proc = subprocess.run(
        ["fzf", "--prompt=return to menu? > ", "--cycle"],
        input="\n".join([RETURN_NO, RETURN_YES]),
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == RETURN_YES


def run_scripts(scripts):
    for i, script in enumerate(scripts):
        label = SCRIPT_LABELS.get(script)
        if label:
            print(label, flush=True)
        if script is RETRIEVE_MODULE:
            cmd = f"{PY_CALL} -m {script}"
            env = os.environ.copy()
            env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
            status = subprocess.run(cmd, shell=True, env=env).returncode
        else:
            cmd = f"{PY_CALL} {script}"
            status = os.system(cmd)
        if status != 0:
            print(f"error: '{cmd}' command failed with status {status}")
            sys.exit(1)
        if i != len(scripts) - 1:
            time.sleep(0.1)
            print()


def main():
    """fzf-driven entry loop. Picks an action and runs it; loops back to the
    menu only after `Update Cache` when the user opts to return. Exits on
    `Exit Session`, a cancelled picker, or after any one-shot action."""
    while True:
        action = pick_action()
        if action == "exit":
            break
        elif action == "retrieve":
            run_scripts([RETRIEVE_MODULE])
            break
        elif action == "ls":
            # launch retrieve with a startup command so it runs /ls on launch
            print("Opening chat browser...", flush=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
            status = subprocess.run(
                [PY_CALL, "-m", RETRIEVE_MODULE, "ls"], env=env
            ).returncode
            if status != 0:
                print(f"error: retrieve ls failed with status {status}")
                sys.exit(1)
            break
        elif action == "update":
            # ask before the long-running build+process so the flow is hands-off
            return_to_menu = confirm_return_to_menu()
            run_scripts([BUILD_SCRIPT, PROCESS_SCRIPT])
            if not return_to_menu:
                break
        elif action == "update_retrieve":
            run_scripts([BUILD_SCRIPT, PROCESS_SCRIPT, RETRIEVE_MODULE])
            break


if __name__ == "__main__":
    require_ch_dirs()
    main()
    sys.exit(0)
