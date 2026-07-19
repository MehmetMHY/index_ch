import time
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_SCRIPT = os.path.join(ROOT_DIR, "build.py")
PROCESS_SCRIPT = os.path.join(ROOT_DIR, "process.py")
GET_SCRIPT = os.path.join(ROOT_DIR, "retrieve.py")

PY_CALL = os.path.join(ROOT_DIR, "env/bin/python3")
if not os.path.isfile(PY_CALL):
    PY_CALL = "python3"

targets = [BUILD_SCRIPT, PROCESS_SCRIPT, GET_SCRIPT]
for i in range(len(targets)):
    cmd = f"{PY_CALL} {targets[i]}"
    status = os.system(cmd)
    if status != 0:
        print(f"error: '{cmd}' command failed with status {status}")
        sys.exit(1)
    if i != len(targets) - 1:
        time.sleep(0.1)
        print()
