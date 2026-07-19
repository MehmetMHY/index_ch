import subprocess
import time
import sys
import os


def run(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return None


# main function calls

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_SCRIPT = os.path.join(ROOT_DIR, "build.py")
PROCESS_SCRIPT = os.path.join(ROOT_DIR, "process.py")
GET_SCRIPT = os.path.join(ROOT_DIR, "retrieve.py")

targets = [BUILD_SCRIPT, PROCESS_SCRIPT, GET_SCRIPT]
for t in targets:
    cmd = f"python3 {t}"
    out = run(cmd)
    if out == None:
        print("error, existing...")
    
