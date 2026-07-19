from multiprocessing import Pool
import time
import json
import os
import re


def file_paths(dir_path):
    return [
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, name))
    ]


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_tokens(text):
    if not text:
        return 0

    # count words and punctuation-like chunks
    chunks = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    # average English word is often around 1.3 tokens
    word_based_estimate = len(chunks) * 1.3

    # character-based estimate, common rough rule
    char_based_estimate = len(text) / 4

    # blend both estimates for a more stable guess
    return round((word_based_estimate + char_based_estimate) / 2)


def is_auto_entry(content):
    if not content:
        return False

    if content.startswith("File: "):
        return True

    if "=== Code Dump ===" in content or "=== FILE:" in content:
        return True

    if "The user executed the following command and here is the output:" in content:
        return True

    return False


def load_and_clean(file_path):
    raw = load_json(file_path)

    text = ""
    for msg in raw["messages"]:
        user = msg["user"] or ""
        bot = msg["bot"] or ""
        if is_auto_entry(user) or is_auto_entry(bot):
            continue
        text = text + "USER: " + user + "\n" + "BOT: " + bot + "\n"

    return {"raw": raw, "cleaned": text}


if __name__ == "__main__":
    start_time = time.time()

    ROOT_DIR = os.path.join(os.path.expanduser("~"), ".ch/tmp/")

    json_paths = [f for f in file_paths(ROOT_DIR) if f.endswith(".json")]

    with Pool() as pool:
        results = pool.map(load_and_clean, json_paths)

    data = dict(zip(json_paths, results))

    counts = sorted(
        (estimate_tokens(data[key]["cleaned"]) for key in data),
        reverse=True,
    )

    for x in counts[:10]:
        print(x)

    runtime = time.time() - start_time

    print("\nRuntime:", runtime, "seconds")
