# index_ch

> WARNING: (August 16, 2026) This project is a rough prototype and needs further work until `index_ch` is published.

## About

`index_ch` turns your saved [Ch](https://github.com/MehmetMHY/ch) conversations into a local, searchable memory. Find past answers, decisions, debugging notes, and ideas by meaning, even when you don't remember the exact words.

## Why

Your best AI answers are easy to lose. When chats pile up, exact keyword search is not enough. You remember the idea, the bug, or the decision, but not the wording. `index_ch` retrieves the right conversation anyway.

## Features

- **semantic search**: find conversations by intent and meaning, not just matching words
- **hybrid retrieval**: meaning-based search plus keyword matching for better recall
- **concise summaries**: each chat becomes a readable preview you can scan quickly
- **local cache**: your index and generated metadata are stored on your machine
- **chat browser**: browse recent sessions and jump back in when needed
- **terminal native**: a CLI with a fuzzy-picking TUI, built for keyboard-driven use

## How It Works

1. **index**: reads saved Ch sessions and strips noisy generated content
2. **process**: builds summaries and searchable representations of each conversation
3. **retrieve**: search interactively, browse results, open old chats, copy references

## Run Site

```sh
# starts a local dev server, opens the page in the browser, & press `Ctrl+C` to stop/kill it
python3 run.py
```

## Sources/Credits

- [Qwen Studio](https://chat.qwen.ai/) by [Qwen](https://qwen.ai/)
- [ChatGPT](https://chatgpt.com/) for image generation and logos
- [tinygrad.org](https://tinygrad.org/)
