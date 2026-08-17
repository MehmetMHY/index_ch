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

1. **build**: reads saved Ch sessions from `~/.ch/tmp/`, strips noisy generated content (code dumps, file pastes, command output), and stores cleaned text in a local SQLite database. New files are added, changed files are re-ingested in place, and deleted source files are flagged `archived` (kept, not dropped, since you paid for their summaries and embeddings).
2. **process**: summarizes each chat (`gpt-5.4-nano`), condenses that into a 1-2 sentence blurb for the search results, and embeds the summary (`text-embedding-3-small`). Each step is skipped when its column is already filled, so re-running never redoes work you already paid for.
3. **retrieve**: an interactive search prompt. Optionally expands your query into a few variants (Groq `openai/gpt-oss-20b`), embeds the original plus variants in one batched call, runs vector search and FTS5 keyword search per query, fuses every ranking with Reciprocal Rank Fusion, reranks the top candidates (Groq `openai/gpt-oss-120b`) against the original query, and shows the top 5 with UTC timestamps from each chat's last message. Also has `/ls` (browse newest->oldest with a right-side preview), `/view`, `/copy`, `/run`, `/dump`, `/time`, `/len`, and toggle flags like `:fast`, `:expand`, `:archived`.

## Run Site

```sh
# starts a local dev server, opens the page in the browser, & press `Ctrl+C` to stop/kill it
python3 docs/run.py
```
