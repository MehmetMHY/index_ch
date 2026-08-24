import sys
import time
import threading

from build import get_connection, backfill_message_epochs, backfill_archived
from config import TOP_K, NUM_EXPANSIONS

from .spinner import Spinner, stop_startup_spinner
from .state import Session
from .cache import ensure_fts, load_vectors
from .search import search, warm_connections
from .display import print_results, format_help, time_filter_label
from .cmds.ls import handle_ls, handle_purge
from .cmds.dump import handle_dump
from .cmds.simple import (
    handle_view,
    handle_copy,
    handle_run,
    handle_time,
    handle_len,
)


def main(argv=None):
    """Entry point for the interactive search REPL."""
    argv = argv or sys.argv[1:]
    startup_cmd = argv[0].lower() if argv else None
    if startup_cmd == "ls":
        startup_cmd = "/ls"

    # `retrieve ls` is a one-shot newest->oldest fzf list. It does not need
    # embeddings, FTS, reranking, query expansion, or API warmup.
    if startup_cmd == "/ls":
        conn = get_connection()
        backfill_message_epochs(conn)
        backfill_archived(conn)
        stop_startup_spinner()
        try:
            handle_ls(conn, False, None)
        finally:
            conn.close()
        return 0

    # warm the API connections while the DB work below runs, so the first query
    # is not slowed by cold-start handshakes
    threading.Thread(target=warm_connections, daemon=True).start()

    conn = get_connection()
    backfill_message_epochs(conn)  # one-time; no-op once the column exists
    backfill_archived(conn)  # one-time; no-op once the column exists
    stop_startup_spinner()
    ensure_fts(conn)
    ids, mat, meta = load_vectors(conn)

    session = Session(
        conn=conn,
        ids=ids,
        mat=mat,
        meta=meta,
        do_rerank=True,
        do_expand=NUM_EXPANSIONS > 0,
        show_archived=False,
        time_filter=None,
        result_len=TOP_K,
    )

    n = len(ids)
    print(f"Loaded {n:,} indexed chat{'s' if n != 1 else ''}")
    print("Type a query or /help")

    while True:
        try:
            if startup_cmd:
                query = startup_cmd
                startup_cmd = None
            else:
                prompt = (
                    f"[{time_filter_label(session.time_filter)}]> "
                    if session.time_filter
                    else "> "
                )
                query = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", ":q"):
            break
        if query.lower() == ":fast":
            session.do_rerank = not session.do_rerank
            print(f"Rerank is now {'ON' if session.do_rerank else 'OFF'}.")
            continue
        if query.lower() == ":expand":
            session.do_expand = not session.do_expand
            print(f"Query expansion is now {'ON' if session.do_expand else 'OFF'}.")
            continue
        if query.lower() == ":archived":
            session.show_archived = not session.show_archived
            print(
                f"Archived chats are now "
                f"{'SHOWN' if session.show_archived else 'HIDDEN'}."
            )
            continue

        parts = query.split()
        if parts[0].lower() in ("/help", "/h"):
            print(format_help(session))
            continue
        if parts[0].lower() in ("/view", "/v"):
            handle_view(session, parts[1:])
            continue
        if parts[0].lower() in ("/copy", "/c"):
            handle_copy(session, parts[1:])
            continue
        if parts[0].lower() in ("/run", "/r"):
            handle_run(session, parts[1:])
            continue
        if parts[0].lower() in ("/dump", "/d"):
            handle_dump(session, parts[1:])
            continue
        if parts[0].lower() == "/ls":
            handle_ls(session.conn, session.show_archived, session.time_filter)
            continue
        if parts[0].lower() in ("/time", "/t"):
            handle_time(session, parts[1:])
            continue
        if parts[0].lower() in ("/len", "/l"):
            handle_len(session, parts[1:])
            continue
        if parts[0].lower() == "/purge":
            if handle_purge(session.conn):
                # row set and embeddings cache signature changed: reload
                # in-memory state and rebuild the FTS index in place
                ensure_fts(session.conn)
                session.ids, session.mat, session.meta = load_vectors(session.conn)
                session.last_results = []
            continue

        start = time.time()
        with Spinner("reranking" if session.do_rerank else "searching"):
            results, usage = search(session, query)
        print_results(results, session.meta, time.time() - start, usage)
        session.last_results = results

    conn.close()
    return 0
