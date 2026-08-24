from dataclasses import dataclass, field
from typing import Any

from config import TOP_K


@dataclass
class Session:
    """Shared state for the interactive REPL, passed to every command handler.

    Replaces the 10-positional-arg search() signature and the 3-4-arg handler
    signatures. Handlers mutate attributes directly (e.g. session.time_filter =
    ...) instead of returning new values. Imports nothing internal, so it sits
    at the bottom of the dependency DAG and cannot create import cycles.
    """

    conn: Any
    ids: Any  # numpy array of chat ids
    mat: Any  # numpy matrix of normalized embeddings
    meta: dict
    last_results: list = field(default_factory=list)
    do_rerank: bool = True
    do_expand: bool = True
    show_archived: bool = False
    time_filter: Any = None  # None, a TIME_RANGES key, or (start, end) tuple
    result_len: int = TOP_K
