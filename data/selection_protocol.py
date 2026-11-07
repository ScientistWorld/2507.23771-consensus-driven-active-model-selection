"""Client side of the label-oracle protocol used by scripts/evaluate*.sh.

A selector is registered by writing ``outputs/selectors/<name>.json`` with
``{"command": ["python", "path/to/policy.py", ...]}``. The evaluator starts
that command once per episode (working directory = repository root) and
talks to it over stdin/stdout, one JSON object per line:

1. evaluator -> selector: the episode
   ``{"protocol": 1, "num_models": H, "num_items": N, "num_classes": C,
      "budget": B, "seed": s, "predicted_classes": [[...N ints...] x H]}``
2. for round t = 0, 1, ..., B the selector sends
   ``{"select": k, "query": i}``: ``k`` is the model it selects after the
   t labels seen so far; ``i`` is the next item to label, or ``null`` to
   stop querying (the selection is then held for all remaining rounds).
   At round B the query must be ``null``.
3. evaluator -> selector after each query: ``{"label": y}``.
4. evaluator -> selector at the end: ``{"type": "end"}``; then exit 0.

Only ``select`` and ``query`` are read. Anything printed to stdout that is
not a protocol message breaks the episode, so ``serve`` redirects
``print`` output to stderr while the selector runs.

``serve(factory)`` runs that loop for any object with ``select() -> int``,
``query() -> int | None`` and ``observe(item, label)`` built by
``factory(episode_dict)``. Standard library only.
"""

from __future__ import annotations

import json
import sys
from typing import Callable, Protocol


class Selector(Protocol):
    def select(self) -> int: ...

    def query(self) -> int | None: ...

    def observe(self, item: int, label: int) -> None: ...


def _read(stream) -> dict:
    line = stream.readline()
    if not line:
        raise EOFError("evaluator closed the protocol stream")
    return json.loads(line)


def serve(factory: Callable[[dict], Selector]) -> None:
    channel = sys.stdout
    sys.stdout = sys.stderr
    try:
        episode = _read(sys.stdin)
        selector = factory(episode)
        budget = int(episode["budget"])
        for round_index in range(budget + 1):
            choice = int(selector.select())
            query = selector.query() if round_index < budget else None
            query = None if query is None else int(query)
            channel.write(json.dumps({"select": choice, "query": query}) + "\n")
            channel.flush()
            if query is None:
                break
            reply = _read(sys.stdin)
            selector.observe(query, int(reply["label"]))
        try:
            _read(sys.stdin)
        except EOFError:
            pass
    finally:
        sys.stdout = channel
