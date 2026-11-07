"""Malformed selectors: each mode breaks the oracle protocol in one way."""
import json
import sys

mode = sys.argv[1]
channel, sys.stdout = sys.stdout, sys.stderr
episode = json.loads(sys.stdin.readline())
budget = int(episode["budget"])


def send(text):
    channel.write(text + "\n")
    channel.flush()


if mode == "empty":
    sys.exit(0)
if mode == "nan":
    send('{"select": NaN, "query": null}')
elif mode == "negative":
    send(json.dumps({"select": -1, "query": None}))
elif mode == "null":
    send(json.dumps({"select": None, "query": None}))
elif mode == "float":
    send(json.dumps({"select": 0.0, "query": None}))
elif mode == "duplicate":
    send(json.dumps({"select": 0, "query": 0}))
    sys.stdin.readline()
    send(json.dumps({"select": 0, "query": 0}))
elif mode == "overbudget":
    for t in range(budget + 1):
        send(json.dumps({"select": 0, "query": t}))
        if not sys.stdin.readline().startswith('{"label"'):
            break
elif mode == "crash":
    send(json.dumps({"select": 0, "query": None}))
    sys.stdin.readline()
    raise SystemExit(3)
sys.stdin.readline()
