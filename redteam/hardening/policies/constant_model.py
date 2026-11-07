"""Degenerate selector: always select one fixed model index, never query."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.selection_protocol import serve  # noqa: E402


class Constant:
    def __init__(self, episode, model):
        self.model = model

    def select(self):
        return self.model

    def query(self):
        return None

    def observe(self, item, label):
        pass


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=0)
args = parser.parse_args()
serve(lambda episode: Constant(episode, args.model))
