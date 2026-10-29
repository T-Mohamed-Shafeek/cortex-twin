"""Build Graph SFT data: transcript -> concept graph JSON (nodes with status, prerequisite edges).

Node statuses come from data/graph_labels.json (Claude-annotated, keyed by row index in sft_dataset.jsonl);
concepts and edges per topic come from graph_templates.py.
"""
import json

from build_diagnoser_dataset import write_split
from generate_sft_dataset import DATA_DIR
from graph_templates import T

GRAPH_SYSTEM = (
    "You map a learner's understanding. Read the transcript of a Socratic assessment and output a concept graph as JSON only: "
    '{"topic": str, "nodes": [{"id": str, "status": "known|shaky|missing|untested"}], "edges": [[prerequisite_id, dependent_id]]}. '
    "known = demonstrated correctly; shaky = partial or hedged; missing = absent or wrong belief; untested = never probed."
)


def graph(i, row):
    topic = row["metadata"]["topic"]
    nodes, edges = T[topic]
    return {"topic": topic, "nodes": [{"id": n, "status": LABELS[str(i)][n]} for n in nodes], "edges": [list(e) for e in edges]}


if __name__ == "__main__":
    LABELS = json.loads((DATA_DIR / "graph_labels.json").read_text())
    write_split("graph", GRAPH_SYSTEM, graph)
