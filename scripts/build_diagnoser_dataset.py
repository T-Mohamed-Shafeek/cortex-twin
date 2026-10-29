"""Build Diagnoser SFT data: transcript -> {"topic", "misconception"} JSON. Same topic split as the Prober."""
import json
import re

from generate_sft_dataset import DATA_DIR, held_out_topics, load_rows

DIAG_SYSTEM = (
    "You are a diagnostic analyst. Read the transcript of a Socratic assessment and name the learner's "
    "underlying misconception. Reply with JSON only: {\"topic\": str, \"misconception\": str}. "
    "If the learner's model is accurate, misconception starts with 'none:'. "
    "If it has not been revealed yet, it starts with 'unknown:'."
)
THOUGHT = re.compile(r"<thought>.*?</thought>\s*", re.S)


def transcript(msgs):
    """Flatten a dialogue into one 'Learner: ... / Assessor: ...' text, without the private thoughts."""
    lines = []
    for m in msgs:
        if m["role"] == "user":
            lines.append(f"Learner: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"Assessor: {THOUGHT.sub('', m['content']).strip()}")
    return "\n\n".join(lines)


def write_split(prefix, system, target):
    """Write data/{prefix}_train.jsonl and _eval.jsonl; target(row_index, row) -> gold dict for the assistant turn."""
    rows = load_rows()
    held = held_out_topics(rows)
    for split, pick in (("train", lambda t: t not in held), ("eval", lambda t: t in held)):
        out = [{"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": transcript(r["messages"])},
            {"role": "assistant", "content": json.dumps(target(i, r), ensure_ascii=False)},
        ]} for i, r in enumerate(rows) if pick(r["metadata"]["topic"])]
        (DATA_DIR / f"{prefix}_{split}.jsonl").write_text(
            "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
        print(f"{prefix}_{split}.jsonl: {len(out)} rows")
    return held


if __name__ == "__main__":
    held = write_split("diag", DIAG_SYSTEM, lambda i, r: {
        "topic": r["metadata"]["topic"], "misconception": r["metadata"]["hidden_misconception"]})
    # check: same eval dialogues as the Prober split
    prober_eval = {json.loads(l)["messages"][1]["content"] for l in open(DATA_DIR / "sft_eval.jsonl")}
    diag_eval = [json.loads(l)["messages"][2]["content"] for l in open(DATA_DIR / "diag_eval.jsonl")]
    assert len(diag_eval) == len(prober_eval) and all(json.loads(g)["topic"] in held for g in diag_eval)
