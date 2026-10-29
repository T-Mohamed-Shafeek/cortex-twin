# CortexTwin

Fine-tuning Llama-3.1-8B-Instruct into a **Socratic diagnostic assessor**: a model that finds out *what a learner misunderstands* by asking questions, never by explaining.

The behaviour is split into three QLoRA adapters on one shared base model:

| Adapter | Job | Input | Output |
|---|---|---|---|
| **Prober** | Run the conversation | Dialogue so far | `<thought>…</thought>` + one open question |
| **Diagnoser** | Name the misconception | Finished transcript | `{"topic", "misconception"}` |
| **Concept-Graph** | Map what the learner knows | Finished transcript | Concepts marked `known / shaky / missing / untested` + prerequisite edges |

## Results (Prober, 79 held-out turns, greedy decoding)

The base model and the fine-tuned model got the same system prompt. All eval dialogues are on topics never seen in training.

| | Base | Fine-tuned |
|---|---|---|
| Format compliance (`<thought>` + exactly one question, no code) | 68/79 (86%) | **79/79 (100%)** |
| Closed yes/no questions | 24/79 (30%) | **0/79 (0%)** |
| Avg. words in `<thought>` | 38 | 28 |

Training: QLoRA (4-bit, r=16, all linear projections), 0.52% of parameters trained, 3 epochs / 30 steps, ~7 min on Kaggle's 2× T4. Loss is computed only on assistant turns. Eval loss: 2.140 → 1.921 → 1.904.

Diagnoser and Concept-Graph: datasets are built and the training/eval code is in the notebook. Results pending.

## Repository layout

```
scripts/
  generate_sft_dataset.py    # dialogue generator (Gemini/Groq, Pydantic-validated turns) + train/eval split
  build_diagnoser_dataset.py # dialogues -> Diagnoser data
  build_graph_dataset.py     # dialogues + labels -> Concept-Graph data
  graph_templates.py         # 5 concepts + prerequisite edges per topic (36 topics)
data/
  sft_dataset.jsonl          # master file: 100 dialogues with metadata (topic, hidden misconception, persona, ...)
  sft_{train,eval}.jsonl     # Prober data (80/20)
  diag_{train,eval}.jsonl    # Diagnoser data (80/20)
  graph_{train,eval}.jsonl   # Concept-Graph data (80/20)
  graph_labels.json          # per-dialogue concept statuses
notebooks/
  sft-for-socratic-probing-behaviour.ipynb  # Kaggle: train all 3 adapters + evaluate vs base model
```

## Reproduce

```bash
uv sync
python scripts/generate_sft_dataset.py --dry-run --mock      # offline smoke test, no API key
python scripts/generate_sft_dataset.py --export-splits       # data/sft_{train,eval}.jsonl
python scripts/build_diagnoser_dataset.py                    # data/diag_{train,eval}.jsonl
python scripts/build_graph_dataset.py                        # data/graph_{train,eval}.jsonl
```

To generate new dialogues, put `GEMINI_API_KEY` or `GROQ_API_KEY` in `.env` (see the script's docstring). Then upload the six `*_train/eval.jsonl` files as a Kaggle dataset and run the notebook on **GPU T4 x2**.

**Split:** all three datasets hold out the same 8 of 36 topics (chosen by hash), so every eval number is on unseen topics.

## Data notes

- The data is fully synthetic: dialogues were generated with LLMs (Gemini/Groq, then Claude-written). Every assistant turn passes the same validator: a thought of 12+ words, exactly one question, no yes/no opener, no code.
- Diagnoser labels are the seeded misconception for each dialogue. Concept-Graph statuses were **annotated by Claude**, with a single annotator. Scores on those adapters measure agreement with that labelling, not with human experts.
- The Prober metrics are rule-based checks of the trained behaviour, not a judgement of question quality.
