#!/usr/bin/env python3
"""Generate a synthetic SFT dataset for a Socratic Diagnostic Assessor.

Each sample is a multi-turn dialogue in ChatML-style JSONL:
    {"messages": [{"role": "system"|"user"|"assistant", "content": ...}, ...], "metadata": {...}}

Assistant turns are `<thought>diagnostic reasoning</thought>` followed by ONE targeted
question. No lecturing, no code. The generator model (Claude, via structured outputs with
Pydantic) writes the whole dialogue, including the simulated learner, who holds a hidden
misconception the assistant must surface through questions alone.

Usage:
    python scripts/generate_sft_dataset.py --dry-run --mock                 # offline, no API key needed
    python scripts/generate_sft_dataset.py --provider gemini --dry-run      # 3 live samples, printed
    python scripts/generate_sft_dataset.py --provider groq -n 500           # append to data/sft_dataset.jsonl
    (re-run the same command to resume: seeds already in the file are skipped)

Keys (env): GEMINI_API_KEY / GROQ_API_KEY, or comma-separated GEMINI_API_KEYS / GROQ_API_KEYS.
When a key hits its daily quota the next key is used; when none are left the run stops cleanly.
Free-tier terms usually forbid creating extra accounts to dodge limits -- check yours.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Literal

import anthropic
import openai
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "sft_dataset.jsonl"
# provider -> (default model, OpenAI-compat base_url or None, default RPM, key env prefix)
PROVIDERS = {
    "anthropic": ("claude-opus-5-5", None, 50, "ANTHROPIC"),
    "gemini": ("gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3.1-flash-lite", "https://generativelanguage.googleapis.com/v1beta/openai/", 4, "GEMINI"),
    "groq": ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1", 30, "GROQ"),
}

SYSTEM_PROMPT = (
    "You are a Socratic Diagnostic Assessor. Your only job is to uncover the learner's "
    "mental model of a technical topic. Before every reply, reason privately inside "
    "<thought></thought>: what the learner's last answer reveals, which hypotheses about "
    "their model remain, and what single question best discriminates between them. Then ask "
    "exactly one short, targeted question. Never explain, correct, teach, praise, or show "
    "code. If the learner asks for the answer, redirect with a question."
)

# --------------------------------------------------------------------------- taxonomy
# domain -> topic -> a common misconception family the simulated learner may hold.
TAXONOMY: dict[str, dict[str, str]] = {
    "Agentic Systems": {
        "ReAct loop termination": "believes the agent stops when it 'feels done' rather than via an explicit stop condition or step budget",
        "Tool-calling reliability": "thinks the LLM executes tools itself instead of emitting a structured request that the runtime executes",
        "Agent memory (short vs long term)": "equates the context window with persistent memory",
        "Multi-agent orchestration": "assumes more agents always means better results, ignoring coordination cost and error compounding",
        "Planning vs reactive execution": "believes a plan, once generated, is binding and never needs replanning",
        "Guardrails and prompt injection via tool output": "trusts tool/web output as if it were operator instructions",
        "Agent evaluation": "judges an agent by final-answer quality only, ignoring trajectory, cost and failure modes",
        "Human-in-the-loop checkpoints": "thinks approval gates only matter for destructive actions, not for irreversible cost or data exposure",
        "Context management in long-horizon tasks": "believes bigger context windows eliminate the need for summarization or retrieval",
        "Idempotency of agent actions": "assumes retrying a failed tool call is always safe",
        "Determinism and temperature 0": "believes temperature 0 makes an agent run fully deterministic and reproducible",
        "Structured output reliability": "believes asking for JSON in the prompt guarantees schema-valid output",
    },
    "RAG / Vector Search": {
        "Embedding similarity semantics": "believes cosine similarity measures relevance to the question rather than closeness in embedding space",
        "Chunking strategy": "thinks smaller chunks are strictly better because they are 'more precise'",
        "Approximate nearest neighbor indexes (HNSW/IVF)": "assumes ANN search returns exact results with just a speed-up",
        "Hybrid search (BM25 + dense)": "believes dense retrieval fully replaces keyword matching",
        "Reranking": "thinks a reranker can recover documents the first-stage retriever never returned",
        "RAG hallucination": "believes retrieval guarantees the answer is grounded in the retrieved text",
        "Embedding model drift and re-indexing": "assumes embeddings from different models or versions are interchangeable in one index",
        "Metadata filtering": "thinks pre-filtering and post-filtering give the same results and cost",
        "RAG evaluation (recall vs answer quality)": "conflates retrieval recall with end-to-end answer correctness",
        "Top-k selection": "believes increasing k monotonically improves answer quality",
        "Query rewriting": "believes the user's raw message is always the best retrieval query",
        "Embedding dimensionality": "believes higher-dimensional embeddings are always better",
    },
    "Backend / Distributed Systems": {
        "Idempotency and retries": "thinks retries are harmless because the request 'already failed'",
        "CAP theorem": "believes you pick two of three permanently, rather than trade off during a partition",
        "Consistency models": "equates eventual consistency with 'data might be lost'",
        "Message queues and delivery guarantees": "believes exactly-once delivery is a property the broker simply provides",
        "Caching and invalidation": "assumes a TTL alone makes a cache correct",
        "Database indexes": "thinks adding more indexes only ever speeds things up",
        "Rate limiting and backpressure": "believes dropping excess requests is the only way to handle overload",
        "Distributed transactions and sagas": "assumes two-phase commit is the default safe choice across services",
        "Horizontal scaling and statelessness": "thinks adding instances scales any service linearly",
        "Leader election and clocks": "believes wall-clock timestamps give a reliable global ordering of events",
        "Transaction isolation levels": "assumes an ACID transaction means no concurrency anomalies by default",
        "Connection pooling": "believes a bigger connection pool always means more throughput",
    },
}

LEARNER_PROFILES = [
    "a junior engineer with 1 year of experience, eager but imprecise with terminology",
    "a mid-level backend engineer who knows the vocabulary but has shallow causal models",
    "a senior engineer from an adjacent field (e.g. data science) who over-transfers analogies",
    "a self-taught developer who learned from tutorials and blog posts",
    "a CS student who knows theory but has never operated this in production",
    "a confident tech lead whose model is outdated by several years",
]
STYLES = [
    "answers tersely, in one sentence",
    "answers at length and rambles slightly",
    "hedges often ('I think', 'probably')",
    "is overconfident and asserts things flatly",
    "occasionally asks the assistant to just give the answer",
]


class Seed(BaseModel):
    domain: str
    topic: str
    misconception: str
    profile: str
    style: str
    n_turns: int  # number of assistant turns


def build_seeds(n: int, rng: random.Random) -> list[Seed]:
    """Round-robin over domains and topics so coverage is balanced, with random persona."""
    topics = [(d, t, m) for d, ts in TAXONOMY.items() for t, m in ts.items()]
    # interleave domains: A1,B1,C1,A2,...
    by_domain = [[x for x in topics if x[0] == d] for d in TAXONOMY]
    for lst in by_domain:
        rng.shuffle(lst)
    ordered = [x for grp in itertools.zip_longest(*by_domain) for x in grp if x]
    seeds = []
    for i in range(n):
        d, t, m = ordered[i % len(ordered)]
        seeds.append(Seed(domain=d, topic=t, misconception=m, profile=rng.choice(LEARNER_PROFILES),
                          style=rng.choice(STYLES), n_turns=rng.choice([3, 4, 5, 6])))
    return seeds


# --------------------------------------------------------------------------- schema
def check_turn(role: str, thought: str | None, content: str) -> None:
    """Per-turn quality gate shared by single-call and two-role generation. Raises ValueError with a fix-it message."""
    if "`" in content or (thought and "`" in thought):
        raise ValueError("code/backticks are not allowed")
    if role == "user":
        if thought:
            raise ValueError("user turns must have null thought")
        if not 3 <= len(content.split()) <= 150:
            raise ValueError("learner message must be 3-150 words")
        return
    if not thought or len(thought.split()) < 12:
        raise ValueError("assistant thought missing or too short")
    if re.search(r"\b(final|last) (probe|question|turn)\b|\bone more (probe|question)\b", thought, re.I):
        raise ValueError("thought must not announce that the dialogue is ending")
    if re.search(r"</?thought>", content + thought):
        raise ValueError("do not include thought tags in fields")
    if content.count("?") != 1 or not content.rstrip().endswith("?"):
        raise ValueError("assistant must ask exactly one question, ending the turn")
    if re.match(r"(would|do|does|did|is|are|can|could|should|will|have|has)\b", content.strip(), re.I):
        raise ValueError("closed (yes/no) question; ask an open prediction or mechanism question")
    if re.search(r"\b(needto|inthe|ofthe|tothe|andthe|forthe|isthe|onthe)\b", content + " " + thought, re.I):
        raise ValueError("missing spaces between words")
    if len(content.split()) > 60:
        raise ValueError("assistant turn too long (lecturing?)")


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    thought: str | None = Field(None, description="Assistant turns only: private diagnostic reasoning, 2-4 sentences. Null for user turns.")
    content: str = Field(description="User turns: the learner's message. Assistant turns: ONE short targeted question, no preamble.")


class Dialogue(BaseModel):
    hidden_misconception: str = Field(description="The specific flawed mental model the learner holds in this dialogue.")
    turns: list[Turn] = Field(description="Alternating turns starting with the learner and ending with an assistant question.")

    @field_validator("turns")
    @classmethod
    def _alternate(cls, turns: list[Turn]) -> list[Turn]:
        if len(turns) < 3 or turns[0].role != "user" or turns[-1].role != "assistant":
            raise ValueError("turns must start with user, end with assistant, and have >= 3 turns")
        for a, b in zip(turns, turns[1:]):
            if a.role == b.role:
                raise ValueError("roles must alternate")
        return turns

    @model_validator(mode="after")
    def _quality(self) -> "Dialogue":
        for i, t in enumerate(self.turns):
            try:
                check_turn(t.role, t.thought, t.content)
            except ValueError as e:
                raise ValueError(f"turn {i}: {e}") from None
        return self


def to_chatml(d: Dialogue, seed: Seed) -> dict:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in d.turns:
        c = f"<thought>{t.thought.strip()}</thought>\n\n{t.content.strip()}" if t.role == "assistant" else t.content.strip()
        msgs.append({"role": t.role, "content": c})
    return {"messages": msgs, "metadata": {**seed.model_dump(), "hidden_misconception": d.hidden_misconception}}


# --------------------------------------------------------------------------- generation
GEN_SYSTEM = (
    "You write training data for a Socratic Diagnostic Assessor. You author BOTH sides of a dialogue: "
    "a realistic learner who sincerely holds a specific misconception, and an assistant who diagnoses "
    "that misconception using only questions.\n\n"
    "Assistant rules (strict):\n"
    "- Each assistant turn has a private `thought` (2-4 sentences: what the learner's answer reveals, "
    "which hypotheses remain, why the next question discriminates between them) and `content` that is "
    "exactly ONE short question (max ~40 words).\n"
    "- Never lecture, explain, correct, hint at the right answer, praise, or summarize. Never include code, "
    "pseudo-code, or backticks. If the learner asks for the answer, ask a question instead.\n"
    "- Questions must be targeted: scenario-based predictions ('what happens if...'), edge cases, or "
    "'why' probes that separate surface vocabulary from a causal model. Escalate in specificity across turns.\n"
    "- Questions must be OPEN: never yes/no or 'which of these two' questions; ask for a prediction or a mechanism "
    "('what happens when...', 'how does X decide...', 'walk me through...').\n"
    "- Questions must be NEUTRAL: never state, imply or presuppose the correct model (no 'why might X fail', no "
    "'given that X is unlikely to help'). The learner should be able to answer in line with their own misconception.\n"
    "- Proofread: every word properly spaced and spelled.\n"
    "- Never say or imply in a thought that this is the last question or that the dialogue is ending; you cannot know that. "
    "Vary how thoughts begin: do not open every thought with 'The learner' or 'They'.\n"
    "- Introduce NO new technique, term or concept the learner has not already used (e.g. never name 'saga', "
    "'reranker', 'diversity filter' yourself). Probe with scenarios and consequences; let the learner supply the vocabulary.\n"
    "- The FINAL question gets the same discipline and must NOT be a template: vary its form across dialogues "
    "(a prediction, a 'walk me through', a request to justify a claim the learner made, a boundary case). Never end with "
    "'where would you draw the line'. Never offer a choice between named options.\n"
    "- One idea per question: no 'X, and how does Y?' double questions.\n"
    "- Every claim embedded in a question must be technically TRUE for the real system; never build a question on a false premise.\n"
    "- Every `thought` may refer ONLY to things the learner has actually said so far; never attribute statements they did not make.\n"
    "- The learner's FIRST message must not state the misconception outright; it should show it indirectly (a design choice, "
    "a plan, a mild question). Afterwards the learner stays consistent with the misconception and never volunteers the correct mechanism.\n"
    "- By the final turn the thought should show the assistant has converged on the learner's actual mental model.\n"
    "Learner rules: stay in persona, reveal the misconception gradually and naturally, never state it outright "
    "on the first turn. The dialogue starts with the learner's opening message and ends with an assistant question."
)


def seed_prompt(s: Seed) -> str:
    return (
        f"Domain: {s.domain}\nTopic: {s.topic}\nLearner's hidden misconception: {s.misconception}\n"
        f"Learner persona: {s.profile}; {s.style}.\nNumber of assistant turns: exactly {s.n_turns} "
        f"(so {s.n_turns * 2 - 1} turns total, starting with the learner)."
    )


class QuotaExhausted(Exception):
    pass


class Pool:
    """API keys for one provider: shared RPM pacer + rotation when a key's daily quota is spent."""

    def __init__(self, provider: str, rpm: int, models: list[str]):
        _, base_url, _, prefix = PROVIDERS[provider]
        raw = os.environ.get(f"{prefix}_API_KEYS") or os.environ.get(f"{prefix}_API_KEY") or ""
        keys = [k.strip() for k in raw.split(",") if k.strip()] or [None]  # None: SDK default auth
        if provider == "anthropic":
            self.clients = [anthropic.AsyncAnthropic(api_key=k, max_retries=2) for k in keys]
        else:
            if keys == [None]:
                sys.exit(f"Set {prefix}_API_KEY (or {prefix}_API_KEYS=k1,k2,...).")
            self.clients = [openai.AsyncOpenAI(api_key=k, base_url=base_url, max_retries=2) for k in keys]
        # free-tier quotas are per (key, model): rotate through every combination
        self.slots = [(c, m) for m in models for c in self.clients]
        self.idx = 0
        self.interval = 60.0 / (rpm * len(keys))  # ponytail: one shared pacer; per-slot pacing if models differ a lot in RPM
        self._next = 0.0
        self._lock = asyncio.Lock()

    def current(self):
        if self.idx >= len(self.slots):
            raise QuotaExhausted("all keys/models exhausted for today")
        return self.slots[self.idx]

    def retire(self, slot) -> None:
        if self.idx < len(self.slots) and self.slots[self.idx] is slot:  # ignore stale concurrent retirements
            self.idx += 1
            print(f"[quota] {slot[1]} exhausted, {len(self.slots) - self.idx} key/model slots left", file=sys.stderr)

    async def pace(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait:
            await asyncio.sleep(wait)

    async def close(self) -> None:
        for c in self.clients:
            await c.close()


RATE_ERRORS = (anthropic.RateLimitError, openai.RateLimitError)
TRANSIENT = RATE_ERRORS + (anthropic.InternalServerError, openai.InternalServerError,
                           anthropic.APIConnectionError, openai.APIConnectionError)


def retry_after(e: Exception) -> float | None:
    h = getattr(getattr(e, "response", None), "headers", None) or {}
    v = h.get("retry-after")
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def is_daily_quota(e: Exception) -> bool:
    ra = retry_after(e)
    return bool((ra and ra > 120) or re.search(r"per ?day|daily|RPD|TPD", str(e), re.I))


def parse_json(text: str, cls):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())  # some models fence JSON despite JSON mode
    return cls.model_validate_json(text)


USAGE = {"calls": 0, "tokens": 0}  # printed at exit: lets you see what a run costs against free-tier caps


async def call_structured(client, provider: str, model: str, system: str, user: str, cls):
    """One model call returning a validated `cls` instance (Anthropic: native structured output; others: JSON mode)."""
    if provider == "anthropic":
        resp = await client.messages.parse(model=model, max_tokens=8000, system=system,
                                           messages=[{"role": "user", "content": user}], output_format=cls)
        USAGE["calls"] += 1
        USAGE["tokens"] += resp.usage.input_tokens + resp.usage.output_tokens
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise ValueError(f"no parsed output (stop_reason={resp.stop_reason})")
        return resp.parsed_output
    schema = json.dumps(cls.model_json_schema())
    resp = await client.chat.completions.create(
        model=model, temperature=0.8, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system + f"\n\nReturn ONLY a JSON object matching this JSON Schema:\n{schema}"},
                  {"role": "user", "content": user}],
    )
    USAGE["calls"] += 1
    USAGE["tokens"] += getattr(resp.usage, "total_tokens", 0) or 0
    return parse_json(resp.choices[0].message.content or "", cls)


async def ask(pool: Pool, provider: str, system: str, user: str, cls, validate=None, max_bad: int = 3, max_waits: int = 12):
    """Pacing, backoff on 429/5xx, slot rotation on daily quota, and re-ask (with the error) on a bad result.

    Bad results and transient waits have separate budgets so a rate-limit blip never eats the re-ask budget
    (each re-ask costs a request, which matters on a small free tier)."""
    feedback = ""
    bad = waits = 0
    last_err: Exception | None = None
    while bad < max_bad and waits < max_waits:
        slot = pool.current()  # raises QuotaExhausted when nothing is left
        client, model = slot
        await pool.pace()
        try:
            out = await call_structured(client, provider, model, system, user + feedback, cls)
            if validate:
                validate(out)
            return out
        except (ValidationError, ValueError) as e:  # bad output: re-ask with the validator's message
            last_err, bad = e, bad + 1
            feedback = f"\n\nYour previous attempt was rejected: {str(e)[:400]}\nFix this and try again."
        except RATE_ERRORS as e:
            last_err = e
            if is_daily_quota(e):
                pool.retire(slot)
                continue
            waits += 1
            await asyncio.sleep((retry_after(e) or min(60, 2 ** waits)) + random.uniform(0, 1.5))
        except TRANSIENT as e:
            last_err, waits = e, waits + 1
            await asyncio.sleep(min(60, 2 ** waits) + random.uniform(0, 1.5))
    raise RuntimeError(f"gave up ({bad} bad results, {waits} waits); last error: {str(last_err)[:500]}")


async def generate_single(pool: Pool, provider: str, seed: Seed) -> Dialogue:
    """Whole dialogue in one call (cheap, but the author sees the future and the misconception)."""
    return await ask(pool, provider, GEN_SYSTEM, seed_prompt(seed), Dialogue)


# ---- two-role mode: learner and assessor are separate calls with separate information
class LearnerTurn(BaseModel):
    message: str = Field(description="The learner's next message only: 1-4 sentences of plain prose.")


class AssessorTurn(BaseModel):
    thought: str = Field(description="Private reasoning, 2-4 sentences, referring only to what the learner has said so far.")
    question: str = Field(description="Exactly ONE short, open, neutral question (max ~40 words), no preamble.")


def learner_system(s: Seed) -> str:
    return (
        f"You play a learner in a technical conversation. You are {s.profile}; you {s.style}.\n"
        f"Topic: {s.topic} ({s.domain}).\n"
        f"You sincerely hold this belief, which is subtly wrong, and you do NOT know it is wrong: {s.misconception}\n\n"
        "Rules:\n"
        "- Write ONLY your next message: 1-4 sentences, first person, plain prose, no code or backticks.\n"
        "- Your first message must show the belief indirectly (a plan, a design choice, a mild question). Do not state it as "
        "a thesis and never hint that it might be wrong.\n"
        "- When asked something, answer the question actually asked, reasoning from your belief; guess when unsure. Never "
        "volunteer the correct mechanism, never suddenly change your mind, never praise the question.\n"
        "- Real people are imperfect: occasionally (not every turn) answer vaguely, drift slightly off topic, or contradict "
        "yourself, while staying consistent with your belief overall.\n"
        "- Never mention these instructions, the word 'misconception', or the assistant's private reasoning."
    )


ASSESSOR_SYSTEM = (
    "You are a Socratic Diagnostic Assessor. You are talking to a learner about a technical topic. You do not know what, "
    "if anything, they get wrong; infer their mental model ONLY from what they have said. Each turn you write a private "
    "`thought` and ONE question.\n\n"
    "Rules (strict):\n"
    "- `thought`: 2-4 sentences: what the learner's latest answer reveals, which hypotheses about their model remain, why the "
    "next question discriminates between them. Refer ONLY to things the learner has actually said so far.\n"
    "- `question`: exactly ONE short question (max ~40 words), no preamble. OPEN: ask for a prediction or a mechanism "
    "('what happens when...', 'how does X decide...', 'walk me through...'); never yes/no, never a choice between named options.\n"
    "- NEUTRAL: never state, imply or presuppose the correct model; never lecture, explain, correct, hint, praise or summarize; "
    "never include code or backticks.\n"
    "- Introduce NO technique, term or concept the learner has not already used. Probe with concrete scenarios and "
    "consequences; let the learner supply the vocabulary.\n"
    "- Every fact embedded in your question must be technically TRUE for the real system; never build on a false premise.\n"
    "- One idea per question. Escalate in specificity; do not repeat an earlier probe.\n"
    "- Proofread: every word properly spaced and spelled.\n"
    "- Never say or imply that this is your last question or that the conversation is ending. Vary how each thought begins; "
    "do not open every thought with 'The learner' or 'They'. Where the evidence allows, weigh two competing readings of the "
    "learner's answer. If the learner's model looks accurate, say so in the thought and probe a harder edge instead of inventing a flaw.\n"
    "- If the learner asks for the answer or for code, or pushes back, do not comply or explain: ask a short concrete question "
    "tied to their stated goal."
)


def transcript(turns: list[Turn], viewer: str) -> str:
    out = []
    for t in turns:
        if t.role == "user":
            out.append(f"{'You' if viewer == 'learner' else 'Learner'}: {t.content}")
        elif viewer == "learner":  # the learner never sees the assessor's private thoughts
            out.append(f"Assistant: {t.content}")
        else:
            out.append(f"[your private thought: {t.thought}]\nYou asked: {t.content}")
    return "\n\n".join(out)


async def generate_two_role(pool: Pool, provider: str, seed: Seed) -> Dialogue:
    # ponytail: a dialogue is lost if quota dies mid-way (~8 calls wasted); checkpoint partial turns if that hurts
    turns: list[Turn] = []
    for k in range(seed.n_turns):
        prompt = ("Conversation so far:\n\n" + transcript(turns, "learner") + "\n\nWrite your next message.") if turns \
            else "Write your opening message."
        lt = await ask(pool, provider, learner_system(seed), prompt, LearnerTurn,
                       lambda r: check_turn("user", None, r.message))
        turns.append(Turn(role="user", thought=None, content=lt.message.strip()))
        prompt = (f"Topic: {seed.topic} ({seed.domain}).\n\nConversation so far:\n\n{transcript(turns, 'assessor')}\n\n"
                  "Write your next thought and question.")
        at = await ask(pool, provider, ASSESSOR_SYSTEM, prompt, AssessorTurn,
                       lambda r: check_turn("assistant", r.thought, r.question))
        turns.append(Turn(role="assistant", thought=at.thought.strip(), content=at.question.strip()))
    return Dialogue(hidden_misconception=seed.misconception, turns=turns)


MOCK_TURNS = [
    ("My agent keeps looping on tool calls and I assume it'll stop when it has enough info.",
     None),
    (None, ("The learner assumes termination is an emergent judgement by the model. I need to know whether they distinguish the model's output from the runtime's control flow. A prediction question on a failing tool separates the two.",
            "Suppose the search tool returns an empty result every time; what, concretely, decides whether the next step happens?")),
    ("The model sees nothing useful, so it probably decides to try again or give up.", None),
    (None, ("They again attribute the decision entirely to the model with no mention of a step cap or runtime check. I should test whether any external bound exists in their picture, by asking about cost over time.",
            "If it never 'decides' to give up, where in your system would the loop actually be cut off?")),
    ("Hm, I guess I never set anything for that. I thought the model knows.", None),
    (None, ("Now the learner admits no explicit bound exists and believes the model self-regulates. The model is: termination is implicit in the LLM. Whether they see the model as stateful across calls is the remaining gap.",
            "What information does the model have at step ten that it did not have at step one?")),
]


def mock_dialogue(seed: Seed) -> Dialogue:
    turns = []
    for u, a in MOCK_TURNS[: seed.n_turns * 2 - 1 + 1]:
        turns.append(Turn(role="user", thought=None, content=u) if u else Turn(role="assistant", thought=a[0], content=a[1]))
    turns = turns[: max(3, (len(turns) // 2) * 2)]
    return Dialogue(hidden_misconception=seed.misconception, turns=turns)


def seed_key(s: Seed) -> str:
    return json.dumps(s.model_dump(), sort_keys=True)


def done_keys() -> set[str]:
    if not OUTPUT_PATH.exists():
        return set()
    keys = set()
    for line in OUTPUT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            m = json.loads(line)["metadata"]
            keys.add(seed_key(Seed(**{k: m[k] for k in Seed.model_fields})))
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
            continue  # ignore malformed/foreign rows
    return keys


async def run(seeds: list[Seed], args) -> list[dict]:
    pool = None if args.mock else Pool(args.provider, args.rpm, args.model.split(","))
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stop = asyncio.Event()
    out_f = None if args.dry_run else OUTPUT_PATH.open("a", encoding="utf-8")
    results: list[dict] = []
    failed = 0

    async def worker(seed: Seed):
        nonlocal failed
        async with sem:
            if stop.is_set():
                return
            try:
                d = mock_dialogue(seed) if args.mock else await (generate_two_role if args.mode == 'two-role' else generate_single)(pool, args.provider, seed)
            except QuotaExhausted as e:
                stop.set()
                print(f"[stop] {e}; re-run later to resume", file=sys.stderr)
                return
            except Exception as e:  # one bad seed must not kill the batch
                failed += 1
                print(f"[fail] {seed.domain} / {seed.topic}: {e}", file=sys.stderr)
                return
        row = to_chatml(d, seed)
        row["metadata"]["mode"] = args.mode
        async with lock:
            results.append(row)
            if out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()  # keep partial progress on crash/ctrl-c

    try:
        for i in range(0, len(seeds), args.batch_size):
            if stop.is_set():
                break
            batch = seeds[i : i + args.batch_size]
            await asyncio.gather(*(worker(s) for s in batch))
            print(f"[batch] {min(i + args.batch_size, len(seeds))}/{len(seeds)} done, {len(results)} ok, {failed} failed", file=sys.stderr)
    finally:
        if out_f:
            out_f.close()
        if pool:
            await pool.close()
    return results


def load_rows() -> list[dict]:
    return [json.loads(l) for l in OUTPUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def held_out_topics(rows: list[dict], eval_frac: float = 0.2) -> set[str]:
    """Eval topics, picked by hash so the split is stable. Shared by all three adapters' datasets."""
    import hashlib, math
    topics = sorted({r["metadata"]["topic"] for r in rows}, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    return set(topics[: max(1, math.ceil(len(topics) * eval_frac))])


def export_splits(eval_frac: float = 0.2) -> None:
    """Write sft_train.jsonl / sft_eval.jsonl (messages only) split BY TOPIC, so near-duplicate dialogues never straddle the split."""
    rows = load_rows()
    held = held_out_topics(rows, eval_frac)
    for name, pick in (("sft_train.jsonl", lambda t: t not in held), ("sft_eval.jsonl", lambda t: t in held)):
        out = [{"messages": r["messages"]} for r in rows if pick(r["metadata"]["topic"])]
        (DATA_DIR / name).write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
        print(f"{name}: {len(out)} rows", file=sys.stderr)
    print(f"held-out topics ({len(held)}): {sorted(held)}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--num-samples", type=int, default=100, help="target total rows in the file (resumes)")
    p.add_argument("--mode", choices=["two-role", "single"], default="two-role",
                   help="two-role: separate learner/assessor calls (~2*turns requests, higher quality); single: 1 call/dialogue")
    p.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    p.add_argument("--model", help="override default model; comma-separate several to fall through as each hits its daily quota")
    p.add_argument("--rpm", type=int, help="client-side requests/minute per key (default per provider)")
    p.add_argument("--batch-size", type=int, default=20, help="seeds per gather() wave; progress is reported per wave")
    p.add_argument("--concurrency", type=int, default=3, help="max in-flight requests")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible seed sampling")
    p.add_argument("--export-splits", action="store_true", help="write sft_train.jsonl / sft_eval.jsonl (messages only, split by topic) from sft_dataset.jsonl and exit")
    p.add_argument("--dry-run", action="store_true", help="generate 3 samples, print them, write nothing")
    p.add_argument("--mock", action="store_true", help="offline canned dialogue (pipeline/format check, no API call)")
    args = p.parse_args()
    if args.export_splits:
        return export_splits()
    args.model = args.model or PROVIDERS[args.provider][0]
    args.rpm = args.rpm or PROVIDERS[args.provider][2]

    n = 3 if args.dry_run else args.num_samples
    seeds = build_seeds(n, random.Random(args.seed))
    if not args.dry_run:
        done = done_keys()
        seeds = [s for s in seeds if seed_key(s) not in done]
        print(f"{n - len(seeds)} already in {OUTPUT_PATH}, {len(seeds)} to generate", file=sys.stderr)
    rows = asyncio.run(run(seeds, args))
    if USAGE["calls"]:
        print(f"[usage] {USAGE['calls']} calls, {USAGE['tokens']} tokens, ~{USAGE['tokens'] // max(len(rows), 1)} tokens/dialogue", file=sys.stderr)

    if args.dry_run:
        for i, row in enumerate(rows, 1):
            m = row["metadata"]
            print(f"\n{'=' * 78}\nSAMPLE {i}  [{m['domain']} / {m['topic']}]\nhidden: {m['hidden_misconception']}\n{'=' * 78}")
            for msg in row["messages"]:
                print(f"\n--- {msg['role'].upper()} ---\n{msg['content']}")
        print(f"\n[dry-run] {len(rows)}/3 samples generated; nothing written.")
    else:
        print(f"Wrote {len(rows)} new samples to {OUTPUT_PATH}")
    if seeds and not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
