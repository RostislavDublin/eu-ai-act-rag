"""
RAG Triad evaluation: Faithfulness, Answer Relevancy, Context Precision.
Each metric uses LLM-as-judge via DIAL.
"""
import json
from dataclasses import dataclass

from .pipeline import DIALClient


@dataclass
class TriadScore:
    faithfulness: float       # 0.0-1.0  is the answer grounded in context?
    answer_relevancy: float   # 0.0-1.0  does the answer address the question?
    context_precision: float  # 0.0-1.0  were the retrieved chunks useful?

    @property
    def average(self) -> float:
        return (self.faithfulness + self.answer_relevancy + self.context_precision) / 3

    def as_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "average": round(self.average, 3),
        }


_FAITHFULNESS_PROMPT = """\
You are evaluating whether a generated answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Task:
1. Break the answer into individual factual claims (sentences or sub-claims).
2. For each claim, decide: SUPPORTED (the context directly supports it) or UNSUPPORTED.
3. Count supported and total claims.

Return ONLY a JSON object with this exact format:
{{"supported": <integer>, "total": <integer>}}

No explanation, no markdown, just the JSON."""


_ANSWER_RELEVANCY_PROMPT = """\
You are evaluating whether an answer directly addresses the original question.

Question: {question}

Answer: {answer}

Score from 0.0 to 1.0:
- 1.0 = answer fully and directly addresses the question
- 0.7 = answer is mostly relevant with minor gaps
- 0.5 = partial answer or contains off-topic content
- 0.2 = tangentially related but does not answer the question
- 0.0 = completely off-topic or refuses to answer

Return ONLY a JSON object:
{{"score": <float between 0.0 and 1.0>}}

No explanation, no markdown, just the JSON."""


_CONTEXT_PRECISION_PROMPT = """\
You are evaluating retrieval quality for a RAG system.

Question: {question}

Retrieved context chunks:
{numbered_chunks}

For each chunk, decide whether it is RELEVANT (would help a human answer the question) or NOT RELEVANT.

Return ONLY a JSON object with the 1-based indices of relevant chunks:
{{"relevant": [<list of integers>]}}

Example: {{"relevant": [1, 3]}} means chunks 1 and 3 are relevant.
No explanation, no markdown, just the JSON."""


class RagTriadEvaluator:
    def __init__(self, dial_client: DIALClient):
        self._dial = dial_client

    def evaluate(
        self,
        question: str,
        answer: str,
        context_chunks: list[dict],
    ) -> TriadScore:
        context_text = "\n\n".join(c["text"] for c in context_chunks)
        return TriadScore(
            faithfulness=self._faithfulness(answer, context_text),
            answer_relevancy=self._answer_relevancy(question, answer),
            context_precision=self._context_precision(question, context_chunks),
        )

    def _faithfulness(self, answer: str, context: str) -> float:
        prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        raw = self._dial.chat([{"role": "user", "content": prompt}], max_tokens=128)
        try:
            data = _parse_json(raw)
            total = int(data.get("total", 1))
            supported = int(data.get("supported", 0))
            return min(1.0, supported / total) if total > 0 else 1.0
        except Exception:
            return 0.5

    def _answer_relevancy(self, question: str, answer: str) -> float:
        prompt = _ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
        raw = self._dial.chat([{"role": "user", "content": prompt}], max_tokens=64)
        try:
            data = _parse_json(raw)
            return float(min(1.0, max(0.0, data.get("score", 0.5))))
        except Exception:
            return 0.5

    def _context_precision(self, question: str, chunks: list[dict]) -> float:
        if not chunks:
            return 0.0
        numbered = "\n\n".join(
            f"Chunk {i + 1}:\n{c['text'][:800]}" for i, c in enumerate(chunks)
        )
        prompt = _CONTEXT_PRECISION_PROMPT.format(
            question=question, numbered_chunks=numbered
        )
        raw = self._dial.chat([{"role": "user", "content": prompt}], max_tokens=128)
        try:
            data = _parse_json(raw)
            relevant_set = set(int(x) for x in data.get("relevant", []))
            if not relevant_set:
                return 0.0

            # Position-weighted Precision@K (standard RAG Triad / TruLens formula).
            # For each relevant chunk at rank k, compute Precision@k = (# relevant in top-k) / k.
            # Final score = mean of Precision@k over all relevant positions.
            # This rewards systems that rank relevant chunks earlier.
            n = len(chunks)
            precisions_at_relevant = []
            relevant_seen = 0
            for k in range(1, n + 1):
                if k in relevant_set:
                    relevant_seen += 1
                    precisions_at_relevant.append(relevant_seen / k)

            return sum(precisions_at_relevant) / len(relevant_set)
        except Exception:
            return 0.5


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response that may include prose."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in: {text!r}")
    return json.loads(text[start:end])
