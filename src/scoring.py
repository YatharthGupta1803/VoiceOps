"""Scoring logic for evaluating agent answers against practice questions.

Scoring criteria (from DATA_DICTIONARY.md):
- must_include: facts separated by ' ; ' — ALL are required.
  Each fact may contain ' / ' alternatives — ANY ONE is enough.
- must_not_include: forbidden phrases — NONE should appear.
- expected_decision / acceptable_decisions: the agent's action type.
"""

import re
import unicodedata


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, normalize unicode for comparison."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _check_fact(fact: str, answer: str) -> bool:
    """Check if a single fact (possibly with ' / ' alternatives) is present in the answer."""
    answer_norm = _normalize(answer)
    alternatives = [alt.strip() for alt in fact.split(" / ")]
    return any(_normalize(alt) in answer_norm for alt in alternatives)


def score_must_include(must_include: str, answer: str) -> dict:
    """Check all required facts are present. Returns per-fact results.
    
    must_include format: "fact1 ; fact2 ; fact3"
    Each fact can have alternatives: "option_a / option_b"
    """
    if not must_include or not must_include.strip():
        return {"score": 1.0, "total": 0, "matched": 0, "details": []}

    facts = [f.strip() for f in must_include.split(" ; ") if f.strip()]
    results = []
    matched = 0
    for fact in facts:
        found = _check_fact(fact, answer)
        if found:
            matched += 1
        results.append({"fact": fact, "found": found})

    return {
        "score": matched / len(facts) if facts else 1.0,
        "total": len(facts),
        "matched": matched,
        "details": results,
    }


def score_must_not_include(must_not_include: str, answer: str) -> dict:
    """Check no forbidden phrases appear. Returns per-phrase results."""
    if not must_not_include or not must_not_include.strip():
        return {"score": 1.0, "total": 0, "violations": 0, "details": []}

    phrases = [p.strip() for p in must_not_include.split(" ; ") if p.strip()]
    answer_norm = _normalize(answer)
    results = []
    violations = 0
    for phrase in phrases:
        found = _normalize(phrase) in answer_norm
        if found:
            violations += 1
        results.append({"phrase": phrase, "found": found})

    return {
        "score": 1.0 if violations == 0 else 0.0,
        "total": len(phrases),
        "violations": violations,
        "details": results,
    }


def score_decision(expected_decision: str, acceptable_decisions: str,
                    answer: str) -> dict:
    """Heuristic check: does the answer match the expected decision type?
    
    Decision types: answer, clarify, escalate, refuse
    """
    answer_norm = _normalize(answer)

    # Parse acceptable decisions
    acceptable = [d.strip() for d in acceptable_decisions.split(" ; ")] \
        if acceptable_decisions else [expected_decision]
    if expected_decision not in acceptable:
        acceptable.append(expected_decision)

    # Heuristic signals for each decision type
    signals = {
        "clarify": [
            "could you", "can you", "what is your", "which order",
            "please provide", "please share", "what was", "how did you pay",
            "could you tell", "can you share", "do you have", "may i ask",
            "kya aap", "order id", "bata sakte", "bataye",
        ],
        "escalate": [
            "ticket", "colleague", "human", "team", "escalat",
            "connect you", "transfer", "senior", "priority",
            "sahayogi", "sahyogi", "सहयोगी",
        ],
        "refuse": [
            "cannot share", "can't share", "not allowed", "not able to",
            "i'm sorry", "i am sorry", "privacy", "unable to",
            "nahi de sakta", "nahi de sakte", "allowed nahi",
        ],
        "answer": [],  # Default if none of the above match
    }

    # Detect which decision the answer most likely represents
    detected = "answer"  # default
    for decision_type in ["refuse", "escalate", "clarify"]:
        if any(signal in answer_norm for signal in signals[decision_type]):
            detected = decision_type
            break

    matched = detected in acceptable

    return {
        "expected": expected_decision,
        "acceptable": acceptable,
        "detected": detected,
        "matched": matched,
    }


def score_answer(row: dict, answer: str) -> dict:
    """Full scoring for one practice question against an agent answer."""
    mi = score_must_include(row.get("must_include", ""), answer)
    mni = score_must_not_include(row.get("must_not_include", ""), answer)
    dec = score_decision(
        row.get("expected_decision", ""),
        row.get("acceptable_decisions", ""),
        answer,
    )

    # Combined score: weighted average
    # must_include matters most, must_not_include is binary, decision is important
    combined = (
        mi["score"] * 0.5 +
        mni["score"] * 0.25 +
        (1.0 if dec["matched"] else 0.0) * 0.25
    )

    return {
        "id": row.get("id", "?"),
        "language": row.get("language", "en"),
        "intent": row.get("intent", ""),
        "must_include": mi,
        "must_not_include": mni,
        "decision": dec,
        "combined_score": round(combined, 3),
    }

