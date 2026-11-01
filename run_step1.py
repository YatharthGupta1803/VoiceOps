#!/usr/bin/env python3
"""Step 1 Baseline Evaluation — Run all 52 practice questions through the
bare LLM (no RAG) and record scores.

Usage:
    python run_step1.py [--dry-run]

    --dry-run   Test the scoring pipeline on reference answers without
                calling the Gemini API. Costs zero tokens.
"""

import csv
import json
import os
import sys
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.agent import create_agent, ask
from src.prompts import SYSTEM_PROMPT_STEP1
from src.scoring import score_answer

QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "practice", "questions.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def load_questions(path: str) -> list[dict]:
    """Load practice questions CSV."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_evaluation(dry_run: bool = False):
    """Run all 52 practice questions and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    questions = load_questions(QUESTIONS_PATH)
    print(f"📋 Loaded {len(questions)} practice questions")
    
    if dry_run:
        print("🧪 DRY RUN — using reference answers (zero API calls)\n")
    else:
        model = create_agent(SYSTEM_PROMPT_STEP1)
        print(f"🤖 Using Groq ({model['model']}) — running {len(questions)} evaluation queries\n")

    results = []
    total_score = 0.0
    mi_total_matched = 0
    mi_total_facts = 0
    mni_violations = 0
    decision_matches = 0

    for i, q in enumerate(questions):
        qid = q["id"]
        question_text = q["question"]
        lang = q["language"]
        
        print(f"[{i+1:2d}/{len(questions)}] {qid} ({lang}) ", end="", flush=True)

        if dry_run:
            # Use the reference answer to test scoring pipeline
            answer = q.get("reference_answer", "")
        else:
            answer = ask(model, question_text)
            # Small buffer for smooth throughput on Groq
            if i < len(questions) - 1:
                time.sleep(1)

        # Score
        scores = score_answer(q, answer)
        scores["question"] = question_text
        scores["answer"] = answer
        scores["reference_answer"] = q.get("reference_answer", "")
        results.append(scores)

        # Accumulate
        total_score += scores["combined_score"]
        mi_total_matched += scores["must_include"]["matched"]
        mi_total_facts += scores["must_include"]["total"]
        if scores["must_not_include"]["violations"] > 0:
            mni_violations += 1
        if scores["decision"]["matched"]:
            decision_matches += 1

        # Status icon
        if scores["combined_score"] >= 0.75:
            icon = "✅"
        elif scores["combined_score"] >= 0.5:
            icon = "🟡"
        else:
            icon = "❌"
        print(f"{icon} score={scores['combined_score']:.2f}  "
              f"mi={scores['must_include']['matched']}/{scores['must_include']['total']}  "
              f"dec={'✓' if scores['decision']['matched'] else '✗'}({scores['decision']['detected']})")

    # Summary
    n = len(questions)
    avg_score = total_score / n
    mi_rate = mi_total_matched / mi_total_facts if mi_total_facts > 0 else 1.0

    summary = {
        "timestamp": timestamp,
        "step": 1,
        "model": "openai/gpt-oss-120b",
        "mode": "dry_run" if dry_run else "live",
        "total_questions": n,
        "average_combined_score": round(avg_score, 4),
        "must_include_facts_matched": mi_total_matched,
        "must_include_facts_total": mi_total_facts,
        "must_include_rate": round(mi_rate, 4),
        "must_not_include_violations": mni_violations,
        "decision_match_rate": round(decision_matches / n, 4),
        "decision_matches": decision_matches,
    }

    print("\n" + "=" * 65)
    print("📊 STEP 1 BASELINE RESULTS")
    print("=" * 65)
    print(f"  Average combined score : {avg_score:.1%}")
    print(f"  Must-include fact rate : {mi_rate:.1%} ({mi_total_matched}/{mi_total_facts})")
    print(f"  Must-not-include clean : {n - mni_violations}/{n} questions")
    print(f"  Decision type match   : {decision_matches}/{n} ({decision_matches/n:.1%})")
    print("=" * 65)

    # Save detailed results
    results_path = os.path.join(RESULTS_DIR,
                                f"step1_{'dry' if dry_run else 'live'}_{timestamp}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n💾 Detailed results saved to: {results_path}")

    # Save summary separately for easy comparison with Step 2
    summary_path = os.path.join(RESULTS_DIR, "step1_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"💾 Summary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_evaluation(dry_run=dry_run)

