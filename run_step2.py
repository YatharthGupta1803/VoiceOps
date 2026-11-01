#!/usr/bin/env python3
"""Step 2 RAG Evaluation — Run all 52 practice questions through the
grounded RAG agent and record scores for comparison against Step 1.

Usage:
    python run_step2.py [--dry-run]
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

from src.agent import ask
from src.prompts_step2 import SYSTEM_PROMPT_STEP2
from src.retriever import HybridRetriever
from src.scoring import score_answer

QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "practice", "questions.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def format_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Combine retrieved knowledge-base context with the customer's question."""
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(
            f"--- [{c['section_id']}] {c['title']} ({c['doc_type']}, status: {c['status']}) ---\n"
            f"{c['text']}"
        )
    context_str = "\n\n".join(context_blocks)

    return (
        f"RETRIEVED KNOWLEDGE BASE CONTEXT:\n"
        f"{context_str}\n\n"
        f"CUSTOMER QUESTION:\n"
        f"{question}\n\n"
        f"INSTRUCTION:\n"
        f"Answer the customer's question grounded strictly in the context above. "
        f"Include relevant numbers, dates, or timelines from the policies. "
        f"If the context does not contain the answer, say you don't have that information "
        f"and offer to connect with a colleague."
    )


def run_step2_evaluation(dry_run: bool = False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(QUESTIONS_PATH, encoding="utf-8", newline="") as f:
        questions = list(csv.DictReader(f))

    print(f"📋 Loaded {len(questions)} practice questions")
    retriever = HybridRetriever()

    agent_cfg = {
        "model": "openai/gpt-oss-120b",
        "system_prompt": SYSTEM_PROMPT_STEP2,
    }

    results = []
    total_score = 0.0
    mi_total_matched = 0
    mi_total_facts = 0
    mni_violations = 0
    decision_matches = 0
    retrieval_hits = 0
    total_with_evidence = 0

    print(f"\n🚀 Running Step 2 Grounded RAG Evaluation ({'DRY RUN' if dry_run else 'LIVE'})...\n")

    for i, q in enumerate(questions):
        qid = q["id"]
        question_text = q["question"]
        lang = q["language"]
        gold_evidence = q.get("gold_evidence", "").strip()

        # Extract SKU entity if present
        sku = q.get("entities", "").replace("sku=", "") if "sku=" in q.get("entities", "") else None

        # 1. Retrieve
        retrieved_chunks = retriever.retrieve(question_text, top_k=5, entity_sku=sku)
        retrieved_ids = [c["section_id"] for c in retrieved_chunks]

        # Check retrieval recall against gold evidence
        retrieval_hit = False
        if gold_evidence:
            total_with_evidence += 1
            gold_set = {x.strip() for x in gold_evidence.split(" ; ") if x.strip()}
            if gold_set.intersection(set(retrieved_ids)):
                retrieval_hit = True
                retrieval_hits += 1

        # 2. Generate answer
        if dry_run:
            answer = q.get("reference_answer", "")
        else:
            rag_query = format_rag_prompt(question_text, retrieved_chunks)
            answer = ask(agent_cfg, rag_query)
            if i < len(questions) - 1:
                time.sleep(1)

        # 3. Score
        scores = score_answer(q, answer)
        scores["retrieved_sections"] = retrieved_ids
        scores["gold_evidence"] = gold_evidence
        scores["retrieval_hit"] = retrieval_hit
        scores["answer"] = answer
        results.append(scores)

        # Accumulate
        total_score += scores["combined_score"]
        mi_total_matched += scores["must_include"]["matched"]
        mi_total_facts += scores["must_include"]["total"]
        if scores["must_not_include"]["violations"] > 0:
            mni_violations += 1
        if scores["decision"]["matched"]:
            decision_matches += 1

        icon = "✅" if scores["combined_score"] >= 0.75 else ("🟡" if scores["combined_score"] >= 0.5 else "❌")
        ret_icon = "🎯" if retrieval_hit else "⚪"
        print(f"[{i+1:2d}/{len(questions)}] {qid} ({lang}) {icon} score={scores['combined_score']:.2f} "
              f"mi={scores['must_include']['matched']}/{scores['must_include']['total']} "
              f"ret={ret_icon} {retrieved_ids[:3]}")

    n = len(questions)
    avg_score = total_score / n
    mi_rate = mi_total_matched / mi_total_facts if mi_total_facts > 0 else 1.0
    recall_rate = retrieval_hits / total_with_evidence if total_with_evidence > 0 else 1.0

    summary = {
        "timestamp": timestamp,
        "step": 2,
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
        "retrieval_recall_rate": round(recall_rate, 4),
        "retrieval_hits": retrieval_hits,
        "total_with_evidence": total_with_evidence,
    }

    # Compare with Step 1 summary if present
    step1_summary_path = os.path.join(RESULTS_DIR, "step1_summary.json")
    step1_data = {}
    if os.path.exists(step1_summary_path):
        with open(step1_summary_path, "r", encoding="utf-8") as f:
            step1_data = json.load(f)

    print("\n" + "=" * 68)
    print("📊 STEP 2 RAG EVALUATION & STEP 1 COMPARISON")
    print("=" * 68)
    s1_score = step1_data.get('average_combined_score', 0)
    s1_mi = step1_data.get('must_include_rate', 0)
    print(f"  Combined Score      : {avg_score:.1%}  (Step 1: {s1_score:.1%}, Δ: +{(avg_score - s1_score):.1%})")
    print(f"  Must-Include Facts  : {mi_rate:.1%}  (Step 1: {s1_mi:.1%}, Δ: +{(mi_rate - s1_mi):.1%})")
    print(f"  Retrieval Quality   : {recall_rate:.1%} ({retrieval_hits}/{total_with_evidence} gold sections retrieved)")
    print(f"  Must-Not-Include    : {n - mni_violations}/{n} questions clean")
    print(f"  Decision Match Rate : {decision_matches}/{n} ({decision_matches/n:.1%})")
    print("=" * 68)

    # Save detailed results and summary
    results_path = os.path.join(RESULTS_DIR, f"step2_{'dry' if dry_run else 'live'}_{timestamp}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(RESULTS_DIR, "step2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Step 2 detailed results saved to: {results_path}")
    print(f"💾 Step 2 summary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_step2_evaluation(dry_run=dry_run)

