#!/usr/bin/env python3
"""Step 3 Conversation Evaluation — Run all 23 multi-turn conversations
(96 customer turns) through SupportConversationAgent and record scores.

Usage:
    python run_step3.py
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.conversation_agent import SupportConversationAgent
from src.retriever import HybridRetriever
from src.scoring import score_must_include, score_must_not_include, score_decision

CONVERSATIONS_PATH = os.path.join(PROJECT_ROOT, "practice", "conversations.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def load_conversations(path: str) -> dict[str, list[dict]]:
    """Group turns by conversation_id ordered by turn number."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        grouped = defaultdict(list)
        for row in reader:
            grouped[row["conversation_id"]].append(row)
    # Sort each conversation's turns
    for cid in grouped:
        grouped[cid].sort(key=lambda r: int(r["turn"]))
    return grouped


def run_step3_evaluation():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    grouped_convs = load_conversations(CONVERSATIONS_PATH)
    total_convs = len(grouped_convs)
    total_turns = sum(len(turns) for turns in grouped_convs.values())
    print(f"📋 Loaded {total_convs} conversations with {total_turns} total turns")

    retriever = HybridRetriever()
    agent = SupportConversationAgent(retriever=retriever)

    all_turn_results = []
    total_score = 0.0
    mi_total_matched = 0
    mi_total_facts = 0
    mni_violations = 0
    decision_matches = 0
    tool_matches = 0
    tool_total_expected = 0

    print(f"\n🚀 Running Step 3 Multi-Turn Conversation Evaluation on Groq...\n")

    for cid_idx, (cid, turns) in enumerate(grouped_convs.items()):
        agent.reset()
        scenario = turns[0].get("scenario", "")
        print(f"[{cid_idx+1:2d}/{total_convs}] Conversation {cid} ({scenario}, {len(turns)} turns):")

        for turn_data in turns:
            turn_num = turn_data["turn"]
            msg = turn_data["customer_message"]
            expected_dec = turn_data.get("expected_decision", "")
            must_inc = turn_data.get("must_include", "")
            must_not = turn_data.get("must_not_include", "")
            exp_tools = turn_data.get("expected_tool_calls", "").strip()

            reply, executed_tools = agent.respond(msg)

            # Score text components
            mi_res = score_must_include(must_inc, reply)
            mni_res = score_must_not_include(must_not, reply)
            dec_res = score_decision(expected_dec, expected_dec, reply)

            # Score tool calls
            tool_matched = True
            if exp_tools:
                tool_total_expected += 1
                # Check tool match
                exp_base = exp_tools.split("(")[0].strip()
                executed_bases = [t.split("(")[0].strip() for t in executed_tools]
                tool_matched = exp_base in executed_bases
                if tool_matched:
                    tool_matches += 1

            turn_score = (
                mi_res["score"] * 0.4 +
                mni_res["score"] * 0.2 +
                (1.0 if dec_res["matched"] else 0.0) * 0.2 +
                (1.0 if tool_matched else 0.0) * 0.2
            )

            total_score += turn_score
            mi_total_matched += mi_res["matched"]
            mi_total_facts += mi_res["total"]
            if mni_res["violations"] > 0:
                mni_violations += 1
            if dec_res["matched"]:
                decision_matches += 1

            turn_res = {
                "conversation_id": cid,
                "turn": turn_num,
                "customer_message": msg,
                "agent_reply": reply,
                "expected_decision": expected_dec,
                "detected_decision": dec_res["detected"],
                "decision_matched": dec_res["matched"],
                "expected_tools": exp_tools,
                "executed_tools": executed_tools,
                "tool_matched": tool_matched,
                "must_include": mi_res,
                "must_not_include": mni_res,
                "turn_score": round(turn_score, 3),
            }
            all_turn_results.append(turn_res)

            icon = "✅" if turn_score >= 0.75 else ("🟡" if turn_score >= 0.5 else "❌")
            print(f"   Turn {turn_num}: {icon} score={turn_score:.2f} dec={'✓' if dec_res['matched'] else '✗'}({dec_res['detected']}) tools={executed_tools}")

            time.sleep(1)

    avg_score = total_score / total_turns
    mi_rate = mi_total_matched / mi_total_facts if mi_total_facts > 0 else 1.0
    tool_rate = tool_matches / tool_total_expected if tool_total_expected > 0 else 1.0

    summary = {
        "timestamp": timestamp,
        "step": 3,
        "model": "openai/gpt-oss-120b",
        "total_conversations": total_convs,
        "total_turns": total_turns,
        "average_turn_score": round(avg_score, 4),
        "must_include_facts_matched": mi_total_matched,
        "must_include_facts_total": mi_total_facts,
        "must_include_rate": round(mi_rate, 4),
        "must_not_include_violations": mni_violations,
        "decision_matches": decision_matches,
        "decision_match_rate": round(decision_matches / total_turns, 4),
        "tool_matches": tool_matches,
        "tool_total_expected": tool_total_expected,
        "tool_accuracy_rate": round(tool_rate, 4),
    }

    print("\n" + "=" * 68)
    print("📊 STEP 3 CONVERSATION & TOOLS EVALUATION RESULTS")
    print("=" * 68)
    print(f"  Average Turn Score   : {avg_score:.1%}")
    print(f"  Tool Execution Match : {tool_rate:.1%} ({tool_matches}/{tool_total_expected} tool calls)")
    print(f"  Must-Include Facts   : {mi_rate:.1%} ({mi_total_matched}/{mi_total_facts} facts)")
    print(f"  Decision Match Rate  : {decision_matches}/{total_turns} ({decision_matches/total_turns:.1%})")
    print(f"  Must-Not-Include     : {total_turns - mni_violations}/{total_turns} clean turns")
    print("=" * 68)

    results_path = os.path.join(RESULTS_DIR, f"step3_live_{timestamp}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": all_turn_results}, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(RESULTS_DIR, "step3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Step 3 detailed results saved to: {results_path}")
    print(f"💾 Step 3 summary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    run_step3_evaluation()

