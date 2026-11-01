"""Knowledge Base parser and chunker for ZentroMart.

Extracts every section from kb/*.md preserving:
- section_id (e.g. RET-2, KB-00.1)
- doc_type (policy, agent_guide, help, faq, marketing, meta)
- status (current, superseded)
- effective_from
- title and file source

Also formats catalog products from data/products.csv as retrievable items
(products.csv:SKU) for unified retrieval.
"""

import csv
import glob
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(PROJECT_ROOT, "kb")
PRODUCTS_PATH = os.path.join(PROJECT_ROOT, "data", "products.csv")

DOC_TYPE_PRECEDENCE = {
    "policy": 5,
    "agent_guide": 4,
    "help": 3,
    "faq": 2,
    "marketing": 1,
    "meta": 0,
}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML-like frontmatter from markdown text."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta_raw, body = parts[1], parts[2]
    meta = {}
    for line in meta_raw.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip()] = val.strip()
    return meta, body


def load_kb_chunks() -> list[dict]:
    """Parse all sections from all 21 Markdown files in kb/."""
    chunks = []
    md_files = glob.glob(os.path.join(KB_DIR, "*.md"))

    for filepath in sorted(md_files):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        meta, body = parse_frontmatter(content)
        doc_type = meta.get("doc_type", "meta")
        status = meta.get("status", "current")
        effective_from = meta.get("effective_from", "")
        doc_title = meta.get("title", filename)

        # Split on section headers: ## [SECTION_ID] Section Title
        pattern = r"(##\s*\[([a-zA-Z0-9_.-]+)\]\s*(.*?))\n"
        splits = list(re.finditer(pattern, body))

        if not splits:
            # Fallback if document has no section tags
            chunks.append({
                "chunk_id": filename,
                "section_id": filename.replace(".md", ""),
                "title": doc_title,
                "doc_type": doc_type,
                "status": status,
                "effective_from": effective_from,
                "precedence": DOC_TYPE_PRECEDENCE.get(doc_type, 1),
                "text": f"Document: {doc_title}\n\n{body.strip()}",
                "source": filename,
            })
            continue

        for i, match in enumerate(splits):
            section_id = match.group(2).strip()
            section_title = match.group(3).strip()
            start_pos = match.start()
            end_pos = splits[i + 1].start() if i + 1 < len(splits) else len(body)
            section_body = body[start_pos:end_pos].strip()

            header_context = (
                f"Source Document: {doc_title} ({filename})\n"
                f"Type: {doc_type} | Status: {status} | Effective: {effective_from}\n"
                f"Section: [{section_id}] {section_title}\n\n"
            )

            chunks.append({
                "chunk_id": section_id,
                "section_id": section_id,
                "title": section_title,
                "doc_type": doc_type,
                "status": status,
                "effective_from": effective_from,
                "precedence": DOC_TYPE_PRECEDENCE.get(doc_type, 1),
                "text": header_context + section_body,
                "source": filename,
            })

    # Add products.csv rows as indexed items (gold_evidence includes products.csv:SKU)
    if os.path.exists(PRODUCTS_PATH):
        with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sku = row["sku"]
                chunk_id = f"products.csv:{sku}"
                title = row.get("title", "")
                text = (
                    f"Product Catalog Entry: {sku} - {title}\n"
                    f"Brand: {row.get('brand', '')}\n"
                    f"Category: {row.get('return_category', '')}\n"
                    f"Price: ₹{row.get('price_inr', '')}\n"
                    f"Fulfilled By: {row.get('fulfilled_by', '')}\n"
                    f"Assured: {row.get('assured', '')}\n"
                    f"Warranty Months: {row.get('warranty_months', '')}\n"
                    f"Description: {row.get('description', '')}"
                )
                chunks.append({
                    "chunk_id": chunk_id,
                    "section_id": chunk_id,
                    "title": f"Product {sku}: {title}",
                    "doc_type": "policy",  # Product facts act as primary catalog policy
                    "status": "current",
                    "effective_from": "2026-08-01",
                    "precedence": 5,
                    "text": text,
                    "source": "products.csv",
                })

    return chunks

