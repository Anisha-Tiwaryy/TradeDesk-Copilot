"""
Recurring-issue -> engineering-backlog generator.

This is the piece of TradeDesk Copilot that answers the JD's "partner with
engineering and product, turn customer feedback into fixes" requirement,
which none of the individual-ticket triage/RAG pieces do on their own:
individually-resolved tickets don't, by themselves, tell engineering
anything is systemically wrong. This module looks ACROSS tickets to find
that signal.

Approach:
  1. Embed every ticket (via the same pluggable Embedder used by the RAG
     index).
  2. Cluster with agglomerative clustering on cosine distance, with no fixed
     k -- clusters emerge from how similar ticket text actually is, capped
     at a tight distance threshold so only near-duplicate/near-identical
     complaints cluster together (paraphrase-level, not just same-category).
  3. Any cluster at or above MIN_CLUSTER_SIZE is treated as a recurring
     issue worth an engineering ticket, not just a support answer.
  4. Each qualifying cluster is scored for priority from volume + average
     urgency, and rendered into a structured EngineeringFeedbackTicket with
     example excerpts and a suggested owning team -- the actual artifact a
     support team would hand to engineering.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .embeddings import get_embedder
from .triage import load_tickets, _urgency_score, URGENT_KEYWORDS

DATA_DIR = Path(__file__).parent.parent / "data"

MIN_CLUSTER_SIZE = 15
DISTANCE_THRESHOLD = 0.12  # cosine distance; tight -> near-duplicate complaints only

CATEGORY_OWNER_TEAM = {
    "Trade Execution": "Trading Systems Engineering",
    "Margin & Risk": "Risk Engineering",
    "API & Integration": "Platform/API Engineering",
    "Account & KYC": "Compliance Engineering",
    "Payments": "Payments Engineering",
    "Fees & Billing": "Billing Engineering",
    "Platform & How-To": "Product / Platform UX",
}

# Not every recurring cluster is the same kind of problem. A cluster of
# tickets all reporting the same failure ("my stop loss didn't trigger") is
# a bug for engineering to fix. A cluster of tickets all ASKING the same
# question ("how does margin call work?") isn't a bug at all -- it's a gap
# in self-serve documentation/auto-response coverage, and the right fix is a
# knowledge-base article or an addition to the auto-response templates
# (src/triage.py), not an engineering ticket. Routing both into one
# undifferentiated "backlog" would bury real bugs under FAQ volume, so
# clusters are split by type using a simple, auditable heuristic.
FAILURE_KEYWORDS = [
    "rejected", "reject", "didn't", "did not", "doesn't", "does not",
    "failed", "fail", "error", "wrong", "stuck", "delayed", "delay",
    "broken", "bug", "higher than", "less than", "not showing",
    "got charged", "put on hold", "auto-closed", "dropped",
]


def _classify_cluster_type(text: str) -> str:
    low = text.lower()
    if any(kw in low for kw in FAILURE_KEYWORDS):
        return "recurring_bug"
    if "?" in text:
        return "recurring_question"
    return "recurring_bug"


_FILLER_OPENERS = re.compile(
    r"^(hi|hey|hey team|hello|so this happened again|not sure who to ask)\b.*?[,.\-]\s*",
    re.IGNORECASE,
)


def _core_text(text: str) -> str:
    """Strip greeting boilerplate and urgency/calm tone sentences before
    embedding for clustering, so recurring-issue detection groups tickets by
    their actual complaint, not by how each customer happened to open or
    close the message. Real ticket text (with greetings, sign-offs, urgency
    framing) is still what's shown to a human in the backlog example
    excerpts -- this normalization only feeds the clustering step.
    """
    text = _FILLER_OPENERS.sub("", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean.split()) < 5:
            continue
        low = s_clean.lower()
        if any(kw in low for kw in URGENT_KEYWORDS):
            continue
        if any(p in low for p in ["no rush", "whenever you get a chance", "just curious", "not blocking anything"]):
            continue
        kept.append(s_clean)
    core = " ".join(kept)
    return core if core else text


@dataclass
class RecurringIssue:
    backlog_id: str
    cluster_type: str  # "recurring_bug" | "recurring_question"
    title: str
    category: str
    owner_team: str
    priority: str
    ticket_count: int
    affected_ticket_ids: list[str]
    first_seen: str
    last_seen: str
    avg_urgency_score: float
    example_excerpts: list[str] = field(default_factory=list)


# Backwards-compatible alias used elsewhere in the codebase/dashboard.
EngineeringFeedbackTicket = RecurringIssue


def _priority_from(count: int, avg_urgency: float, cluster_type: str) -> str:
    if cluster_type == "recurring_question":
        # A documentation/FAQ gap is never an incident -- cap at P2 even at
        # high volume, since nothing is actually broken.
        return "P2" if (count >= 18 or avg_urgency >= 0.3) else "P3"
    if count >= 25 or avg_urgency >= 0.5:
        return "P1"
    if count >= 15 or avg_urgency >= 0.3:
        return "P2"
    return "P3"


def _cluster_title(texts: list[str]) -> str:
    """Short human-readable title from the cluster's shortest representative
    ticket text (shortest is usually closest to the 'core' complaint, before
    urgency markers/generic openers get appended)."""
    shortest = min(texts, key=len)
    title = shortest.strip()
    if len(title) > 110:
        title = title[:107] + "..."
    return title


def generate_backlog(
    tickets: list[dict] = None,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> list[EngineeringFeedbackTicket]:
    tickets = tickets or load_tickets()
    texts = [t["text"] for t in tickets]
    core_texts = [_core_text(t) for t in texts]

    embedder = get_embedder("tfidf-svd")
    embedder.fit(core_texts)
    vecs = embedder.embed(core_texts)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(vecs)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    backlog = []
    backlog_id_counter = 1
    for label, idxs in clusters.items():
        if len(idxs) < min_cluster_size:
            continue

        cluster_tickets = [tickets[i] for i in idxs]
        cluster_texts = [t["text"] for t in cluster_tickets]
        dates = sorted(t["created_at"] for t in cluster_tickets)
        urgencies = [_urgency_score(t["text"]) for t in cluster_tickets]
        # majority category (clusters are near-duplicate text, so this
        # should be near-unanimous, but we don't assume it)
        cats = [t["category"] for t in cluster_tickets]
        top_category = max(set(cats), key=cats.count)

        avg_urgency = float(np.mean(urgencies))
        # Classify on the full representative text, NOT the truncated
        # display title -- a bug caught during testing: truncating to 110
        # chars for display before classifying silently chopped off a
        # trailing "?" on longer questions, misrouting them into the
        # engineering backlog instead of the knowledge-base bucket.
        representative_text = min(cluster_texts, key=len)
        cluster_type = _classify_cluster_type(representative_text)
        title = _cluster_title(cluster_texts)
        priority = _priority_from(len(idxs), avg_urgency, cluster_type)

        if cluster_type == "recurring_question":
            owner_team = "Support Content / Knowledge Base"
            id_prefix = "KB"
        else:
            owner_team = CATEGORY_OWNER_TEAM.get(top_category, "Unassigned")
            id_prefix = "ENG"

        backlog.append(RecurringIssue(
            backlog_id=f"{id_prefix}-{backlog_id_counter:03d}",
            cluster_type=cluster_type,
            title=title,
            category=top_category,
            owner_team=owner_team,
            priority=priority,
            ticket_count=len(idxs),
            affected_ticket_ids=[t["ticket_id"] for t in cluster_tickets],
            first_seen=dates[0],
            last_seen=dates[-1],
            avg_urgency_score=round(avg_urgency, 3),
            example_excerpts=cluster_texts[:3],
        ))
        backlog_id_counter += 1

    backlog.sort(key=lambda b: (-{"P1": 3, "P2": 2, "P3": 1}[b.priority], -b.ticket_count))
    return backlog


def save_backlog(backlog: list[EngineeringFeedbackTicket], path: Path = None):
    path = path or (DATA_DIR / "engineering_backlog.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(b) for b in backlog], f, indent=2)
    return path


if __name__ == "__main__":
    backlog = generate_backlog()
    bugs = [b for b in backlog if b.cluster_type == "recurring_bug"]
    faqs = [b for b in backlog if b.cluster_type == "recurring_question"]

    print(f"Found {len(backlog)} recurring-issue clusters "
          f"(min {MIN_CLUSTER_SIZE} near-duplicate tickets): "
          f"{len(bugs)} engineering bugs, {len(faqs)} knowledge-base gaps.\n")

    print(f"=== ENGINEERING BACKLOG ({len(bugs)}) ===\n")
    for item in bugs:
        print(f"[{item.priority}] {item.backlog_id} -- {item.title}")
        print(f"    owner={item.owner_team}  count={item.ticket_count}  "
              f"avg_urgency={item.avg_urgency_score}  "
              f"window={item.first_seen[:10]} -> {item.last_seen[:10]}")
        print()

    print(f"=== KNOWLEDGE-BASE / AUTO-RESPONSE GAPS ({len(faqs)}) ===\n")
    for item in faqs:
        print(f"[{item.priority}] {item.backlog_id} -- {item.title}")
        print(f"    count={item.ticket_count}  "
              f"window={item.first_seen[:10]} -> {item.last_seen[:10]}")
        print()

    path = save_backlog(backlog)
    print(f"Saved to {path}")
