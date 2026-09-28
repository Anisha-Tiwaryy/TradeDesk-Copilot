"""
Ticket triage engine for TradeDesk Copilot.

Pipeline:
  1. Classify the ticket into one of 7 support categories (TF-IDF + Logistic
     Regression), with a confidence score.
  2. Independently score urgency/tone from the raw text (keyword + heuristic
     guardrail -- deliberately NOT the same signal as classifier confidence,
     so a confidently-classified ticket can still be escalated for tone).
  3. Route the ticket to one of four actions based on both signals:
       - AUTO_RESPOND        : high classifier confidence, not urgent
       - ESCALATE_URGENT     : urgent/angry tone detected -> always to a human,
                                regardless of classification confidence
       - FLAG_LOW_CONFIDENCE : classifier isn't sure -> human review, not
                                auto-answered (avoids a wrong auto-response)
       - AUTO_RESPOND_LOW_CONF (never auto-sent; kept for analytics only)

This mirrors a real support-desk automation pattern: never let an automated
system self-certify on an angry customer, and never auto-answer something
the model isn't confident about.
"""
from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = Path(__file__).parent.parent / "data"

CONFIDENCE_THRESHOLD = 0.55

URGENT_KEYWORDS = [
    "urgent", "asap", "immediately", "right now", "losing money",
    "unacceptable", "escalate", "emergency", "critical", "time-sensitive",
    "time sensitive", "capital at risk", "please respond", "waiting for hours",
]

AUTO_RESPONSE_TEMPLATES = {
    "Trade Execution": (
        "Thanks for reaching out. Order rejections are usually caused by "
        "insufficient margin, price moving outside the allowed band, or the "
        "instrument being halted -- the exact reason is in the `reject_reason` "
        "field of your order response. If you'd like, share the ticket/order ID "
        "and we'll check the specific reject_reason code for you."
    ),
    "Margin & Risk": (
        "Margin calls trigger when account equity falls below 100% of used "
        "margin, and auto-liquidation begins at 50% margin level, starting "
        "with the largest unrealized loss. You can see live margin level "
        "under Account > Margin Overview. Let us know if you'd like a walk-"
        "through of your specific account's margin history."
    ),
    "API & Integration": (
        "The REST API allows 120 requests/min for market data and 60/min for "
        "order placement, per API key. If you're seeing 429s under that limit, "
        "it may be a burst pattern rather than a sustained rate -- please share "
        "your API key ID (not the secret) and a rough timestamp so we can pull "
        "the request log."
    ),
    "Account & KYC": (
        "KYC Level 2 (ID + proof of address) is typically reviewed within 1 "
        "business day, and Level 3 (source of funds) within 2-3 business days. "
        "If your account shows a restricted state, it's most often a routine "
        "sanctions/PEP re-screen, cleared once compliance reviews it -- usually "
        "within 1-2 business days."
    ),
    "Payments": (
        "Card/e-wallet deposits post instantly; bank wires take 1-3 business "
        "days. Withdrawals process internally within 1 business day, with an "
        "additional 2-5 business days for bank wires to land, depending on the "
        "receiving bank. Withdrawals above lifetime deposits or to an "
        "unverified destination are held for manual review."
    ),
    "Fees & Billing": (
        "Swap/rollover fees are based on the interest-rate differential "
        "between the pair's two currencies and are charged at 5pm New York "
        "time on open overnight positions. Raw accounts pay $3.50/lot/side "
        "commission with tighter spreads; Standard accounts pay wider spreads "
        "with zero commission."
    ),
    "Platform & How-To": (
        "You can connect via the MT5 bridge from Account Settings > "
        "Connected Platforms, and enable 2FA from Security Settings on both "
        "web and mobile. All platforms (web, desktop, mobile, MT5) share the "
        "same account and execution, so positions update everywhere in real "
        "time."
    ),
}


@dataclass
class TriageResult:
    ticket_id: str
    text: str
    predicted_category: str
    confidence: float
    is_urgent: bool
    urgency_score: float
    action: str
    auto_response: Optional[str]


def _urgency_score(text: str) -> float:
    """Cheap, explainable urgency/tone heuristic (0.0 - 1.0+).

    Not ML -- deliberately rule-based so it's auditable and can't silently
    drift the way a learned tone classifier could. Combines keyword hits,
    exclamation density, and ALL-CAPS shouting.
    """
    t = text.lower()
    score = 0.0
    for kw in URGENT_KEYWORDS:
        if kw in t:
            score += 0.3
    score += 0.15 * text.count("!")
    caps_words = [w for w in text.split() if len(w) > 3 and w.isupper()]
    score += 0.25 * len(caps_words)
    return round(score, 3)


def is_urgent(text: str, threshold: float = 0.3) -> bool:
    return _urgency_score(text) >= threshold


class TriageEngine:
    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=4000, ngram_range=(1, 2), min_df=2, stop_words="english"
        )
        self.model = LogisticRegression(max_iter=1000, C=4.0)
        self._fitted = False
        self.eval_report_: Optional[str] = None
        self.eval_accuracy_: Optional[float] = None

    def fit(self, texts: list[str], labels: list[str], test_size: float = 0.2, random_state: int = 42):
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        Xtr = self.vectorizer.fit_transform(X_train)
        self.model.fit(Xtr, y_train)
        self._fitted = True

        Xte = self.vectorizer.transform(X_test)
        preds = self.model.predict(Xte)
        self.eval_accuracy_ = accuracy_score(y_test, preds)
        self.eval_report_ = classification_report(y_test, preds, zero_division=0)
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("TriageEngine.fit() must be called before predict/route.")

    def predict_category(self, text: str) -> tuple[str, float]:
        self._check_fitted()
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        best_idx = proba.argmax()
        return classes[best_idx], float(proba[best_idx])

    def route(self, ticket_id: str, text: str) -> TriageResult:
        category, confidence = self.predict_category(text)
        urgency = _urgency_score(text)
        urgent = urgency >= 0.3

        if urgent:
            action = "ESCALATE_URGENT"
            response = None
        elif confidence >= self.confidence_threshold:
            action = "AUTO_RESPOND"
            response = AUTO_RESPONSE_TEMPLATES.get(category)
        else:
            action = "FLAG_LOW_CONFIDENCE"
            response = None

        return TriageResult(
            ticket_id=ticket_id,
            text=text,
            predicted_category=category,
            confidence=round(confidence, 4),
            is_urgent=urgent,
            urgency_score=urgency,
            action=action,
            auto_response=response,
        )


def load_tickets(path: Path = None) -> list[dict]:
    path = path or (DATA_DIR / "support_tickets.csv")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_default_engine() -> TriageEngine:
    rows = load_tickets()
    engine = TriageEngine()
    engine.fit([r["text"] for r in rows], [r["category"] for r in rows])
    return engine


if __name__ == "__main__":
    engine = build_default_engine()
    print(f"Held-out accuracy: {engine.eval_accuracy_:.4f}\n")
    print(engine.eval_report_)

    rows = load_tickets()
    from collections import Counter

    actions = Counter()
    for r in rows[:200]:
        result = engine.route(r["ticket_id"], r["text"])
        actions[result.action] += 1
    print("Action distribution on first 200 tickets:", dict(actions))
