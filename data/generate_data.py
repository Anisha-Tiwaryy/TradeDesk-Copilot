"""
Synthetic data generator for TradeDesk Copilot.

Generates:
  1. data/support_tickets.csv   -- ~900 realistic support tickets for a B2B/retail
     FX & multi-asset trading platform, across 7 categories, with urgency/tone
     variation and DELIBERATE recurring-issue clusters (so the backlog generator
     has real signal to find).
  2. data/platform_docs.json    -- a small "product documentation" knowledge base
     the RAG assistant retrieves from to answer technical questions.

Everything here is synthetic (randomly assembled from templates + realistic
parameter values). No real customer data, no real company is represented.
"""
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
import csv

random.seed(42)

DATA_DIR = Path(__file__).parent
OUT_TICKETS = DATA_DIR / "support_tickets.csv"
OUT_DOCS = DATA_DIR / "platform_docs.json"

# ---------------------------------------------------------------------------
# 1. Platform documentation corpus (RAG source of truth)
# ---------------------------------------------------------------------------

PLATFORM_DOCS = [
    {
        "doc_id": "ORD-001",
        "title": "Order Types Supported",
        "section": "Trading",
        "text": (
            "The platform supports Market, Limit, Stop, Stop-Limit, and Trailing "
            "Stop orders on all FX pairs and CFDs. Market orders execute at the "
            "best available price with slippage protection capped at 0.5 pips "
            "by default (configurable per account tier). Limit and Stop orders "
            "rest on the order book until triggered or cancelled, and expire "
            "automatically after 90 days unless Good-Till-Cancelled (GTC) is set."
        ),
    },
    {
        "doc_id": "ORD-002",
        "title": "Why an Order Was Rejected",
        "section": "Trading",
        "text": (
            "Orders are rejected for one of five reasons: (1) insufficient "
            "margin, (2) price outside the current market band (>2% from last "
            "traded price), (3) instrument in an auction/halted state, (4) "
            "order size below the minimum lot (0.01 lots) or above the maximum "
            "per-order size for the instrument, or (5) account is in a "
            "restricted trading state (e.g. pending KYC re-verification). "
            "The rejection reason code is always returned in the order "
            "response payload under `reject_reason`."
        ),
    },
    {
        "doc_id": "MGN-001",
        "title": "How Margin Calls Work",
        "section": "Risk & Margin",
        "text": (
            "A margin call is triggered when account equity falls below 100% "
            "of used margin. At 100%, the client receives a margin call "
            "notification. At 50% margin level, the platform begins "
            "auto-liquidating open positions starting with the largest "
            "unrealized loss until margin level is restored above 100%, or no "
            "positions remain. Margin level is recalculated in real time on "
            "every price tick, not on a fixed interval."
        ),
    },
    {
        "doc_id": "MGN-002",
        "title": "Leverage Tiers by Account Type",
        "section": "Risk & Margin",
        "text": (
            "Retail accounts are capped at 1:30 leverage on major FX pairs, "
            "1:20 on minors, and 1:10 on exotics, per standard retail investor "
            "protection rules. Professional (elective) accounts can access up "
            "to 1:200 leverage after passing the professional-client "
            "assessment (trading history, portfolio size, and financial "
            "sector experience criteria)."
        ),
    },
    {
        "doc_id": "API-001",
        "title": "REST API Rate Limits",
        "section": "API & Integration",
        "text": (
            "The REST API allows 120 requests per minute per API key for "
            "market data endpoints, and 60 requests per minute for order "
            "placement/modification endpoints. Exceeding the limit returns "
            "HTTP 429 with a `Retry-After` header in seconds. Sustained abuse "
            "(>5 consecutive 429s) triggers a temporary 10-minute key "
            "suspension. WebSocket streaming does not count against the REST "
            "rate limit and is the recommended path for live price feeds."
        ),
    },
    {
        "doc_id": "API-002",
        "title": "Authenticating API Requests",
        "section": "API & Integration",
        "text": (
            "API requests are authenticated with a bearer token obtained via "
            "OAuth2 client-credentials flow, valid for 24 hours. Tokens must "
            "be refreshed before expiry; there is no grace period after "
            "expiry, and an expired token returns HTTP 401 with error code "
            "`token_expired`. API keys can be scoped to read-only or "
            "trade-enabled at creation time in Account Settings > API Keys."
        ),
    },
    {
        "doc_id": "API-003",
        "title": "Price Feed Latency and Gaps",
        "section": "API & Integration",
        "text": (
            "Live price ticks are delivered over WebSocket with typical "
            "end-to-end latency of 8-15ms during normal market hours. During "
            "high-volatility windows (major economic releases) latency can "
            "spike to 40-60ms and brief gaps (under 200ms) can occur as the "
            "feed re-synchronizes with the liquidity aggregator; this is "
            "expected behavior and does not indicate a feed outage unless the "
            "gap exceeds 3 seconds, which triggers an automatic stale-feed "
            "alert to the client's connected session."
        ),
    },
    {
        "doc_id": "KYC-001",
        "title": "KYC Verification Levels",
        "section": "Account & KYC",
        "text": (
            "Level 1 KYC (email + phone verification) allows demo trading "
            "only. Level 2 KYC (government ID + proof of address, reviewed "
            "within 1 business day) unlocks live trading up to $10,000 "
            "monthly deposit volume. Level 3 KYC (source-of-funds "
            "documentation, reviewed within 2-3 business days) removes the "
            "deposit cap and is required for professional-client status."
        ),
    },
    {
        "doc_id": "KYC-002",
        "title": "Why Trading Is Restricted After Re-verification Request",
        "section": "Account & KYC",
        "text": (
            "Accounts are periodically re-screened against updated sanctions "
            "and PEP (Politically Exposed Person) lists. If a partial match "
            "is flagged, the account is placed in a restricted state "
            "(existing positions remain open and can be closed, but new "
            "orders are blocked) until the compliance team clears the review, "
            "typically within 1-2 business days. This is a routine control, "
            "not an indication of wrongdoing."
        ),
    },
    {
        "doc_id": "PAY-001",
        "title": "Deposit and Withdrawal Processing Times",
        "section": "Payments",
        "text": (
            "Card and e-wallet deposits are credited instantly to available "
            "balance. Bank wire deposits take 1-3 business days depending on "
            "the originating bank. Withdrawals to the original funding "
            "source are processed within 1 business day internally, but bank "
            "wire withdrawals can take an additional 2-5 business days to "
            "appear in the client's account depending on the receiving bank. "
            "Withdrawals are always returned to the original deposit method "
            "where possible, per standard anti-fraud policy."
        ),
    },
    {
        "doc_id": "PAY-002",
        "title": "Why a Withdrawal Was Delayed or Held",
        "section": "Payments",
        "text": (
            "Withdrawals are held for manual review if: the withdrawal "
            "amount exceeds the account's total lifetime deposits, the "
            "destination payment method does not match a previously "
            "verified funding source, or the account has an open compliance "
            "flag. Held withdrawals are reviewed within 1 business day and "
            "the client is notified by email with the specific reason and "
            "any documents required to release it."
        ),
    },
    {
        "doc_id": "FEE-001",
        "title": "Fee Schedule Overview",
        "section": "Fees",
        "text": (
            "Spreads start from 0.1 pips on major FX pairs on the Raw "
            "account tier (plus a fixed commission of $3.50 per lot per "
            "side), or from 1.0 pip with zero commission on the Standard "
            "account tier. Overnight positions incur a swap/rollover fee "
            "based on the interest rate differential between the two "
            "currencies, charged at 5pm New York time. An inactivity fee of "
            "$10/month applies after 90 days with no trading activity."
        ),
    },
    {
        "doc_id": "PLT-001",
        "title": "Supported Trading Platforms",
        "section": "Platform",
        "text": (
            "The platform is accessible via the web trading terminal, native "
            "desktop app (Windows/macOS), mobile apps (iOS/Android), and via "
            "MetaTrader 5 bridge for clients who prefer MT5's charting and "
            "EA (Expert Advisor) ecosystem. All platforms share the same "
            "underlying account and execute against the same liquidity pool, "
            "so a position opened on mobile is immediately visible and "
            "manageable on web or MT5."
        ),
    },
    {
        "doc_id": "PLT-002",
        "title": "Trading Hours and Market Status",
        "section": "Platform",
        "text": (
            "FX markets are open 24 hours a day, 5 days a week, from Sunday "
            "22:00 UTC to Friday 22:00 UTC, with a brief liquidity gap "
            "around the Friday close / Sunday open where spreads widen "
            "significantly. Individual CFD instruments (indices, "
            "commodities) follow their underlying exchange's trading hours, "
            "shown in real time on the instrument's detail page as "
            "Open / Closed / Pre-Market."
        ),
    },
    {
        "doc_id": "SEC-001",
        "title": "Account Security and 2FA",
        "section": "Security",
        "text": (
            "Two-factor authentication (2FA) via authenticator app is "
            "mandatory for all live accounts and cannot be disabled once "
            "enabled without a support-verified identity check. Withdrawal "
            "requests always require a fresh 2FA confirmation regardless of "
            "session age. If a client loses access to their 2FA device, "
            "account recovery requires a video identity verification call "
            "with the security team, which is available 24/5 during market "
            "hours."
        ),
    },
]

# ---------------------------------------------------------------------------
# 2. Ticket generation
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Trade Execution",
    "Margin & Risk",
    "API & Integration",
    "Account & KYC",
    "Payments",
    "Fees & Billing",
    "Platform & How-To",
]

# (category, template, tone_pool) -- {X} placeholders filled at generation time
TEMPLATES = {
    "Trade Execution": [
        "My {order_type} order on {pair} keeps getting rejected, it just says error code {code}.",
        "I tried to close my {pair} position and it failed with a rejection, no clear reason given.",
        "Why did my {order_type} order on {pair} not fill even though the price clearly traded through it?",
        "Getting 'order rejected' every time I try to trade {pair} today, is there an issue on your end?",
        "My stop loss on {pair} didn't trigger even though price went past it, did I lose money because of a bug?",
        "Placed a {order_type} order on {pair} an hour ago and it's still showing pending, is that normal?",
    ],
    "Margin & Risk": [
        "I got margin called on {pair} and my position was liquidated, can you explain why?",
        "My account shows margin level at {margin_pct}% but I don't understand how that's calculated.",
        "Why is my leverage capped at 1:30 on {pair}, I thought professional accounts get more?",
        "Some of my positions were auto-closed overnight, was that a margin call liquidation?",
        "I don't understand the margin call process, can someone walk me through it before it happens again?",
    ],
    "API & Integration": [
        "My API key keeps returning 429 errors even though I'm well under what I think the rate limit is.",
        "Getting token_expired errors on the API randomly, my refresh logic should be handling this.",
        "The WebSocket price feed dropped for a few seconds during the NFP release, is that expected?",
        "Can't authenticate to the REST API, keep getting 401 even with a fresh token.",
        "Our integration keeps hitting rate limits on order placement, what's the actual limit per minute?",
        "Price feed via API seems delayed by several hundred ms compared to the web terminal, is that normal?",
    ],
    "Account & KYC": [
        "My account suddenly shows as restricted and I can't place new orders, why?",
        "I submitted my KYC documents 3 days ago and I'm still on Level 1, what's the delay?",
        "Got an email about re-verification and now I can't trade, is something wrong with my account?",
        "What's the difference between Level 2 and Level 3 KYC, and why do I need Level 3?",
        "My deposit is capped even though I already did ID verification, did something go wrong?",
    ],
    "Payments": [
        "My withdrawal has been pending for 4 days now, when will it actually arrive?",
        "Why was my withdrawal put on hold, I never got a clear reason.",
        "Deposited via bank wire 2 days ago and it's still not showing in my balance.",
        "Tried to withdraw to a new bank account and it got rejected, do I have to use the original one?",
        "Withdrawal amount is less than what I requested, were fees deducted without telling me?",
    ],
    "Fees & Billing": [
        "Got charged an inactivity fee but I traded last month, can you check this?",
        "The swap fee on my overnight {pair} position seems way higher than expected, can you explain it?",
        "What's the actual commission on the Raw account tier, the fee schedule page is confusing.",
        "Why am I paying commission on Standard account, I thought that tier was commission-free?",
        "Overnight rollover charge doesn't match what I calculated, is there a bug in the fee engine?",
    ],
    "Platform & How-To": [
        "How do I connect my account to MetaTrader 5, can't find the bridge setup option.",
        "Is there a way to see if {pair} is currently in pre-market or fully open?",
        "Can I trade the same account from mobile and desktop at the same time?",
        "Where do I enable 2FA, I can't find the setting on the mobile app.",
        "Lost access to my authenticator app, how do I get back into my account?",
    ],
}

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "EUR/GBP", "XAU/USD", "GBP/JPY"]
ORDER_TYPES = ["market", "limit", "stop", "stop-limit", "trailing stop"]
REJECT_CODES = ["INSUFFICIENT_MARGIN", "PRICE_BAND_EXCEEDED", "INSTRUMENT_HALTED", "MIN_LOT_SIZE", "ACCOUNT_RESTRICTED"]

GENERIC_OPENERS = [
    "Hi, I have a question about my account. ",
    "Hey team, quick issue I ran into. ",
    "Not sure who to ask, but ",
    "Hello, ",
    "So this happened again today -- ",
    "",
    "",
    "",
]

# Ambiguous / cross-category tickets: genuinely hard to classify from text
# alone, on purpose, so held-out accuracy reflects real-world messiness
# rather than a toy dataset with perfectly separable categories.
AMBIGUOUS_TEMPLATES = [
    ("Trade Execution", "Margin & Risk",
     "My order on {pair} got rejected and now I'm also seeing a margin warning, not sure which issue to report."),
    ("API & Integration", "Trade Execution",
     "Placed an order through the API and it shows filled on my end but I don't see it in the web terminal, is that a sync issue?"),
    ("Account & KYC", "Payments",
     "My withdrawal is stuck and support mentioned something about my verification level, can you clarify what's blocking it?"),
    ("Fees & Billing", "Margin & Risk",
     "The swap charge on my position seems to have affected my margin level more than expected, is that normal?"),
    ("Platform & How-To", "API & Integration",
     "Trying to set up the MT5 bridge but also want to use the REST API at the same time, is that supported together?"),
    ("Account & KYC", "Trade Execution",
     "Can't place any new orders today and I recently updated my ID documents, are these related?"),
]


def corrupt(text: str) -> str:
    """Light, realistic noise: a couple of common typos/shorthand swaps."""
    swaps = [
        ("I am", "Im"), ("cannot", "cant"), ("do not", "dont"),
        ("please", "pls"), ("thanks", "thx"), ("account", "acct"),
        ("because", "bc"), ("today", "2day"),
    ]
    for a, b in swaps:
        if a in text and random.random() < 0.5:
            text = text.replace(a, b, 1)
    return text


URGENT_MARKERS = [
    "This is URGENT, I'm losing money right now.",
    "Need this fixed ASAP, it's affecting my live trading.",
    "Please respond immediately, this is time-sensitive.",
    "I've been waiting for hours and this is unacceptable.",
    "Escalate this now, I have significant capital at risk.",
]
CALM_MARKERS = [
    "No rush, just wanted to understand what happened.",
    "Whenever you get a chance to look into this, thanks.",
    "Just curious about this, not blocking anything.",
    "",
    "",
]

random.seed(42)


# Phrase-level synonym swaps applied probabilistically after template fill.
# This gives combinatorial wording diversity across the dataset without
# hand-authoring dozens of near-duplicate templates, so baseline (one-off)
# complaints don't accidentally cluster as tightly as the deliberately
# injected recurring-issue clusters do.
PARAPHRASE_SWAPS = [
    ("keeps getting rejected", ["keeps getting rejected", "won't go through", "keeps failing", "gets declined every time"]),
    ("can you explain why", ["can you explain why", "could you tell me why", "why is that", "not sure why"]),
    ("is that normal", ["is that normal", "is that expected", "should I be worried about that", "is this a known issue"]),
    ("did something go wrong", ["did something go wrong", "is something broken", "what happened here", "is this a mistake"]),
    ("what's the delay", ["what's the delay", "why is it taking so long", "what's holding it up", "why hasn't this moved"]),
    ("no clear reason given", ["no clear reason given", "with no explanation", "and I don't know why", "without any details"]),
    ("can you check this", ["can you check this", "could you look into this", "can someone review this", "please look into it"]),
    ("is there an issue on your end", ["is there an issue on your end", "is this a platform issue", "is something wrong on your side", "is this a known problem"]),
]


def paraphrase(text: str) -> str:
    for phrase, alternatives in PARAPHRASE_SWAPS:
        if phrase in text and random.random() < 0.7:
            text = text.replace(phrase, random.choice(alternatives), 1)
    return text


def fill(template):
    filled = template.format(
        order_type=random.choice(ORDER_TYPES),
        pair=random.choice(PAIRS),
        code=random.choice(REJECT_CODES),
        margin_pct=random.choice([45, 60, 80, 95, 110]),
    )
    return paraphrase(filled)


def make_ticket(ticket_id, category, created_at, forced_text=None, forced_urgent=None):
    if forced_text is not None:
        text = forced_text
    else:
        template = random.choice(TEMPLATES[category])
        text = fill(template)

    text = random.choice(GENERIC_OPENERS) + text

    is_urgent = forced_urgent if forced_urgent is not None else (random.random() < 0.22)
    if is_urgent:
        text = text + " " + random.choice(URGENT_MARKERS)
    else:
        marker = random.choice(CALM_MARKERS)
        if marker:
            text = text + " " + marker

    if random.random() < 0.25:
        text = corrupt(text)

    return {
        "ticket_id": ticket_id,
        "created_at": created_at.isoformat(timespec="seconds"),
        "category": category,
        "text": text.strip(),
        "true_urgent": is_urgent,
    }


def generate():
    rows = []
    start = datetime(2026, 8, 1, 9, 0, 0)
    ticket_id = 1

    # --- baseline ticket volume, spread across categories and time ---
    for _ in range(700):
        category = random.choices(
            CATEGORIES,
            weights=[0.22, 0.14, 0.13, 0.13, 0.14, 0.12, 0.12],
        )[0]
        created_at = start + timedelta(
            minutes=random.randint(0, 60 * 24 * 30)  # spread over ~30 days
        )
        rows.append(make_ticket(f"TCK-{ticket_id:05d}", category, created_at))
        ticket_id += 1

    # --- deliberate recurring-issue clusters (signal for the backlog generator) ---
    # Cluster A: a real, specific, recurring bug -- API 429s under the documented limit
    cluster_a_text = (
        "My API key keeps returning 429 rate limit errors even though I'm placing "
        "under 60 order requests per minute, well under the documented limit."
    )
    for _ in range(34):
        created_at = start + timedelta(minutes=random.randint(60 * 24 * 10, 60 * 24 * 30))
        rows.append(
            make_ticket(
                f"TCK-{ticket_id:05d}", "API & Integration", created_at,
                forced_text=cluster_a_text, forced_urgent=(random.random() < 0.4),
            )
        )
        ticket_id += 1

    # Cluster B: stop-loss not triggering on XAU/USD during fast moves
    cluster_b_text = (
        "My stop loss on XAU/USD did not trigger during a fast price move, the "
        "position stayed open well past my stop level and I lost more than expected."
    )
    for _ in range(27):
        created_at = start + timedelta(minutes=random.randint(60 * 24 * 5, 60 * 24 * 28))
        rows.append(
            make_ticket(
                f"TCK-{ticket_id:05d}", "Trade Execution", created_at,
                forced_text=cluster_b_text, forced_urgent=(random.random() < 0.6),
            )
        )
        ticket_id += 1

    # Cluster C: swap fee on overnight GBP/JPY looks wrong to clients
    cluster_c_text = (
        "The overnight swap fee charged on my GBP/JPY position is much higher than "
        "what the fee schedule implies, can you check if the fee engine is correct?"
    )
    for _ in range(19):
        created_at = start + timedelta(minutes=random.randint(60 * 24 * 12, 60 * 24 * 29))
        rows.append(
            make_ticket(
                f"TCK-{ticket_id:05d}", "Fees & Billing", created_at,
                forced_text=cluster_c_text, forced_urgent=(random.random() < 0.15),
            )
        )
        ticket_id += 1

    # --- ambiguous / cross-category tickets: real-world label noise ---
    # These genuinely could go either way; the "true" category is picked
    # randomly between the two plausible ones, the way a human labeler's
    # judgment call would land differently ticket to ticket.
    for _ in range(110):
        cat_a, cat_b, template = random.choice(AMBIGUOUS_TEMPLATES)
        true_category = random.choice([cat_a, cat_b])
        created_at = start + timedelta(minutes=random.randint(0, 60 * 24 * 30))
        rows.append(
            make_ticket(f"TCK-{ticket_id:05d}", true_category, created_at, forced_text=fill(template))
        )
        ticket_id += 1

    random.shuffle(rows)
    rows.sort(key=lambda r: r["created_at"])
    return rows


def main():
    rows = generate()
    with open(OUT_TICKETS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id", "created_at", "category", "text", "true_urgent"])
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT_DOCS, "w", encoding="utf-8") as f:
        json.dump(PLATFORM_DOCS, f, indent=2)

    print(f"Wrote {len(rows)} tickets to {OUT_TICKETS}")
    print(f"Wrote {len(PLATFORM_DOCS)} doc chunks to {OUT_DOCS}")


if __name__ == "__main__":
    main()
