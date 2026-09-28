"""
TradeDesk Copilot -- unified Streamlit dashboard.

Four tabs, one per capability this project demonstrates against the
AI Engineering Intern (Customer Engineering) JD:
  1. Ticket Triage        -- "support customers" + "automations to speed
                              up resolutions"
  2. Ask the Platform      -- "answer technical questions", "AI Knowledge"
  3. Engineering Backlog   -- "turn customer feedback into fixes"
  4. Metrics               -- overview of the whole pipeline for a reviewer
"""
import sys
from pathlib import Path

# Let `streamlit run src/dashboard.py` find the `src` package from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import altair as alt

from src.triage import build_default_engine, load_tickets, AUTO_RESPONSE_TEMPLATES
from src.rag import build_default_index, answer_question
from src.backlog import generate_backlog

st.set_page_config(page_title="TradeDesk Copilot", layout="wide")


@st.cache_resource
def get_triage_engine():
    return build_default_engine()


@st.cache_resource
def get_rag_index():
    return build_default_index()


@st.cache_data
def get_tickets():
    return load_tickets()


@st.cache_data
def get_backlog():
    return generate_backlog()


st.title("TradeDesk Copilot")
st.caption(
    "AI support automation for a trading platform's Customer Engineering team -- "
    "ticket triage, technical Q&A, and recurring-issue escalation to engineering."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Ticket Triage", "Ask the Platform", "Engineering Backlog", "Metrics"]
)

# ---------------------------------------------------------------------------
# TAB 1: Ticket Triage
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Live ticket triage")
    st.write(
        "Classifies an incoming ticket, screens it for urgent/angry tone, and "
        "only auto-responds when confident AND calm -- urgent tickets always "
        "go to a human, regardless of classifier confidence."
    )

    engine = get_triage_engine()

    col1, col2 = st.columns([2, 1])
    with col1:
        default_text = "My API key keeps returning 429 errors even though I'm well under the documented limit. This is URGENT, affecting live trading."
        user_text = st.text_area("Paste a support ticket", value=default_text, height=100)

    if st.button("Triage this ticket", type="primary"):
        result = engine.route("MANUAL", user_text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted category", result.predicted_category)
        c2.metric("Confidence", f"{result.confidence:.0%}")
        c3.metric("Action", result.action)

        if result.action == "ESCALATE_URGENT":
            st.error("Urgent/angry tone detected -- routed straight to a human agent, no auto-response.")
        elif result.action == "AUTO_RESPOND":
            st.success("Auto-response sent:")
            st.info(result.auto_response)
        else:
            st.warning("Classifier confidence below threshold -- flagged for human review, not auto-answered.")

    st.divider()
    st.subheader("Held-out evaluation")
    st.text(f"Accuracy: {engine.eval_accuracy_:.4f}")
    with st.expander("Full classification report"):
        st.text(engine.eval_report_)

# ---------------------------------------------------------------------------
# TAB 2: Ask the Platform (RAG)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Ask the platform a technical question")
    st.write(
        "Retrieval-augmented answers grounded in the platform documentation -- "
        "every answer cites its source doc(s), so a human agent can verify "
        "before it reaches a customer."
    )

    index = get_rag_index()
    query = st.text_input("Ask a question", value="How does margin call liquidation work?")

    if st.button("Ask", type="primary"):
        result = answer_question(index, query)
        st.write(f"**Backend:** `{result.generation_backend}`")
        st.info(result.answer)
        st.write("**Sources:**")
        for s in result.sources:
            st.write(f"- `{s.doc_id}` {s.title} (similarity: {s.score:.3f})")

# ---------------------------------------------------------------------------
# TAB 3: Engineering Backlog
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Recurring issues -> engineering backlog")
    st.write(
        "Clusters near-duplicate tickets across the whole ticket volume to find "
        "systemic patterns a single-ticket view can't see, then splits them into "
        "real bugs (engineering-owned) vs. documentation gaps (knowledge-base-owned)."
    )

    backlog = get_backlog()
    bugs = [b for b in backlog if b.cluster_type == "recurring_bug"]
    faqs = [b for b in backlog if b.cluster_type == "recurring_question"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total clusters found", len(backlog))
    c2.metric("Engineering bugs", len(bugs))
    c3.metric("Knowledge-base gaps", len(faqs))

    st.write("### Engineering backlog")
    if bugs:
        df_bugs = pd.DataFrame([{
            "ID": b.backlog_id, "Priority": b.priority, "Title": b.title,
            "Owner": b.owner_team, "Tickets": b.ticket_count,
            "Avg urgency": b.avg_urgency_score,
        } for b in bugs])
        st.dataframe(df_bugs, use_container_width=True, hide_index=True)

        selected = st.selectbox("Inspect a cluster", [b.backlog_id for b in bugs])
        item = next(b for b in bugs if b.backlog_id == selected)
        st.write(f"**{item.title}**")
        st.write(f"Owner: {item.owner_team} | Priority: {item.priority} | "
                 f"{item.ticket_count} tickets, {item.first_seen[:10]} to {item.last_seen[:10]}")
        st.write("Example tickets:")
        for ex in item.example_excerpts:
            st.caption(f"> {ex}")

    st.write("### Knowledge-base / auto-response gaps")
    if faqs:
        df_faqs = pd.DataFrame([{
            "ID": b.backlog_id, "Priority": b.priority, "Title": b.title,
            "Tickets": b.ticket_count,
        } for b in faqs])
        st.dataframe(df_faqs, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4: Metrics
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Pipeline overview")
    tickets = get_tickets()
    df = pd.DataFrame(tickets)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total tickets", len(df))
    c2.metric("Categories", df["category"].nunique())
    c3.metric("Urgent tickets", int(df["true_urgent"].apply(lambda x: x == "True").sum()))

    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    chart = alt.Chart(cat_counts).mark_bar().encode(
        x=alt.X("count:Q", title="Tickets"),
        y=alt.Y("category:N", sort="-x", title=None),
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Data is synthetic (see data/generate_data.py) -- realistic support "
        "tickets for a B2B/retail FX trading platform, generated with "
        "deliberate recurring-issue clusters so the backlog generator has "
        "real signal to detect, plus paraphrase and typo noise so category "
        "classification isn't a trivial 100%-accuracy toy problem."
    )
