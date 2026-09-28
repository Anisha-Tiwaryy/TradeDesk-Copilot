# Testing notes

30 pytest tests across three modules. Run with:

```bash
python -m pytest tests/ -v
```

## What's covered

**`tests/test_triage.py`** (8 tests)
- Held-out accuracy lands in a realistic band (0.80-0.98), not suspiciously perfect
- Category prediction on a clear-cut example
- Urgency detection: true positive and true negative
- Routing: urgent tickets escalate even with high classifier confidence (never auto-respond to an angry customer, regardless of how sure the model is)
- Routing: low-confidence tickets are flagged for human review, not guessed
- Every category has an auto-response template (would fail loudly if a category were added without one)
- Urgency score is monotonic in keyword/marker density

**`tests/test_rag.py`** (11 tests)
- 7 parametrized retrieval-accuracy tests: each test query must retrieve the *exact* correct source document as the top-ranked result
- Retrieval scores are returned in descending order
- Answers are grounded in (literally derived from) the retrieved text, not generated independently
- Answers always cite their sources
- The extractive fallback explicitly declines to answer rather than guessing, when nothing relevant is retrieved
- The generation backend is honestly labeled (`extractive_fallback` when no API key is present — never silently claims to be a live LLM call)

**`tests/test_backlog.py`** (9 tests)
- All 3 deliberately-injected recurring-issue clusters (API rate-limit bug, stop-loss execution bug, swap-fee complaint) are found with approximately correct ticket counts, from text similarity alone
- Injected bug clusters are classified as bugs, not FAQs
- Every surfaced cluster meets the minimum size threshold
- FAQ-type clusters are never assigned P1 priority (a documentation gap isn't an incident)
- Regression test for the truncation bug (below)
- Boilerplate-stripping (`_core_text`) correctly removes greetings and urgency language before clustering
- Backlog IDs are unique
- Every ticket ID referenced in the backlog actually exists in the source data

## A real bug found and fixed

While writing `tests/test_backlog.py::test_classification_survives_truncation`, I found that `generate_backlog()` was classifying each cluster as a "bug" or "question" using the **truncated 110-character display title**, not the full ticket text.

**Why it broke:** `_cluster_title()` truncates long ticket text to 110 characters (with `...`) for readable dashboard display. `_classify_cluster_type()` was being called on that same truncated string. For a longer question like:

> "Trying to set up the MT5 bridge but also want to use the REST API at the same time, is that supported together for my account?"

...truncation cut the string before the trailing `?`, so the classifier's "does this contain a `?`" check silently failed, and the cluster fell through to the `recurring_bug` default — a documentation-gap question ended up misrouted into the engineering backlog instead of the knowledge-base backlog.

**Fix:** classification now runs on the full, untruncated representative ticket text (`min(cluster_texts, key=len)`), computed *before* the display title is truncated. The two are now separate variables with separate purposes. See the comment in `src/backlog.py` at the fix site, and `test_classification_survives_truncation` for the regression test (asserts on a 130-character question that would previously have been chopped to 110 and misclassified).

## Known limitations (stated honestly, not hidden)

- **Data is synthetic**, generated from templates with paraphrase/typo noise (see `data/generate_data.py`). It is realistic in structure but not real production ticket data, and real customer language would likely have more edge cases than any template-generation process anticipates.
- **The embedding backend used in this build is TF-IDF + truncated SVD, not a transformer.** `src/embeddings.py` includes a complete `SentenceTransformerEmbedder` class, but downloading the model requires network access to huggingface.co, which was not available in this build sandbox. See README.md's "Design decisions" section for detail. This is disclosed, not hidden — the tests and dashboard results above are all against the TF-IDF+SVD backend.
- **RAG generation is extractive, not a live LLM call**, for the same reason (no network access to the Gemini API in this sandbox, and no API key configured). The `src/rag.py` Gemini-calling code path is real and complete but not exercised by these tests.
- **Clustering thresholds (`MIN_CLUSTER_SIZE=15`, `DISTANCE_THRESHOLD=0.12`) were tuned against the known synthetic clusters**, not against a separately labeled validation set — in a real deployment these would need tuning against actual historical incident data, and likely a time-windowed (not whole-dataset) approach.
- **The bug/FAQ classifier in `src/backlog.py` is a keyword heuristic**, not a trained model. It's intentionally simple and auditable (you can read the exact keyword list and know why any cluster was classified the way it was), but it will misclassify edge cases a trained classifier might catch — e.g., a complaint phrased entirely as a question with no failure keyword present.
