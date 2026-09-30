# Evidence-Aware RAG for Multi-Hop Financial QA

Retrieval-augmented QA over SEC filings (10-K, 10-Q, 8-K, earnings releases), built on [FinanceBench](https://arxiv.org/abs/2311.11944). SEC filings are dense with numeric tables, cross-referenced across documents, and written in a vocabulary where two line items can look almost identical on the page while meaning different things - a combination that breaks the assumptions standard RAG retrieval (a sparse/dense retriever over generic chunks, with no reranking) is built around. Two properties of this domain drive that failure, each in a specific, measurable way:

1. **Table precision**: the right answer and a wrong answer can sit a few rows apart in the same table, worded almost the same way. For example, "Net income" and "Net income attributable to noncontrolling interest" are different numbers, but most of the words in those two labels are identical. This defeats keyword matching (BM25) for the obvious reason - it just counts shared words - but it also defeats dense embedding similarity: because the two labels share almost all their tokens, a general-purpose sentence embedding model places their vectors close together in embedding space too, so cosine similarity doesn't separate them either. Neither signal is sensitive to the one phrase that changes the meaning; a signal that reads the question and the candidate together is needed instead.
2. **Multi-hop, redundant evidence**: many questions ask for several numbers at once - e.g. "What was revenue, cost of sales, and total current assets for FY2018?" asks for three separate facts in one question. To make it harder, a given fact is sometimes stated more than once in the corpus (an earnings release and the 10-K filed later can both report the same net income figure). A retriever that just ranks by relevance can fill its results with several restatements of one fact and miss the others the question also needs.

Two concrete cases from this corpus illustrate each failure:

- **Table precision**: for "What was 3M's net income in FY2018?", the correct row is `Net income attributable to 3M: $5,349 million`. Two other rows in the same filing are one or two words away from that label but report completely different figures: `Net income attributable to noncontrolling interest: $14 million` (the portion of earnings that belongs to non-3M owners of 3M's subsidiaries, not 3M's own income) and `Net income including noncontrolling interest: $5,363 million` (the combined total before that split is subtracted back out). A retriever scoring on word overlap sees three rows that are almost entirely the same text and has no reason to prefer the first one - it would need to notice that "attributable to noncontrolling interest" flips the meaning entirely. This is exactly the kind of near-duplicate row the cross-encoder reranker in RQ1 has to learn to tell apart.
- **Multi-hop, redundant evidence**: the question *"What was 3M's cash and cash equivalents, cost of sales, total current assets and total current liabilities in FY2018?"* needs four distinct facts. Queried as one bundled string, the pipeline's own diagnostic showed the gold chunk for "cash and cash equivalents" missing from the top 100 results entirely - it lost out to chunks that partially matched the other three metrics' vocabulary - even though it ranked #1 when the same fact was queried alone. Separately, a redundant-hop question about net income can have its top-10 results fill up with 3-4 chunks all restating the same net income figure (earnings release, 10-K, investor presentation excerpt), crowding out the chunk containing a different required fact.

This project investigates two research questions against these properties:

- **RQ1**: does a domain fine-tuned cross-encoder reranker improve retrieval over BM25+dense fusion? Result: **recall@10 0.738 → 0.942 (+27.6%)** on a held-out test set.
- **RQ2**: can a reranker be trained to explicitly maximize *coverage* of distinct required facts, rather than raw relevance, for multi-hop questions with redundant evidence? Result: **recall@10 0.238 → 0.316 (+32.9%), coverage@10 0.497 → 0.576 (+15.9%)**, on a held-out redundant-hop test set.

Full technical writeups: [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md), [`RQ1_RESULTS.md`](RQ1_RESULTS.md), [`RQ2_RESULTS.md`](RQ2_RESULTS.md).

## Dataset

**Corpus**: 146,821 chunks from 84 SEC filings across 32 companies (2015-2024). PDFs are converted to structured markdown via [Docling](https://github.com/docling-project/docling), then split into two chunk types: table **rows** (one row = one chunk) and narrative **paragraphs**. Row-level granularity is used for tables because the correct answer is often a single row among dozens of visually similar ones; paragraph-level granularity is used for narrative text because evidence there tends to span multiple sentences.

**What a chunk actually contains**: each chunk is a record with a stable ID, a pointer to its parent, and metadata alongside the text itself - not just a raw string. A real table-row chunk from 3M's FY2018 10-K looks like:

```
chunk_id:        3M_2018_10K__block188__row0
parent_id:       3M_2018_10K__block188
chunk_type:      table_row
entity:          3M
fiscal_period:   2018
section_title:   Net Income Attributable to Noncontrolling Interest:
subsection_title: (Millions) | 2018 | 2017 | 2016     <- the table's column headers
raw_text:        Net income attributable to noncontrolling interest | $ 14 $ | 11 | $ 8
```

**Parent-child indexing**: every row cut from the same source table shares one `parent_id` (the ID of that table as a whole), while each row gets its own `chunk_id`. Retrieval and reranking only ever operate on the row-level `chunk_id`s - the `parent_id` isn't used to widen what gets retrieved, but it does mean any row can be traced back to the exact table it came from, which is how the corpus's whole-table lookup view (`data/processed/coarse_chunks.parquet`) is built without re-parsing anything.

**Contextual chunks (why `raw_text` alone isn't what gets indexed)**: `raw_text` for a table row is just the row itself and doesn't repeat the company name or year - a row can say `$ 14 $ | 11 | $ 8` with no company mentioned at all. Indexed and embedded in isolation, that row would be retrievable for the wrong company's question just as easily as the right one. To fix this, both the BM25 index and the dense embeddings are built over `f"{entity}, FY{fiscal_period}, {section_title}: {raw_text}"` instead of `raw_text` alone - so the example chunk above is actually indexed as `"3M, FY2018, Net Income Attributable to Noncontrolling Interest:: Net income attributable to noncontrolling interest | $ 14 $ | 11 | $ 8"`. This gives every chunk's own text a chance to match a question's company/year mention, on top of the separate regex-based entity/year filtering described below.

**"Hop" and "hop size"** - two independent properties used to describe a question. A **hop** is one distinct fact the question needs (a question needing revenue and total assets has 2 hops). **Hop size** is how many chunks in the corpus contain that fact - hop size 1 means the fact appears in exactly one place; hop size 2+ means it's restated in more than one chunk (e.g. the same net income figure reported in both an earnings release and the subsequent 10-K). `coverage@k` and `recall@k` are mathematically identical whenever every hop has size 1; they diverge only on redundant-hop questions, which is the subject of RQ2.

**Question types**, with one example each:

| Type | Hops | Hop size | Example |
|---|---|---|---|
| Single-metric (factual) | 1 | 1 | *"What was 3M's revenue in FY2018?"* |
| Multi-metric (factual, multi-hop) | 2-4 | 1 | *"What was 3M's cash and cash equivalents, cost of sales, total current assets and total current liabilities in FY2018?"* |
| Narrative | 1 | 1 | FinanceBench questions whose evidence is prose rather than a table cell |
| Cross-distribution (analytical) | varies | 1 | *"What is the FY2017-FY2019 3-year average of capex as a % of revenue for Activision Blizzard?"* - requires locating multiple raw figures and computing a derived ratio |
| Redundant-hop | 2+ | 2+ | Same shape as multi-metric, but at least one fact is restated in more than one chunk (e.g. an earnings release and the subsequent 10-K both reporting the same net income figure) |

**Test sets used below**: a 591-question training set and a 185-question held-out set (built from 7 metrics never used in training, for a same-distribution generalization check) for RQ1; a 32-question redundant-hop set (mined from the corpus, verified by exact numeric-value matching) for RQ2; natural FinanceBench train/val splits for cross-distribution generalization checks.

## Method

### Stage 1: BM25 + dense fusion (baseline)

- Sparse: BM25 (`bm25s`), NLTK's 198-word stopword list, each chunk's text prefixed with `entity, FYyear, section_title:`.
- Dense: `sentence-transformers/all-mpnet-base-v2` embeddings (off-the-shelf, not fine-tuned), cosine similarity.
- Entity/year filtering: regex-based company and fiscal-year detection restricts candidates to the detected company/year.
- **Fusion**: BM25 and cosine similarity scores are on different, incomparable scales (BM25 is an unbounded term-weighting score; cosine similarity is bounded in [-1, 1]), so they can't be combined directly - a method with naturally larger numbers would dominate the sum regardless of which one is actually more informative. Both are first min-max normalized to a shared [0, 1] scale within their own ranked list:
  ```
  norm(s) = (s - min(scores)) / (max(scores) - min(scores))
  ```
  and then combined as a **convex combination** - a weighted sum whose weights are non-negative and sum to 1, so the result is guaranteed to stay within the range of its inputs rather than overshooting either one:
  ```
  score = α · bm25_norm + (1-α) · dense_norm,   α=0.4
  ```
  `α=0.4` was found by sweeping α and evaluating recall@10; this fusion outperforms BM25 alone, dense alone, and reciprocal rank fusion (an alternative, rank-based fusion method) on this corpus.

### Query decomposition for multi-metric questions

Bundling several metric names into one query dilutes BM25/dense term-overlap scoring for each individual metric against the vocabulary of the others also present in the query. Diagnosed directly: for the multi-metric example above, the gold chunk for "cash and cash equivalents" was absent from the top 100 results when bundled with the other three metrics, but ranked #1 when queried alone.

**Fix**: detect the metrics named in a question, split into one single-metric sub-question per metric, retrieve and rerank each independently, and interleave (round-robin, not concatenated) their top results into one combined ranking. Results (measured with the cross-encoder already applied on both sides, to isolate the decomposition/interleaving change itself) are reported in the RQ1 results table below, since this is a refinement on top of RQ1's reranked pipeline rather than a separate stage.

### Stage 2 (RQ1): cross-encoder reranking

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) processes the query and a candidate jointly via self-attention and outputs one relevance score, as opposed to a bi-encoder's separately-computed, distance-compared embeddings. Fine-tuned via **RankNetLoss** (Burges et al., 2005) on domain-informed hard negatives - the pipeline's own top-ranked *incorrect* results, not random negatives.

For a pair (i, j) where i is more relevant than j:
```
P_ij = 1 / (1 + exp(-σ(s_i - s_j)))
L = -log(P_ij)
```
Minimizing this pushes the relevant document's score up and the hard negative's down. Trained on 591 questions / 3,682 (question, document) pairs, 5 epochs.

**Score blending**: rather than fully trusting the reranker, its score is blended with the stage-1 rank: `final = β · reranker_score + (1-β) · stage1_rank`. Swept over `β ∈ {0.0, 0.3, 0.5, 0.7, 1.0}`:

| Test set | β=1.0 (full trust) | β=0.3 |
|---|---|---|
| Held-out, same distribution (185 q) | **+64% P@5** | +40% P@5 |
| Cross-distribution, natural FinanceBench (30 q) | **-21% P@5 (regression)** | +5% P@5 |

`β=1.0` is optimal for factual-lookup questions but regresses on analytically-phrased questions; `β=0.3` is positive on every test set and is the recommended default when the query distribution is uncertain.

### Stage 3 (RQ2): coverage-aware LambdaMART

Standard LambdaRank does not optimize NDCG by writing it directly into a loss (NDCG is non-differentiable). Instead, for each pair (i, j), it computes how much a target metric would change if their ranks were swapped, and scales an ordinary pairwise gradient by that amount:
```
ρ_ij = 1 / (1 + exp(σ(s_i - s_j)))
λ = σ · ρ_ij · |Δmetric|
```
This mechanism is metric-agnostic. RQ2 substitutes `Δcoverage@k` for the standard `ΔNDCG`. `coverage@k` is a set-membership function (a hop is covered iff any of its chunks is in the top-k), so swapping two candidates that are both inside, or both outside, the top-k provably cannot change it - only a pair straddling the rank-k boundary can. This makes `Δcoverage@k` exact and cheap to compute, and gives zero gradient to two redundant copies of an already-covered hop competing with each other.

**Training**: implemented from scratch (`sklearn.tree.DecisionTreeRegressor` in a manual gradient-boosting loop - LightGBM/XGBoost require `libomp`, unavailable in this environment, and a custom objective needs gradient control those libraries don't expose). Two choices were necessary to get a result that generalizes:
- **Warm start**: the ensemble is initialized to the stage-1 fused score, not zero, and learns only a small correction (`max_depth=2`, `min_samples_leaf=25`, `learning_rate=0.05`). An earlier, un-warm-started version with richer features memorized company-specific score patterns from the ~26 available training examples and did not transfer to new companies' filings.
- **Feature set excludes the cross-encoder's score.** An earlier version that included it mostly learned to copy that one feature, and it doubled as a contamination path (44% of the mined training questions' gold chunks were the same chunks the cross-encoder was fine-tuned on in RQ1).

| Feature | Description |
|---|---|
| `bm25_norm`, `dense_norm`, `cc_score` | Stage-1 signals |
| `is_value_dup_of_higher_ranked` | Does this candidate share an extracted numeric value with a candidate already ranked above it |
| `entity_match`, `year_match` | Detected company/year match |
| `is_table_row` | Table row vs. narrative |
| `lexical_overlap` | Fraction of question content words present in the candidate |

## Results

**RQ1** (185-question held-out set):

| | Recall@10 | Coverage@10 | NDCG@10 |
|---|---|---|---|
| Baseline (stage 1 only) | 0.738 | 0.741 | 0.522 |
| + cross-encoder (β=1.0) | **0.942 (+27.6%)** | **0.945 (+27.5%)** | **0.842 (+61.3%)** |
| + query decomposition, multi-metric subset only (n=40)¹ | **0.969 (+20.2%)** | **0.969 (+20.2%)** | **0.856 (+13.8%)** |

¹ Evaluated only on the 40 multi-metric questions in the 185-question set (the subset query decomposition applies to), against that same subset's bundled-query + cross-encoder numbers (recall@10=0.806, coverage@10=0.806, NDCG@10=0.753) - not against the full-set baseline row above, since the two rows use different denominators.

**RQ2** (32-question redundant-hop set, 5-fold cross-validation, out-of-fold):

| | Recall@10 | Coverage@10 | NDCG@5 |
|---|---|---|---|
| Baseline (stage 1 only) | 0.352 | 0.497 | 0.260 |
| + coverage-aware LambdaMART | **0.438 (+24.4%)** | **0.576 (+15.9%)** | **0.331 (+27.1%)** |

**Full comparison across question types** (baseline → +LambdaMART → +CE, all held-out):

| Tier (n) | Baseline | +LambdaMART | +CE |
|---|---|---|---|
| Single-metric (139), NDCG@5 | 0.557 | 0.652 | **0.888** |
| Multi-metric (40), NDCG@5 | 0.297 | 0.434 | **0.719** |
| Redundant-hop (32), NDCG@5 | 0.260 | 0.331 | **0.483** |

The cross-encoder is the strongest single scorer on every tier tested; LambdaMART's role is as a cheaper alternative when the cross-encoder is unavailable, not a per-question routing choice when it is (a score-blend sweep between the two, `final = β·CE + (1-β)·LambdaMART`, found pure CE (`β=1.0`) dominates monotonically on every tier, including the redundant-hop tier LambdaMART targets).

### Ablation: is the coverage-specific objective load-bearing, or would any regularized booster do?

A second LambdaMART was trained with identical features, warm-start, and regularization, substituting the standard `ΔNDCG@k` gradient for `Δcoverage@k`:

| Tier (n) | Baseline | Coverage-aware LambdaMART | Plain NDCG-objective LambdaMART |
|---|---|---|---|
| Multi-metric (40), Coverage@5 | 0.348 | **0.508** | 0.171 (worse than baseline) |
| Redundant-hop (6, held-out), Coverage@5 | 0.375 | **0.708** | 0.042 (worse than baseline) |

The plain-NDCG version performs worse than no reranking at all, not merely worse than the coverage-aware version - the objective modification is load-bearing, not incidental. Interpretation: standard NDCG's gradient is denser (it fires on every pair where one candidate is more relevant and either is within the top-k), giving it more surface area to overfit given only 145 training questions; `Δcoverage@k`'s sparser, boundary-crossing-only gradient acts as an implicit regularizer.

## Limitations

- RQ2's effect size is modest and validated on a small pool (32 questions, roughly a dozen distinct underlying filings).
- Stages 2 and 3 are not combined into one serving pipeline (see the score-blend result above).
- Retrieval-metric improvements are necessary but not sufficient for end-to-end answer accuracy; no downstream LLM generation was evaluated in this environment.
- Several additional reranking approaches were tested for the redundant-hop tier (a training-free value-similarity penalty, a submodular facility-location formulation grounded in the diversified-retrieval literature) and did not outperform the cross-encoder alone; full results in `RQ2_RESULTS.md`.

## Repository structure

```
scripts/          retrieval pipeline, corpus construction, fine-tuning, RQ2 mining/training
eval/              precision/recall/NDCG and coverage@k metrics
data/processed/    question sets, gold labels, RQ2 datasets, generated prompts
```
