# Evaluation framework

The MVP exposes `GET /api/evaluation/{id}`. It reports only metrics the current deterministic flow can calculate: number of source-checked claims and exact-source support ratio. It explicitly lists ROUGE, semantic similarity, human preference, and latency benchmarks as unavailable.

## Reproducible evaluation plan

1. Pin GovReport document IDs, reference summaries, preprocessing version, prompt/provider versions, and system commit.
2. Run fixed Smoke, Development, Benchmark, and Full Evaluation splits.
3. Record ROUGE-1/2/L, semantic similarity, citation precision, claim validation precision/recall/F1, groundedness, completeness, latency, errors, and fallback recovery.
4. Review factual consistency and translated outputs with human-labeled samples; do not treat embedding similarity as factual verification.
5. Compare ablations for hybrid vs lexical retrieval, hierarchical vs flat summarization, and validation/regeneration.
6. Never report an unavailable metric as zero or estimate it from another metric.

Dataset download and benchmark execution are intentionally not bundled; the user must supply the dataset under its applicable terms.
