# Evaluation results

43 questions (25 specific, 18 broad). Answers: `gpt-6.1-sol`, judge: `gpt-6.1-sol`.
Recall@5 and MRR count retrieved passages from the question's source pages (RAPTOR summaries never count as hits). Context recall, faithfulness, citation accuracy and relevance are LLM-judged.

Documents: OpenStax *Introduction to Python Programming* (397 pages) and *Principles of Data
Science* (569 pages), CC BY-NC-SA 4.0. 50 questions were drafted with `gpt-6-luna`; 7 were removed
in manual review (5 depended on context a real user would not have, 2 were duplicates).
Run on 2026-10-18 through the OpenAI Batch API.

## All questions

| Config | Recall@5 | MRR | Context recall | Faithfulness | Citation accuracy | Answer relevance (1-5) |
|---|---|---|---|---|---|---|
| Dense only (baseline) | 90.7% | 64.0% | 78.3% | 97.7% | 97.4% | 4.28 |
| Hybrid | 86.0% | 70.1% | 80.1% | 100.0% | 98.5% | 4.23 |
| Hybrid + rerank | 97.7% | 82.5% | 83.0% | 100.0% | 100.0% | 4.37 |
| Hybrid + rerank + RAPTOR | 83.7% | 67.4% | 92.1% | 99.9% | 98.2% | 4.60 |

## Specific questions

| Config | Recall@5 | MRR | Context recall | Faithfulness | Citation accuracy | Answer relevance (1-5) |
|---|---|---|---|---|---|---|
| Dense only (baseline) | 92.0% | 65.8% | 92.0% | 96.0% | 95.8% | 4.84 |
| Hybrid | 88.0% | 68.5% | 96.0% | 100.0% | 97.9% | 4.80 |
| Hybrid + rerank | 100.0% | 89.3% | 100.0% | 100.0% | 100.0% | 5.00 |
| Hybrid + rerank + RAPTOR | 88.0% | 80.0% | 100.0% | 100.0% | 98.0% | 4.96 |

## Broad questions

| Config | Recall@5 | MRR | Context recall | Faithfulness | Citation accuracy | Answer relevance (1-5) |
|---|---|---|---|---|---|---|
| Dense only (baseline) | 88.9% | 61.5% | 59.2% | 100.0% | 99.6% | 3.50 |
| Hybrid | 83.3% | 72.4% | 58.0% | 100.0% | 99.3% | 3.44 |
| Hybrid + rerank | 94.4% | 73.1% | 59.4% | 100.0% | 100.0% | 3.50 |
| Hybrid + rerank + RAPTOR | 77.8% | 50.0% | 81.2% | 99.9% | 98.4% | 4.11 |
