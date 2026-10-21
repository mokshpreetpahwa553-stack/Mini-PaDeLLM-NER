# Mini-PaDeLLM-NER
Reproducing the PaDeLLM-NER paper, implementing a lightweight version optimized for free cloud GPUs.


## Goal

Reproduce the core PaDeLLM-NER pipeline at a scale feasible using free GPU resources.

## Planned scope

- Task: English Named Entity Recognition
- Dataset: CoNLL2003 English subset
- Base model: To be decided
- Fine-tuning: QLoRA
- Inference: Two-step parallel decoding
- Serving: FastAPI and Docker
- Metrics: Entity-level precision, recall, F1, latency, and speedup


## What is not being reproduced

- Exact paper F1 scores
- Full-scale training
- 7B-parameter model experiments
- Chinese-language experiments

## Repository structure

```text
src/          Source code
configs/      Configuration files
data/         Raw and processed datasets
notebooks/    Exploration and experiments
results/      Evaluation results
tests/        Tests
```