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

## Evaluation

I use entity-level micro-averaged precision, recall, and F1 to evaluate the NER system.

A predicted mention is correct only when both the entity label and the exact mention text match a gold mention.

For example, if the gold mention is `("PER", "John Smith")`, then `("PER", "John Smith")` is correct, but `("ORG", "John Smith")` and `("PER", "John")` are wrong.

Each gold mention and predicted mention can be matched only once. If the same correct prediction appears more than once, the extra copies are counted as false positives.

I calculate the metrics using true positives (TP), false positives (FP), and false negatives (FN).

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = (2 * Precision * Recall) / (Precision + Recall)

If the denominator is zero, I return 0.0 for that metric. If there are no gold mentions and no predicted mentions, I return 1.0 for precision, recall, and F1.

For dataset-level evaluation, I first add TP, FP, and FN across all examples. Then I calculate precision, recall, and F1 from these total values. I do not calculate the average of each sentence’s F1 score.

I use this function for dataset-level evaluation:

```python
evaluate_dataset(examples, predicted_mentions_per_example)
```

I tested the evaluation system before developing the model. My tests check perfect predictions, extra predictions, missing predictions, wrong labels, duplicate predictions, empty predictions, case-sensitive text, multi-token entities, and dataset-level micro-averaging.

I run the tests with:

```bash
python -m pytest tests/test_entity_metrics.py -v
python -m pytest tests/test_data_evaluation.py -v
```

## Training

I fine-tune the model using LoRA. It trains only a small number of adapter
parameters instead of updating the whole model.

I use the PaDeLLM data format for this overfit test. Each training example asks
the model to find one specific mention:


Apple hired John Smith in Dublin .
Find the 1st ORG mention.

The target output is:
Apple


### Loss masking

I calculate loss only on the output tokens.

The prompt contains the sentence and the task, for example:


Apple hired John Smith in Dublin .
Find the 1st ORG mention.

I set the labels for these prompt tokens to `-100`, so they do not contribute
to the loss. Only the target mention and the EOS token contribute to the loss.

For the example above, the supervised output is:

Apple</s>


### Overfit test

Before training on the full dataset, I tested the training script on CPU using
the first 20 sentences from the dataset.

I built 20 PaDeLLM training examples and trained the model to memorise them.

The purpose of this test is to check that:

- the LoRA model loads correctly;
- the loss is calculated only on output tokens;
- the loss decreases during training;
- checkpoints are saved;
- training can resume from a checkpoint;
- the trained model can generate the expected target outputs.

### Commands

Train on 20 examples:

```bash
python -m src.training.train_lora --num_epochs 20
```

Resume from a checkpoint:

```bash
python -m src.training.train_lora --num_epochs 20 --resume_from_checkpoint results/overfit_test/checkpoint-400
```

### Generation check

After training, I check 20 generated examples against their expected outputs.

For each example, I print:

- the task;
- the expected output;
- the generated output;
- whether they match exactly.

The overfit test is successful when the training loss becomes close to zero and
the model reproduces the target outputs for the training examples.

### Result

After training on the first 20 CoNLL sentences, the overfit test achieved:

Overfit test: 20/20 exact matches

This confirms that the training pipeline, loss masking, LoRA setup,
checkpointing, and generation evaluation all work correctly.


## Compute budget

I measured the compute budget for AutoReg-Struct by training on 1% of the
AutoReg-Struct training data.

I used 140 examples, which is approximately 1% of the CoNLL-2003 training
sentences. Since my AutoReg-Struct builder creates one training example per
sentence, 140 examples is also 140 training steps for one epoch.

I trained the model for one epoch on a Google Colab T4 GPU.

Setting : Value

Base model : TinyLlama-1.1B-Chat-v1.0  
Fine-tuning method : LoRA  
LoRA rank : 8  
LoRA alpha : 16  
Target modules : q_proj, v_proj  
Sequence cap : 256 tokens  
Batch size : 1  
Gradient accumulation : 1  
Budget-test examples : 140  
Epochs in budget test : 1  
Training steps : 140  
Training time : 21.4 seconds  
Speed : 6.65 steps per second  
Seconds per step : 0.153  
Peak GPU memory : 4.62 GB  
GPU used : Google Colab T4, 15 GB GPU RAM  

The full CoNLL-2003 training set contains approximately 14,000 sentences.
Because AutoReg-Struct creates one example per sentence, a full training epoch
would contain approximately 14,000 training steps.

Using the measured speed, one full epoch should take approximately:

14,000 × 0.153 = 2,142 seconds

This is about 36 minutes per epoch.

Because one epoch is expected to take less than one hour, I can train for two
epochs and still keep the full run within a few GPU-hours. Two epochs would
take approximately 72 minutes.

The peak GPU memory was only 4.62 GB, which is well below the 15 GB available
on the Colab T4 GPU. This means the chosen model size, sequence cap, and batch
size are suitable for the available hardware.

Full-run setting : Chosen value

Training examples : Full AutoReg-Struct training set, approximately 14,000  
Epochs : 2  
Sequence cap : 256 tokens  
Batch size : 1  
Gradient accumulation : 1  
Estimated training time : About 1.2 hours  
Fits available GPU quota? : Yes  



## PaDeLLM Count Prediction

PaDeLLM counter predicts the number of mentions for each entity label
before generating the actual mentions.

For every CoNLL-2003 sentence, the model receives four separate prompts:

- Count the number of ORG mentions
- Count the number of PER mentions
- Count the number of MISC mentions
- Count the number of LOC mentions

Each prompt has a single integer target, such as `0`, `1`, or `2`.
This is separate from the AutoRegStruct baseline and from PaDeLLM Step 2
mention generation.

## Data

The count-prediction examples are created from CoNLL-2003 BIO tags.

For one sentence, the data builder creates four examples: one for each
entity label. The number of mentions is calculated from BIO tags, where
`B-LABEL` starts a mention and following `I-LABEL` tokens continue the
same mention.

For the overfit test, I used 8 sentences from the CoNLL-2003 training
split. Since each sentence produces four count-prediction examples, this
gives 32 training examples.

## Training

I fine-tuned TinyLlama/TinyLlama-1.1B-Chat-v1.0 using LoRA.

- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Target modules: `q_proj` and `v_proj`
- Bias: none
- Task type: causal language modeling
- Batch size: 1
- Gradient accumulation: 1
- Learning rate: 1e-4
- Maximum steps: 600
- Number of overfit sentences: 8
- Number of training examples: 32

Prompt tokens were masked with `-100`, so the model was trained only to
predict the integer count and EOS token.

## Evaluation

After training, I evaluated the model on the same 8 overfit sentences.
The evaluation creates four prompts per sentence, batches them, and
compares each generated output with the gold count from the BIO tags.

An output is counted as correct only if it is exactly an integer and
matches the gold count. Outputs with extra text, empty strings, or
non-integer responses are marked as malformed.

## Commands

overfit 8 sentences
```bash
python -m src.training.padellm_counter_overfit --max_steps 600 --num_sentences 8
```

Evaluation
```bash
python -m src.inference.evaluate_padellm_counter_overfit --adapter_dir results/padellm_counter_overfit --num_sentences 8 --batch_size 4
```

## Result

The PaDeLLM count-prediction model achieved:

- Correct predictions: 32/32
- Exact-match accuracy: 100.00%
- Malformed outputs: 0
- Empty outputs: 0
This confirms that PaDeLLM can predict per-label mention counts
in a batched setting.
