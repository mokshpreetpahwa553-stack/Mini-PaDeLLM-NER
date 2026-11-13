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

## PaDeLLM End-to-End Inference

PaDeLLM uses two steps.

In Step 1, the count prediction model predicts how many mentions
there are for each entity label in a sentence.

In Step 2, the mention generation model generates each mention
using a separate prompt for each predicted mention.

For example, if the count model predicts:
ORG : 1  
PER : 0  
MISC : 2  
LOC : 0  

then the mention model receives prompts for:

ORG mention 1  
MISC mention 1  
MISC mention 2  

The outputs from Step 2 are collected as the final predicted
mentions for the sentence.

I built an end-to-end inference script that combines both steps.

The script:

1. Loads TinyLlama.
2. Loads the count-prediction LoRA adapter.
3. Loads the mention-generation LoRA adapter.
4. Loads validation sentences.
5. Creates four count-prediction prompts for each sentence.
6. Runs the count model.
7. Uses the predicted counts to create mention prompts.
8. Runs the mention model.
9. Saves the predicted mentions to a JSON file.

I ran the end-to-end pipeline on 20 validation sentences.

I used the following settings:

Model : TinyLlama-1.1B-Chat-v1.0  
Count adapter : results/padellm_counter_overfit  
Mention adapter : results/padellm_data_builder_overfit  
Number of validation sentences : 20  
Batch size : 4  
Maximum new tokens : 15  
Output file : results/padellm_end_to_end_overfit_predictions.json  

The pipeline saved 41 mention predictions.

The number of predictions is larger than the number of sentences
because each sentence can produce multiple mention-generation
prompts. 

The end-to-end pipeline works correctly. It connects count prediction,
mention prompt construction, mention generation, and JSON output.

## PaDeLLM NER Counting Pipeline
Today, the project was separated into individual training and inference
scripts for two LoRA adapted TinyLlama components:

1. PaDeLLM Data Builder
2. PaDeLLM Entity Counter

The overall pipeline is:

CoNLL sentence
-> Data-builder adapter
-> Entity mentions / structured output
-> Counter adapter
-> Predicted entity counts
-> Comparison with gold counts

## Scripts Prepared

### Data Builder

Training script:

src/training/train_padellm_data_builder.py

Inference script:

src/inference/pred_padellm_data_builder.py

Training output directory:

results/padellm_data_builder

Inference output file:

results/padellm_data_builder/predictions.jsonl

### Counter

Training script:

src/training/train_padellm_counter.py

Inference script:

src/inference/pred_padellm_counter.py

Training output directory:

results/padellm_counter

Inference output file:

results/adellm_counter/predictions.jsonl

## Data Split Policy

Training uses the CoNLL-2003 training split:

load_dataset("lhoestq/conll2003", split="train")

Inference and evaluation use the validation split:

load_dataset("lhoestq/conll2003", split="validation")

The validation split is not used during training.

## Deduplication
Deduplication is applied after inference in the end-to-end evaluation
pipeline.

After generating predictions for the first 20 validation examples,
duplicate prediction records are removed before further analysis or
evaluation.

This prevents identical prediction entries from being counted multiple
times in later evaluation or inspection.

## Model and LoRA Configuration

Base model:

TinyLlama/TinyLlama-1.1B-Chat-v1.0

LoRA configuration:

r = 8
lora_alpha = 16
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj"]
bias = "none"
task_type = "CAUSAL_LM"

Only LoRA adapter parameters are trained; the base TinyLlama weights remain frozen.

## Training Settings

Both training scripts use:

num_train_epochs = 2
learning_rate = 1e-4
per_device_train_batch_size = 1
gradient_accumulation_steps = 1
max_steps = -1

Because max_steps is -1, training is controlled by num_train_epochs.

## Inference Settings

Data-builder inference:

max_new_tokens = 128

Counter inference:

max_new_tokens = 16

The counter needs only a small number of generated tokens because its
expected output is a count.

## Important Design Decision

The data-builder adapter and counter adapter are tested separately before
being connected in an end-to-end pipeline.

This makes it easier to identify whether errors come from:

- Entity-mention extraction
- Counter prediction
- Prompt formatting
- Output parsing
- Adapter loading


## Onestep Training and Infernece

I implemented and validated the OneStep NER pipeline using a causal language model fine-tuned with LoRA.

In OneStep NER, I converted each sentence into four separate examples, one for each entity label:

- `ORG`
- `PER`
- `MISC`
- `LOC`

For each label, I asked the model to extract only the mentions belonging to that label.

### Example

```text
Input: Apple hired John Smith in Dublin
Task:  Find all PER mentions.
Output: John Smith
```

If a label has multiple mentions:

```text
Input: Find all MISC mentions.
Output: German <sep> British
```

If a label has no mentions:

```text
Output: <none>
```

## OneStep Data Builder

I created logic to convert CoNLL-style BIO-tagged sentences into OneStep examples.

For every sentence, I created four examples:

- Find all `ORG` mentions.
- Find all `PER` mentions.
- Find all `MISC` mentions.
- Find all `LOC` mentions.

Each example contains:

- `input`
- `task`
- `output`
- `format`
- `sentence_id`

Example:

```json
{
  "input": "Apple hired John Smith in Dublin",
  "task": "Find all PER mentions.",
  "output": "John Smith",
  "format": "onestep",
  "sentence_id": 0
}
```

### Overfit Training Script
- Loads TinyLlama.
- Applies LoRA fine-tuning.
- Creates OneStep training examples.
- Masks prompt tokens using `-100`.
- Trains only on the target output and EOS token.
- Saves the trained LoRA adapter.
- Tests whether the model can reproduce the exact training targets.

### 3. Overfit Test Result

The overfit test was successful.

```text
Overfit test: 20/20 exact matches
```

The model correctly reproduced:

- Single mentions
- Multiple mentions separated by `<sep>`
- `<none>` outputs
- `ORG`, `PER`, `MISC`, and `LOC` entity types

## Saved Overfit Model

I saved the overfit LoRA adapter to:

```text
results/onestep_overfit
```

## What the Overfit Test Proves

The successful 20/20 overfit result confirms that:

- The OneStep data format is correct.
- The prompt format is consistent.
- Tokenization works correctly.
- Loss masking works correctly.
- LoRA fine-tuning works.
- Model generation works.
- The model can produce the expected output format.
- The model understands `<none>` and `<sep>`.


### 1. Run Full OneStep Training

```bash
python -m src.training.train_onestep \
  --num_epochs 2 \
  --output_dir models/onestep_model
```

### 2. Run OneStep Inference on the Validation Set

```bash
python -m src.inference.pred_onestep \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter_dir results/training_onestep \
  --output_file results/inference_onestep.jsonl \
  --max_new_tokens 50
```

### 3.OneStep Evaluation Script

- Load inference results.
- Group predictions by `sentence_id`.
- Use `task` to determine the entity label.
- Parse generated mentions.
- Convert predictions into `(label, mention)` tuples.
- Compare predictions with gold mentions.
- Compute micro precision, recall, and F1.


### Setup
- Model: TinyLlama-1.1B-Chat-v1.0 + LoRA adapter 
- Data: CoNLL-2003 validation split (`split="validation"`)
- Decoding: greedy, `max_new_tokens=128` 


### How I scored it
- Micro P/R/F1 over (label, mention-string) pairs. I add up TP/FP/FN over all sentences and compute F1 once at the end (not an average of per-sentence F1).
- Gold pairs come from the BIO tags, using the same converter as in training.
- If the parser can't parse an output, I treat that sentence's prediction as **empty**, so all its gold mentions count as false negatives. This is strict on purpose.

### Results (dev)
| Metric | Value |
|---|---|
| Precision | 94.46 |
| Recall | 92.46 |
| **F1** | **93.45** |
| TP / FP / FN | 5494 / 322 / 448 |
| Gold mentions | 5942 |
| Parse failures | 6 / 3250 (0.18%) |
| Total mention count matches gold | 95.1% of sentences |

### Sanity checks I did
- Gold total = 5,942, which is the standard CoNLL-2003 dev count, so the BIO converter and split are right.
- Recomputed P and R by hand from TP/FP/FN, and they match.
- **Hand check:** I scored 24 sentences by hand (dev ids 1541–1562, plus 2184 and 2185) and ran the pipeline on the same records. Expected TP 44, FP 2, FN 24, with 1 parse failure, and the script gave exactly that. The 24 records are frozen in `tests/fixtures/test_hand_check_autoreg_evaluation.jsonl`.
  - **Rule:** after any change to the parser or metric,rerun the hand check script. It must still give 44 / 2 / 24.

### What I found
**1. All 6 parse failures are truncation.**
Failed ids: 2184, 2385, 2594, 2595, 2616, 2617. Four of them come in consecutive pairs, because they are crowded sentences The model hits the 128-token cap and gets cut off. 
- These 6 sentences alone give **98 FN = 21.9% of all FN**.
- If they had not been truncated (upper bound, assuming all 98 were correct): F1 would be about 94.3, and the P−R gap would drop from 2.0 to about 0.45 points.
- So most of the "recall < precision" gap comes from truncation on crowded sentences, not from the model being bad at finding entities.

**2. Comparing with the paper.**
The paper reports 91.87 F1 for AutoReg-Struct, but that is a 7B model on the **test** set. My 93.45 is on **dev**, which is usually easier on CoNLL. I expect my test score to be lower. This is not a like-for-like comparison yet.

### Deviations from the paper (so far)
- 1.1B model with LoRA, not a 7B full fine-tune
- `max_new_tokens=128` (paper allows 512)
