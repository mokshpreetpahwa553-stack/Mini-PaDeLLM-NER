"""
Checking FN for 6 parse faield sentences from autoreg infernece
"""
from src.evaluation.evaluation_autoreg import load_jsonl,evaluation

records=load_jsonl("results/inference_autoreg_struct.jsonl")

subset = []
for record in records:
    if record["sentence_id"] in [2184, 2385, 2594, 2595, 2616, 2617]:
        subset.append(record)

_,metrics=evaluation(subset)
print(metrics["false_negatives"])

