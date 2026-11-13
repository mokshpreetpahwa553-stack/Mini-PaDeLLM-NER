"""
Hand checking test for the autoreg_struct evaluation pipeline.

"""
from src.evaluation.evaluation_autoreg import load_jsonl,evaluation,save_jsonl

records=load_jsonl("results/inference_autoreg_struct.jsonl")

subset = []
for record in records:
    if 1541<=record["sentence_id"]<=1562 or 2184<=record["sentence_id"]<=2185:
        subset.append(record)

_,metrics=evaluation(subset)
print(metrics)

#we save subset because it containg prediction adn generation, of our current model, and if in future we upgrade our model we would need it to compare
# and also it checks parser and metric independent of future model runs
save_jsonl(subset,"tests/fixtures/test_hand_check_autoreg_evaluation.jsonl")