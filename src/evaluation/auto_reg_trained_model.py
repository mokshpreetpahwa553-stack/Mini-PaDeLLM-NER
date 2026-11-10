"""
Evaluate autoreg_struct predictions.

Reads a JSONL file produced by inference, parses each model generation,
and reports:
- total examples
- valid structured outputs
- empty outputs
- malformed outputs
- entity-level precision, recall, and F1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from src.data.output_parser_autoreg_struct import parse_prediction


def load_jsonl(path):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    return examples


def extract_entities(label_entities):
    entitiy_text = []
    for label,entities in label_entities.items():
        for entity in entities:
            en = entity.strip()
            if en:
                entitiy_text.append((label,en))

    return entitiy_text

def get_type_counts(per_type, entity_type):
    if entity_type not in per_type:
        per_type[entity_type]={
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    return per_type[entity_type]

def evaluate_predictions(predictions):
    total = len(predictions)
    valid,empty,malformed=0,0,0
    true_positives,false_positives,false_negatives=0,0, 0
   

    per_type = {}

    for prediction in predictions:
        entities=extract_entities(prediction.get("entities", {}))

        status, parsed_entities=parse_prediction(prediction.get("generation", ""))

        if status == "valid":
            valid += 1
            predicted_entities =extract_entities(parsed_entities)
        elif status == "empty":
            empty += 1
            predicted_entities =[]
        else:
            malformed += 1
            predicted_entities=[]

        for entity in predicted_entities:
            label=entity[0]
            counts=get_type_counts(per_type, label)

            if entity in entities:
                true_positives += 1
                per_type[entity[0]]["tp"] += 1
            else:
                false_positives += 1
                per_type[entity[0]]["fp"] += 1

        for entity in entities:
            if entity not in predicted_entities:
                false_negatives += 1
                per_type[entity[0]]["fn"] += 1

    precision = (true_positives/ (true_positives+ false_positives) if true_positives+false_positives>0 else 0.0)

    recall = (true_positives /(true_positives+false_negatives) if true_positives+ false_negatives>0 else 0.0 )

    f1 = (2*precision *recall /(precision +recall) if precision +recall > 0 else 0.0)

    return {
        "total": total,
        "valid": valid,
        "empty": empty,
        "malformed": malformed,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "per_type": dict(per_type),
    }


def print_report(metrics):
    print("\n===AUTOREG-STRUCT evaluation ===")
    print(f"Total examples:{metrics['total']}")
    print(f"Valid outputs: {metrics['valid']}")
    print(f"Empty outputs:{metrics['empty']}")
    print(f"Malformed outputs: {metrics['malformed']}")
    print("\n=== Entity-level results ===")
    print(f"Precision:{metrics['precision']:.4f}")
    print(f"Recall:{metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")

    print("\n=== Per entity type ===")

    for entity_type, counts in metrics["per_type"].items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        type_precision = tp/(tp+fp) if tp +fp > 0 else 0.0
        type_recall = tp/ (tp +fn) if tp +fn> 0 else 0.0
        type_f1 = (2*type_precision*type_recall/ (type_precision+type_recall) if type_precision + type_recall > 0 else 0.0)

        print(
            f"{entity_type}: "
            f"P={type_precision:.4f}, "
            f"R={type_recall:.4f}, "
            f"F1={type_f1:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Path to the JSONL file produced by inference",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save the evaluation report as JSON",
    )

    args=parser.parse_args()
    predictions=load_jsonl(args.predictions_file)

    if not predictions:
        raise ValueError(f"No predictions found in {args.predictions_file}")

    metrics=evaluate_predictions(predictions)

    print_report(metrics)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nEvaluation report saved to: {output_path}")


if __name__ == "__main__":
    main()