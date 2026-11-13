from src.evaluation.entity_metrics import compute_micro_entity_metrics
import argparse
import json
from pathlib import Path
from src.data.bio_to_mentions_converter import bio_to_mention_conv
from src.data.output_parser_autoreg_struct import autoreg_parser

def load_jsonl(prediction_saved_file):
    records=[]
    with open(prediction_saved_file,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

        return records

def save_metrics(metrics,output_file):
    output_path=Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w",encoding="utf-8") as f:
        json.dump(metrics,f,indent=2,ensure_ascii=False)


def save_jsonl(records, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def autoreg_parser_to_predictions(entities):
    predictions = []

    for label, mentions in entities.items():
        for mention in mentions:
            predictions.append((label, mention))

    return predictions

def evaluation(predictions):
    total_true_positives,total_false_positives,total_false_negatives=0,0,0
    parse_failures,count_correct=0,0
    total_sentences=len(predictions)
    all_records=[]
    failed_parse=[] #it will keep record of predictions that faield parsing
    for prediction in predictions:
        
        record={}
        pred=prediction["generation"]
        gold_mentions=bio_to_mention_conv(prediction["tokens"],prediction["bio_tags"])

        parse=autoreg_parser(pred)
        status,entities =parse["status"],parse["entities"]

        parse_success = True if status=="valid" else False
        if not parse_success:
            failed_parse.append(prediction["sentence_id"])
            parse_failures+=1
        predicted_entities=autoreg_parser_to_predictions(entities)
        metrics = compute_micro_entity_metrics(gold_mentions,predicted_entities)

        total_true_positives += metrics["true_positives"]
        total_false_positives += metrics["false_positives"]
        total_false_negatives += metrics["false_negatives"]
        
        if len(gold_mentions)==len(predicted_entities):
            count_correct+=1

        record={
            "sentence_id": prediction["sentence_id"],
            "tokens": prediction["tokens"],
            "bio_tags": prediction["bio_tags"],
            "generation": pred,
            "gold_mentions": gold_mentions,
            "predicted_mentions": entities,
            "parse_success": parse_success,
            "sentence_metrics": metrics
        }

        all_records.append(record)

    precision = total_true_positives / (total_true_positives+total_false_positives) if (total_true_positives+total_false_positives)>0 else 0.0
    recall = total_true_positives / (total_true_positives+total_false_negatives) if (total_true_positives+total_false_negatives) else 0.0
    f1 = (2*precision*recall) / (precision+recall) if (precision+recall) else 0.0

    metrics= {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives" : total_true_positives,
        "false_positives": total_false_positives,
        "false_negatives": total_false_negatives,
        "number_sentences":total_sentences,
        "parse_failure_rate":parse_failures/total_sentences if total_sentences>0 else 0.0,
        "count_accuracy": count_correct/total_sentences if total_sentences>0 else 0.0
        }

    print(f"---------------\n{failed_parse}\n---------------- \n")

    return all_records,metrics

    
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        default="results/inference_autoreg_struct.jsonl"
    )

    parser.add_argument(
        "--parsed_output_file",
        type=str,
        default="results/parsed_inferenced_autoreg_struct.jsonl"
    )

    parser.add_argument(
        "--metrics_output_file",
        type=str,
        default="results/metrics_inferenced_autoreg_struct.json"
    )

    args = parser.parse_args()

    input_records = load_jsonl(args.input_file)

    parsed_records,metrics=evaluation(input_records)

    save_jsonl(
        records=parsed_records,
        output_file=args.parsed_output_file
    )

    save_metrics(
        metrics=metrics,
        output_file=args.metrics_output_file
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()