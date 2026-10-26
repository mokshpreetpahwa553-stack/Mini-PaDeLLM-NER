from src.evaluation.entity_metrics import evaluate_dataset

def test_dataset_micro_average():
    examples = [{"tokens": ["Apple", "hired", "John", "Smith", "in", "Dublin", "."],
            "bio_tags": ["B-ORG", "O", "B-PER", "I-PER", "O", "B-LOC", "O"],
        },
        {
            "tokens": ["Elon", "Musk", "founded", "SpaceX", "in", "2002", "."],
            "bio_tags": ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O"],
        },
    ]

    predicted_mentions_per_example = [[("ORG", "Apple"),("PER", "John Smith"),("LOC", "Dublin"),("MISC", "German")],
        [("PER", "Elon Musk"),("ORG", "SpaceX")]]

    metrics = evaluate_dataset(all_examples=examples,all_predicted_mentions=predicted_mentions_per_example)

    assert metrics["true_positives"]==5
    assert metrics["false_positives"]== 1
    assert metrics["false_negatives"]== 0
    assert metrics["precision"] == 5/6
    assert metrics["recall"] == 1.0