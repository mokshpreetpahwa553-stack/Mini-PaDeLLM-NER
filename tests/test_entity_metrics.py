from src.evaluation.entity_metrics import helper


apple_tokens=["Apple", "hired", "John", "Smith", "in", "Dublin", "."]
apple_bio_tags =["B-ORG", "O", "B-PER", "I-PER", "O", "B-LOC", "O"]

elon_tokens= ["Elon", "Musk", "founded", "SpaceX", "in", "2002", "."]
elon_bio_tags=["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O"]

conference_tokens=["The", "conference", "will", "be", "held", "in", "Paris", "on", "Monday", "."]
conference_bio_tags= ["O", "O", "O", "O", "O", "O", "B-LOC", "O", "B-DATE", "O"]


def test_extra_false_positive():
    metrics = helper(tokens=apple_tokens,bio_tags=apple_bio_tags,predicted_mentions=[("ORG", "Apple"),("PER", "John Smith"),("LOC", "Dublin"),("MISC", "German")])

    assert metrics["true_positives"]== 3
    assert metrics["false_positives"]== 1
    assert metrics["false_negatives"]== 0
    assert metrics["precision"]== 0.75
    assert metrics["recall"]== 1.0
    assert metrics["f1"]== 0.8571428571428571


def test_perfect_predictions():
    metrics = helper(tokens=apple_tokens,bio_tags=apple_bio_tags,predicted_mentions=[("ORG", "Apple"),("PER", "John Smith"),("LOC", "Dublin")])

    assert metrics["true_positives"]== 3
    assert metrics["false_positives"]== 0
    assert metrics["false_negatives"]== 0
    assert metrics["precision"]== 1.0
    assert metrics["recall"]== 1.0
    assert metrics["f1"]== 1.0


def test_missing_prediction():
    metrics = helper(tokens=apple_tokens,bio_tags=apple_bio_tags,predicted_mentions=[("ORG", "Apple"),("PER", "John Smith")])

    assert metrics["true_positives"]== 2
    assert metrics["false_positives"]== 0
    assert metrics["false_negatives"]== 1
    assert metrics["precision"]== 1.0
    assert metrics["recall"]==2/3
    assert metrics["f1"]==0.8


def test_wrong_label_counts_as_false_positive_and_false_negative():
    metrics = helper(tokens=apple_tokens,bio_tags=apple_bio_tags,predicted_mentions=[("ORG", "Apple"),("ORG", "John Smith"),("LOC", "Dublin")])

    assert metrics["true_positives"]== 2
    assert metrics["false_positives"]== 1
    assert metrics["false_negatives"]== 1
    assert metrics["precision"]==2/3
    assert metrics["recall"]==2/3
    assert metrics["f1"]==2/3

def test_empty_predictions():
    metrics = helper(tokens=apple_tokens, bio_tags=apple_bio_tags, predicted_mentions=[])

    assert metrics["true_positives"]== 0
    assert metrics["false_positives"]== 0
    assert metrics["false_negatives"]== 3
    assert metrics["precision"]== 0.0
    assert metrics["recall"]== 0.0
    assert metrics["f1"]== 0.0


def test_wrong_entity_label_for_spacex():
    metrics = helper(tokens=elon_tokens, bio_tags=elon_bio_tags, predicted_mentions=[("PER", "Elon Musk"), ("PER", "SpaceX")])

    assert metrics["true_positives"]== 1
    assert metrics["false_positives"]== 1
    assert metrics["false_negatives"]== 1
    assert metrics["precision"]== 0.5
    assert metrics["recall"]== 0.5
    assert metrics["f1"]== 0.5


def test_correct_multi_token_person_and_organization():
    metrics = helper(tokens=elon_tokens, bio_tags=elon_bio_tags, predicted_mentions=[("PER", "Elon Musk"), ("ORG", "SpaceX")])

    assert metrics["true_positives"]== 2
    assert metrics["false_positives"]== 0
    assert metrics["false_negatives"]== 0
    assert metrics["precision"]== 1.0
    assert metrics["recall"]== 1.0
    assert metrics["f1"]== 1.0


def test_wrong_label_duplicate_and_case_sensitive_prediction():
    metrics = helper(tokens=conference_tokens, bio_tags=conference_bio_tags, predicted_mentions=[("ORG", "Conference"), ("LOC", "Paris"), ("ORG", "Paris"), ("DATE", "MONDAY")])

    assert metrics["true_positives"]== 1
    assert metrics["false_positives"]== 3
    assert metrics["false_negatives"]== 1
    assert metrics["precision"]== 0.25
    assert metrics["recall"]== 0.5
    assert metrics["f1"]== 1 / 3


def test_duplicate_prediction_is_counted_once():
    metrics = helper(tokens=conference_tokens, bio_tags=conference_bio_tags, predicted_mentions=[("ORG", "Conference"), ("LOC", "Paris"), ("ORG", "Paris"), ("DATE", "Monday")])

    assert metrics["true_positives"]== 2
    assert metrics["false_positives"]== 2
    assert metrics["false_negatives"]== 0
    assert metrics["precision"]== 0.5
    assert metrics["recall"]== 1.0
    assert metrics["f1"]==2/3


def test_correct_location_and_date():
    metrics = helper(tokens=conference_tokens, bio_tags=conference_bio_tags, predicted_mentions=[("LOC", "Paris"), ("DATE", "Monday")])

    assert metrics["true_positives"]== 2
    assert metrics["false_positives"]== 0
    assert metrics["false_negatives"]== 0
    assert metrics["precision"]== 1.0
    assert metrics["recall"]== 1.0
    assert metrics["f1"]== 1.0