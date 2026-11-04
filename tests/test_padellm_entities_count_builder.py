from src.data.padellm_entities_count_builder import padellm_counter

def test_padellm_counter():
    tokens=["Tim", "Cook", "works", "at", "Apple", "in", "Cork", "."]
    bio_tags=["B-PER", "I-PER", "O", "O","B-ORG", "O", "B-LOC", "O"]

    examples=padellm_counter(tokens, bio_tags, sentence_id=0)

    assert len(examples)==4

    expected_counts={
        "PER": "1",
        "ORG": "1",
        "MISC": "0",
        "LOC": "1",
    }

    for example in examples:
        label=example["label"]
        assert example["output"]==expected_counts[label]
        assert example["sentence_id"]==0
        assert example["input"]=="Tim Cook works at Apple in Cork ."
        assert example["task"]==f"Count the number of {label} mentions."


def test_multiple_mentions():
    tokens = ["John","met","Mary", "in", "Paris", "."]
    bio_tags=["B-PER","O", "B-PER", "O", "B-LOC","O"]

    examples = padellm_counter(tokens, bio_tags, sentence_id=1)

    counts = {example["label"]: example["output"] for example in examples}

    assert counts["PER"] =="2"
    assert counts["LOC"] =="1"
    assert counts["ORG"]=="0"
    assert counts["MISC"] =="0"

def test_zero_mentions():
    tokens=[]
    bio_tags=[]
    examples=padellm_counter(tokens, bio_tags, sentence_id=0)

    counts = {example["label"]: example["output"] for example in examples}
    assert counts["PER"] =="0"
    assert counts["LOC"] =="0"
    assert counts["ORG"]=="0"
    assert counts["MISC"] =="0"
