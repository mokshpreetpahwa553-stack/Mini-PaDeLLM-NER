from src.data.onestep_data_builder import build_onestep_examples

def test_onestep_data_bulder():
    tokens = ["Apple", "hired", "John", "Smith", "in", "Dublin", "."]
    bio_tags = ["B-ORG", "O", "B-PER", "I-PER", "O", "B-LOC", "O"]

    examples = build_onestep_examples(tokens, bio_tags, sentence_id=0)

    assert len(examples)==4

    assert examples[0]["task"]=="Find all ORG mentions."
    assert examples[0]["output"]=="Apple"

    assert examples[1]["task"]=="Find all PER mentions."
    assert examples[1]["output"]=="John Smith"

    assert examples[2]["task"]=="Find all MISC mentions."
    assert examples[2]["output"]=="<none>"

    assert examples[3]["task"]=="Find all LOC mentions."
    assert examples[3]["output"]=="Dublin"

def test_onestep_no_entities():
    tokens = ["The", "weather", "was", "nice", "."]
    bio_tags = ["O", "O", "O", "O", "O"]

    examples = build_onestep_examples(tokens, bio_tags, sentence_id=2)

    assert len(examples)==4

    assert examples[0]["input"]=="The weather was nice ."
    assert examples[0]["sentence_id"]==2
    assert examples[0]["format"]=="onestep"

    assert examples[0]["output"]=="<none>"
    assert examples[1]["output"]=="<none>"
    assert examples[2]["output"]=="<none>"
    assert examples[3]["output"]=="<none>"



def test_onestep_multiple_mentions_same_label():
    tokens = ["Apple", "hired", "John", "Smith", "and", "Mary", "Byrne", "."]
    bio_tags = ["B-ORG", "O","B-PER","I-PER", "O","B-PER","I-PER","O"]

    examples = build_onestep_examples(tokens, bio_tags, sentence_id=3)

    assert len(examples)==4

    assert examples[0]["task"]=="Find all ORG mentions."
    assert examples[0]["output"]=="Apple"

    assert examples[1]["task"]=="Find all PER mentions."
    assert examples[1]["output"]=="John Smith <sep> Mary Byrne"

    assert examples[2]["task"]=="Find all MISC mentions."
    assert examples[2]["output"]=="<none>"

    assert examples[3]["task"]=="Find all LOC mentions."
    assert examples[3]["output"]=="<none>"