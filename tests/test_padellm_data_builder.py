from src.data.padellm_data_builder import build_padellm_examples

def test_padellm_data_builder_apple_sentence():
    tokens = ["Apple", "hired", "John", "Smith", "in", "Dublin", "."]
    bio_tags = ["B-ORG", "O", "B-PER", "I-PER", "O", "B-LOC", "O"]

    examples = build_padellm_examples(tokens, bio_tags, sentence_id=0)

    assert len(examples)==4

    assert examples[0]["task"]=="Find the 1st ORG mention."
    assert examples[0]["output"]=="Apple"

    assert examples[1]["task"]=="Find the 1st PER mention."
    assert examples[1]["output"]=="John Smith"

    assert examples[2]["task"]=="Find the 1st LOC mention."
    assert examples[2]["output"]=="Dublin"

    assert examples[3]["task"]=="Find the 1st MISC mention."
    assert examples[3]["output"]=="<eos>"


def test_padellm_data_builder_with_two_per_mentions():
    tokens = ["Alice", "met", "Bob", "in", "Paris", "."]
    bio_tags = ["B-PER", "O", "B-PER", "O", "B-LOC", "O"]

    examples = build_padellm_examples(tokens, bio_tags, sentence_id=1)

    assert len(examples)==5

    assert examples[0]["task"]=="Find the 1st PER mention."
    assert examples[0]["output"]=="Alice"

    assert examples[1]["task"]=="Find the 2nd PER mention."
    assert examples[1]["output"]=="Bob"

    assert examples[2]["task"]=="Find the 1st LOC mention."
    assert examples[2]["output"]=="Paris"

    assert examples[3]["task"]=="Find the 1st ORG mention."
    assert examples[3]["output"]=="<eos>"

    assert examples[4]["task"]=="Find the 1st MISC mention."
    assert examples[4]["output"]=="<eos>"