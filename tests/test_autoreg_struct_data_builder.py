from src.data.autoreg_struct_data_builder import build_autoreg_struct_examples

def test_autoreg_struct():
    tokens = ["Apple", "hired", "John", "Smith", "in", "Dublin", "."]
    bio_tags = ["B-ORG", "O", "B-PER", "I-PER", "O", "B-LOC", "O"]

    examples = build_autoreg_struct_examples(tokens, bio_tags, sentence_id=0)

    assert len(examples)==1

    assert examples[0]["output"]==("ORG: Apple <sep> PER: John Smith <sep> MISC: <none> <sep> LOC: Dublin")

def test_autoreg_struct_no_entities():
    tokens = ["The", "weather", "was", "nice", "."]
    bio_tags = ["O", "O", "O", "O", "O"]

    examples = build_autoreg_struct_examples(tokens, bio_tags, sentence_id=1)

    assert len(examples)==1

    assert examples[0]["input"]=="The weather was nice ."
    assert examples[0]["task"]=="Find all entity mentions in the sentence."
    assert examples[0]["format"]=="autoreg-struct"
    assert examples[0]["sentence_id"]==1

    assert examples[0]["output"]==("ORG: <none> <sep> PER: <none> <sep> MISC: <none> <sep> LOC: <none>")


def test_autoreg_struct_multiple_mentions_same_label():
    tokens =["Apple", "hired", "John", "Smith", "and", "Mary", "Byrne", "."]
    bio_tags =["B-ORG", "O","B-PER", "I-PER", "O","B-PER", "I-PER", "O"]

    examples = build_autoreg_struct_examples(tokens, bio_tags, sentence_id=2)

    assert len(examples)==1

    assert examples[0]["input"]=="Apple hired John Smith and Mary Byrne ."
    assert examples[0]["task"]=="Find all entity mentions in the sentence."
    assert examples[0]["format"]=="autoreg-struct"
    assert examples[0]["sentence_id"]==2

    assert examples[0]["output"]==("ORG: Apple <sep> PER: John Smith <sep> Mary Byrne <sep> ""MISC: <none> <sep> LOC: <none>")