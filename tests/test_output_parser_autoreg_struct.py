from src.inference.output_parser_autoreg_struct import autoreg_parser


def test_valid_output_with_multiple_mentions():
    output = ("ORG: Apple <sep> PER: John Smith <sep> Mary Byrne <sep> MISC: <none> <sep> LOC: Dublin")

    result = autoreg_parser(output)

    assert result["status"]=="valid"
    assert result["entities"]=={
        "ORG": ["Apple"],
        "PER": ["John Smith", "Mary Byrne"],
        "MISC": [],
        "LOC": ["Dublin"],
    }


def test_valid_output_with_all_none():
    output = ("ORG: <none> <sep> PER: <none> <sep> MISC: <none> <sep> LOC: <none>")

    result = autoreg_parser(output)

    assert result["status"]=="valid"
    assert result["entities"]=={
        "ORG": [],
        "PER": [],
        "MISC": [],
        "LOC": [],
    }


def test_empty_output():
    result = autoreg_parser("")

    assert result["status"]=="empty"
    assert result["entities"]==[]

    result = autoreg_parser("   ")
    
    assert result["status"]=="empty"
    assert result["entities"]==[]
    

def test_missing_label_is_malformed():
    output = ("ORG: Apple <sep> PER: John Smith <sep> MISC: <none>" )

    result = autoreg_parser(output)

    assert result["status"]=="malformed"
    assert result["reason"]=="missing_labels"


def test_wrong_label_order_is_malformed():
    output = ("PER: John Smith <sep> ORG: Apple <sep> MISC: <none> <sep> LOC: Dublin")

    result = autoreg_parser(output)

    assert result["status"]=="malformed"
    assert result["reason"]=="missing_labels"


def test_missing_colon_is_malformed():
    output = ("ORG Apple <sep> PER: John Smith ""<sep> MISC: <none> <sep> LOC: Dublin")

    result = autoreg_parser(output)

    assert result["status"]=="malformed"


def test_empty_entity_value_is_malformed():
    output = ("ORG: <sep> PER: John Smith ""<sep> MISC: <none> <sep> LOC: Dublin")

    result = autoreg_parser(output)

    assert result["status"]=="malformed"
    assert result["reason"]=="empty_entity_value"