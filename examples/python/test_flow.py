def test_negative() -> None:
    assert classify(-1) == "negative"

def test_zero() -> None:
    assert classify(0) == "zero"

def test_positive() -> None:
    assert classify(1) == "positive"

def test_while() -> None:
    assert sum_to(4) == 10

def test_list_loop() -> None:
    assert sum_values([1, 2, 3, 4]) == 10

def test_boolean() -> None:
    assert both(True, False) == False

def test_literal_fidelity() -> None:
    assert echo(["->", ")", "}", "é"]) == ["->", ")", "}", "é"]
