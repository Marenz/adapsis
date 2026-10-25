def test_add_two() -> None:
    assert add_two(3) == 5

def test_local_calls() -> None:
    assert twice_added(3) == 7
