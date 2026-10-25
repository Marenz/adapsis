"""Control flow, read-only lists, and Boolean values."""

def classify(n: int) -> str:
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    else:
        return "positive"

def sum_to(n: int) -> int:
    total: int = 0
    index: int = 0
    while index <= n:
        total = total + index
        index = index + 1
    return total

def sum_values(values: list[int]) -> int:
    total: int = 0
    for item in values:
        total = total + item
    return total

def both(left: bool, right: bool) -> bool:
    return left and right

def echo(values: list[str]) -> list[str]:
    return values
