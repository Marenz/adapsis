"""Typed arithmetic over the Adapsis runtime."""

def add_two(x: int) -> int:
    """Add two using explicitly typed intermediate bindings."""
    first: int = x + 1
    second: int = first + 1
    return second

def twice_added(x: int) -> int:
    return add_two(add_two(x))
