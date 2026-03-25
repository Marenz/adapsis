#!/usr/bin/env python3
"""Benchmark: same algorithms in Python for comparison with Adapsis VM/tree-walker."""

import time


def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


def sum_to(n):
    total = 0
    i = 0
    while i <= n:
        total += i
        i += 1
    return total


def collatz_steps(n):
    steps = 0
    current = n
    while current != 1:
        if current % 2 == 0:
            current //= 2
        else:
            current = current * 3 + 1
        steps += 1
    return steps


def nested_loops(n):
    total = 0
    i = 0
    while i < n:
        j = 0
        while j < n:
            total += i * j
            j += 1
        i += 1
    return total


benchmarks = [
    ("fib(25)", lambda: fib(25), 75025),
    ("sum_to(100000)", lambda: sum_to(100000), 5000050000),
    ("collatz_steps(837799)", lambda: collatz_steps(837799), 524),
    ("nested_loops(100)", lambda: nested_loops(100), 24502500),
]

print(f"\n{'Function':<30} {'Time':>12} {'Result':>15}")
print("-" * 60)

for name, fn, expected in benchmarks:
    # warmup
    fn()
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    assert result == expected, f"{name}: got {result}, expected {expected}"
    print(f"{name:<30} {elapsed * 1000:>10.2f}ms {result:>15}")
