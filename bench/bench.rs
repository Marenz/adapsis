// Native Rust benchmark: same algorithms, compiled to native code.
// Compile: rustc -O bench/bench.rs -o bench/bench_native && ./bench/bench_native

use std::time::Instant;

fn fib(n: i64) -> i64 {
    if n <= 1 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn sum_to(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i <= n {
        total += i;
        i += 1;
    }
    total
}

fn collatz_steps(n: i64) -> i64 {
    let mut steps: i64 = 0;
    let mut current = n;
    while current != 1 {
        if current % 2 == 0 {
            current /= 2;
        } else {
            current = current * 3 + 1;
        }
        steps += 1;
    }
    steps
}

fn nested_loops(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        let mut j: i64 = 0;
        while j < n {
            total += i * j;
            j += 1;
        }
        i += 1;
    }
    total
}

fn main() {
    let benchmarks: Vec<(&str, Box<dyn Fn() -> i64>, i64)> = vec![
        ("fib(25)", Box::new(|| fib(25)), 75025),
        ("sum_to(100000)", Box::new(|| sum_to(100000)), 5000050000),
        (
            "collatz_steps(837799)",
            Box::new(|| collatz_steps(837799)),
            524,
        ),
        (
            "nested_loops(200)",
            Box::new(|| nested_loops(100)),
            24502500,
        ),
    ];

    println!("\n{:<30} {:>12} {:>15}", "Function", "Time", "Result");
    println!("{}", "-".repeat(60));

    for (name, f, expected) in &benchmarks {
        // warmup
        f();
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        assert_eq!(
            result, *expected,
            "{name}: got {result}, expected {expected}"
        );
        println!(
            "{:<30} {:>10.2}ms {:>15}",
            name,
            elapsed.as_secs_f64() * 1000.0,
            result,
        );
    }
}
