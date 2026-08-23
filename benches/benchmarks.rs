use criterion::{Criterion, criterion_group, criterion_main};
use laplace::stats::mean;

fn eda(c: &mut Criterion) {
    let arr: Vec<i32> = vec![1; 100_000];
    c.bench_function("mean", |b| b.iter(|| mean(&arr)));
}

criterion_group!(benches, eda);
criterion_main!(benches);
