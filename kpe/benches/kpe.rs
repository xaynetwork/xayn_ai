//! Run as `cargo bench --bench kpe`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kpe::{Configuration, Pipeline};
use test_utils::kpe::{bert, classifier, cnn, vocab};

fn bench_kpe(manager: &mut Criterion) {
    let config = Configuration::from_files(
        vocab().unwrap(),
        bert().unwrap(),
        cnn().unwrap(),
        classifier().unwrap(),
    )
    .unwrap()
    .with_token_size(128)
    .unwrap();

    let pipeline = Pipeline::from(config).unwrap();

    let sequence = "This sequence will be split into key phrases.";
    manager.bench_function("KPE", |bencher| {
        bencher.iter(|| pipeline.run(black_box(sequence)).unwrap())
    });
}

criterion_group! {
    name = bench;
    config = Criterion::default();
    targets =
        bench_kpe,
}

criterion_main! {
    bench,
}
