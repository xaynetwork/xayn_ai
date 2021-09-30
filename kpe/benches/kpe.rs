//! Run as `cargo bench --bench mbert.

use std::{io::Result, path::Path};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{s, Array1, Axis};

use kpe::Builder;
use test_utils::smbert;

fn bench_kpe(manager: &mut Criterion) {
    // TODO: change assets once available
    let pipeline = Builder::from_files(
        smbert::vocab().unwrap(),
        smbert::model().unwrap(),
        smbert::model().unwrap(),
        smbert::model().unwrap(),
    )
    .unwrap()
    .with_accents(false)
    .with_lowercase(true)
    .with_token_size(512)
    .unwrap()
    .with_key_phrase_size(5)
    .unwrap()
    .build()
    .unwrap();
    let sequence = "This embedding fits perfectly and this embedding fits well.";
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
