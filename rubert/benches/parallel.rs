//! Run as `cargo bench --bench parallel --features bench`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    join,
};

use data::{qambert, smbert};
use rubert::{
    kinds::{QAMBert, SMBert},
    AveragePooler,
    Builder,
};

const BATCH_SIZE: usize = 16;
const TOKEN_SIZE: usize = 64;

/// Creates a number of dummy sequences as input for the MBert pipelines.
fn sequences(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("This is sequence number {}.", i))
        .collect()
}

/// Builds a MBert pipeline of the given kind with the corresponding vocabulary and model files.
macro_rules! mbert {
    ($kind:ty, $vocab:expr, $model:expr $(,)?) => {
        Builder::<_, _, $kind, _>::from_files($vocab.unwrap(), $model.unwrap())
            .unwrap()
            .with_accents(false)
            .with_lowercase(true)
            .with_token_size(TOKEN_SIZE)
            .unwrap()
            .with_pooling(AveragePooler)
            .build()
            .unwrap()
    };
}

/// Benches the SMBert pipeline sequentially.
fn bench_smbert_sequential(manager: &mut Criterion) {
    let smbert = mbert!(SMBert, smbert::vocab(), smbert::model());
    let sequences = sequences(BATCH_SIZE);

    manager.bench_function("SMBert Sequential", |bencher| {
        bencher.iter(|| {
            let _ = sequences
                .iter() // runs each sequence sequentially
                .map(|sequence| smbert.run(black_box(sequence)).unwrap())
                .collect::<Vec<_>>();
        });
    });
}

/// Benches the SMBert pipeline parallelly.
fn bench_smbert_parallel(manager: &mut Criterion) {
    let smbert = mbert!(SMBert, smbert::vocab(), smbert::model());
    let sequences = sequences(BATCH_SIZE);

    manager.bench_function("SMBert Parallel", |bencher| {
        bencher.iter(|| {
            let _ = sequences
                .par_iter() // runs each sequence parallelly
                .map(|sequence| smbert.run(black_box(sequence)).unwrap())
                .collect::<Vec<_>>();
        })
    });
}

/// Benches the QAMBert pipeline sequentially.
fn bench_qambert_sequential(manager: &mut Criterion) {
    let qambert = mbert!(QAMBert, qambert::vocab(), qambert::model());
    let sequences = sequences(BATCH_SIZE);

    manager.bench_function("QAMBert Sequential", |bencher| {
        bencher.iter(|| {
            let _ = sequences
                .iter() // runs each sequence sequentially
                .map(|sequence| qambert.run(black_box(sequence)).unwrap())
                .collect::<Vec<_>>();
        });
    });
}

/// Benches the QAMBert pipeline parallelly.
fn bench_qambert_parallel(manager: &mut Criterion) {
    let qambert = mbert!(QAMBert, qambert::vocab(), qambert::model());
    let sequences = sequences(BATCH_SIZE);

    manager.bench_function("QAMBert Parallel", |bencher| {
        bencher.iter(|| {
            let _ = sequences
                .par_iter() // runs each sequence parallelly
                .map(|sequence| qambert.run(black_box(sequence)).unwrap())
                .collect::<Vec<_>>();
        })
    });
}

/// Benches both MBert pipelines fully sequentially.
fn bench_mberts_sequential(manager: &mut Criterion) {
    let smbert = mbert!(SMBert, smbert::vocab(), smbert::model());
    let qambert = mbert!(QAMBert, qambert::vocab(), qambert::model());
    let sequences = sequences(BATCH_SIZE);

    manager.bench_function("SMBert & QAMBert Sequential", |bencher| {
        bencher.iter(|| {
            // runs the SMBert & QAMBert pipelines sequentially
            let _ = sequences
                .iter() // runs each sequence sequentially within the SMBert pipeline
                .map(|sequence| smbert.run(black_box(sequence)).unwrap())
                .collect::<Vec<_>>();
            let _ = sequences
                .iter() // runs each sequence sequentially within the QAMBert pipeline
                .map(|sequence| qambert.run(black_box(sequence)).unwrap())
                .collect::<Vec<_>>();
        });
    });
}

/// Benches both MBert pipelines parallelly, while within each pipeline it runs sequentially.
fn bench_mberts_join(manager: &mut Criterion) {
    let smbert = mbert!(SMBert, smbert::vocab(), smbert::model());
    let qambert = mbert!(QAMBert, qambert::vocab(), qambert::model());
    let sequences = sequences(BATCH_SIZE);

    manager.bench_function("SMBert & QAMBert Join", |bencher| {
        bencher.iter(|| {
            // runs the SMBert & QAMBert pipelines parallelly
            let _ = join(
                || {
                    sequences
                        .iter() // runs each sequence sequentially within the SMBert pipeline
                        .map(|sequence| smbert.run(black_box(sequence)).unwrap())
                        .collect::<Vec<_>>()
                },
                || {
                    sequences
                        .iter() // runs each sequence sequentially within the QAMBert pipeline
                        .map(|sequence| qambert.run(black_box(sequence)).unwrap())
                        .collect::<Vec<_>>()
                },
            );
        });
    });
}

/// Benches both MBert pipelines sequentially, while within each pipeline it runs parallely.
fn bench_mberts_parallel(manager: &mut Criterion) {
    let smbert = mbert!(SMBert, smbert::vocab(), smbert::model());
    let qambert = mbert!(QAMBert, qambert::vocab(), qambert::model());
    let sequences = sequences(BATCH_SIZE);

    manager.bench_function("SMBert & QAMBert Parallel", |bencher| {
        bencher.iter(|| {
            // runs the SMBert & QAMBert pipelines sequentially
            let _ = sequences
                .par_iter() // runs each sequence parallelly within the SMBert pipeline
                .map(|sequence| smbert.run(black_box(sequence)).unwrap())
                .collect::<Vec<_>>();
            let _ = sequences
                .par_iter() // runs each sequence parallelly within the QAMBert pipeline
                .map(|sequence| qambert.run(black_box(sequence)).unwrap())
                .collect::<Vec<_>>();
        });
    });
}

/// Benches both Mbert pipelines fully parallelly.
fn bench_mberts_join_parallel(manager: &mut Criterion) {
    let smbert = mbert!(SMBert, smbert::vocab(), smbert::model());
    let qambert = mbert!(QAMBert, qambert::vocab(), qambert::model());
    let sequences = sequences(BATCH_SIZE);

    manager.bench_function("SMBert & QAMBert Join Parallel", |bencher| {
        bencher.iter(|| {
            // runs the SMBert & QAMBert pipelines parallelly
            let _ = join(
                || {
                    sequences
                        .par_iter() // runs each sequence parallelly within the SMBert pipeline
                        .map(|sequence| smbert.run(black_box(sequence)).unwrap())
                        .collect::<Vec<_>>()
                },
                || {
                    sequences
                        .par_iter() // runs each sequence parallelly within the QAMBert pipeline
                        .map(|sequence| qambert.run(black_box(sequence)).unwrap())
                        .collect::<Vec<_>>()
                },
            );
        });
    });
}

criterion_group! {
    name = bench_smbert;
    config = Criterion::default();
    targets =
        bench_smbert_sequential,
        bench_smbert_parallel,
}

criterion_group! {
    name = bench_qambert;
    config = Criterion::default();
    targets =
        bench_qambert_sequential,
        bench_qambert_parallel,
}

criterion_group! {
    name = bench_mberts;
    config = Criterion::default();
    targets =
        bench_mberts_sequential,
        bench_mberts_join,
        bench_mberts_parallel,
        bench_mberts_join_parallel,
}

criterion_main! {
    bench_smbert,
    bench_qambert,
    bench_mberts,
}
