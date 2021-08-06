//! Run as `cargo bench --bench matmul --features bench`.

use std::{fs::File, io::BufReader, path::Path};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};
use tract_onnx::prelude::{tvec, Datum, Framework, InferenceFact, InferenceModelExt};

use data::bench::matmul::data;

fn bench_tract(manager: &mut Criterion, name: &str, model: impl AsRef<Path>) {
    let mut model = BufReader::new(File::open(data().unwrap().join(model)).unwrap());
    let plan = tract_onnx::onnx()
        .model_for_read(&mut model)
        .unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), &[10, 128]))
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();
    manager.bench_function(name, |bencher| {
        bencher.iter(|| {
            plan.run(black_box(tvec![Array2::<f32>::zeros((10, 128)).into()]))
                .unwrap();
        })
    });
}

fn bench_onnx(manager: &mut Criterion, name: &str, model: impl AsRef<Path>) {
    let environment = Environment::builder()
        .with_log_level(LoggingLevel::Verbose)
        .build()
        .unwrap();
    let mut session = environment
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::DisableAll)
        .unwrap()
        .with_model_from_file(data().unwrap().join(model))
        .unwrap();
    manager.bench_function(name, |bencher| {
        bencher.iter(|| {
            session
                .run::<f32, f32, _>(black_box(vec![Array2::zeros((10, 128))]))
                .unwrap();
        })
    });
}

fn bench_tract_matmul_nonquant(manager: &mut Criterion) {
    bench_tract(manager, "Tract Matmul", "matmul.onnx");
}

fn bench_tract_matmul_dynquant(manager: &mut Criterion) {
    bench_tract(manager, "Tract Matmul Quantized", "matmul-dynquant.onnx");
}

fn bench_tract_dynquant_overhead(manager: &mut Criterion) {
    bench_tract(
        manager,
        "Tract Quantization Overhead",
        "dynquant-overhead.onnx",
    );
}

fn bench_onnx_matmul_nonquant(manager: &mut Criterion) {
    bench_onnx(manager, "Onnx Matmul", "matmul.onnx");
}

fn bench_onnx_matmul_dynquant(manager: &mut Criterion) {
    bench_onnx(manager, "Onnx Matmul Quantized", "matmul-dynquant.onnx");
}

fn bench_onnx_dynquant_overhead(manager: &mut Criterion) {
    bench_onnx(
        manager,
        "Onnx Quantization Overhead",
        "dynquant-overhead.onnx",
    );
}

criterion_group! {
    name = bench_tract_matmul;
    config = Criterion::default();
    targets =
        bench_tract_matmul_nonquant,
        bench_tract_matmul_dynquant,
        bench_tract_dynquant_overhead,
}

criterion_group! {
    name = bench_onnx_matmul;
    config = Criterion::default();
    targets =
        bench_onnx_matmul_nonquant,
        bench_onnx_matmul_dynquant,
        bench_onnx_dynquant_overhead,
}

criterion_main! {
    bench_tract_matmul,
    bench_onnx_matmul,
}
