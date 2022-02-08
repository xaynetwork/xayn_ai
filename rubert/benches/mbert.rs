//! Run as `cargo bench --bench mbert --features bench`.

use std::{io::Result, path::Path};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{s, Array1, Axis};
use onnxruntime::{environment::Environment, GraphOptimizationLevel};

use rubert::{
    kinds::{QAMBert, SMBert},
    AveragePooler,
    Config,
    Embedding2,
    FirstPooler,
    NonePooler,
    Pipeline,
};
use rubert_tokenizer::{Builder as TokenizerBuilder, Padding, Truncation};
use test_utils::{qambert, smbert};

const TOKEN_SIZE: usize = 64;
const SEQUENCE: &str = "This is a sequence.";

macro_rules! bench_tract {
    (
        $manager:expr,
        $group:expr => $kind:ty,
        $vocab:expr,
        $model:expr,
        [$($name:expr => $pooler:expr),+ $(,)?] $(,)?
    ) => {
        let mut group = $manager.benchmark_group(format!("{} {}", $group, TOKEN_SIZE));
        $(
            let config = Config::<$kind, _>::from_files($vocab.unwrap(), $model.unwrap())
                .unwrap()
                .with_accents(false)
                .with_lowercase(true)
                .with_token_size(TOKEN_SIZE)
                .unwrap()
                .with_pooling($pooler);
            let pipeline = Pipeline::from(config).unwrap();
            group.bench_function($name, |bencher| {
                bencher.iter(|| pipeline.run(black_box(SEQUENCE)).unwrap())
            });
        )+
    };
}

fn bench_onnx(
    manager: &mut Criterion,
    name: &str,
    vocab: Result<impl AsRef<Path>>,
    model: Result<impl AsRef<Path>>,
) {
    let tokenizer = TokenizerBuilder::from_file(vocab.unwrap())
        .unwrap()
        .with_normalizer(true, false, false, true)
        .with_model("[UNK]", "##", 100)
        .with_post_tokenizer("[CLS]", "[SEP]")
        .with_truncation(Truncation::fixed(TOKEN_SIZE, 0))
        .with_padding(Padding::fixed(TOKEN_SIZE, "[PAD]"))
        .build()
        .unwrap();
    let environment = Environment::builder().build().unwrap();
    let mut session = environment
        .new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::DisableAll)
        .unwrap()
        .with_model_from_file(model.unwrap())
        .unwrap();

    manager.bench_function(name, |bencher| {
        bencher.iter(|| {
            let encoding = tokenizer.encode(black_box(SEQUENCE));
            let (token_ids, type_ids, _, _, _, _, attention_mask, _) = encoding.into();
            let inputs = vec![
                Array1::<i64>::from(token_ids).insert_axis(Axis(0)),
                Array1::<i64>::from(attention_mask).insert_axis(Axis(0)),
                Array1::<i64>::from(type_ids).insert_axis(Axis(0)),
            ];
            let outputs = session.run(inputs).unwrap();

            Embedding2::from(outputs[0].slice(s![0, .., ..]).to_owned());
        })
    });
}

fn bench_tract_smbert_nonquant(manager: &mut Criterion) {
    bench_tract!(
        manager,
        "Tract SMBert" => SMBert,
        smbert::vocab(),
        smbert::model(),
        [
            "None Pooler" => NonePooler,
            "First Pooler" => FirstPooler,
            "Average Pooler" => AveragePooler,
        ],
    );
}

fn bench_tract_smbert_dynquant(manager: &mut Criterion) {
    bench_tract!(
        manager,
        "Tract SMBert Quantized" => SMBert,
        smbert::vocab(),
        smbert::model_quant(),
        [
            "None Pooler" => NonePooler,
            "First Pooler" => FirstPooler,
            "Average Pooler" => AveragePooler,
        ],
    );
}

fn bench_tract_qambert_nonquant(manager: &mut Criterion) {
    bench_tract!(
        manager,
        "Tract QAMBert" => QAMBert,
        qambert::vocab(),
        qambert::model(),
        [
            "None Pooler" => NonePooler,
            "First Pooler" => FirstPooler,
            "Average Pooler" => AveragePooler,
        ],
    );
}

fn bench_tract_qambert_dynquant(manager: &mut Criterion) {
    bench_tract!(
        manager,
        "Tract QAMBert Quantized" => QAMBert,
        qambert::vocab(),
        qambert::model_quant(),
        [
            "None Pooler" => NonePooler,
            "First Pooler" => FirstPooler,
            "Average Pooler" => AveragePooler,
        ],
    );
}

fn bench_onnx_smbert_nonquant(manager: &mut Criterion) {
    bench_onnx(manager, "Onnx SMBert", smbert::vocab(), smbert::model());
}

fn bench_onnx_smbert_dynquant(manager: &mut Criterion) {
    bench_onnx(
        manager,
        "Onnx SMBert Quantized",
        smbert::vocab(),
        smbert::model_quant(),
    );
}

fn bench_onnx_qambert_nonquant(manager: &mut Criterion) {
    bench_onnx(manager, "Onnx QAMBert", qambert::vocab(), qambert::model());
}

fn bench_onnx_qambert_dynquant(manager: &mut Criterion) {
    bench_onnx(
        manager,
        "Onnx QAMBert Quantized",
        qambert::vocab(),
        qambert::model_quant(),
    );
}

criterion_group! {
    name = bench_tract_smbert;
    config = Criterion::default();
    targets =
        bench_tract_smbert_nonquant,
        bench_tract_smbert_dynquant,
}

criterion_group! {
    name = bench_tract_qambert;
    config = Criterion::default();
    targets =
        bench_tract_qambert_nonquant,
        bench_tract_qambert_dynquant,
}

criterion_group! {
    name = bench_onnx_smbert;
    config = Criterion::default();
    targets =
        bench_onnx_smbert_nonquant,
        bench_onnx_smbert_dynquant,
}

criterion_group! {
    name = bench_onnx_qambert;
    config = Criterion::default();
    targets =
        bench_onnx_qambert_nonquant,
        bench_onnx_qambert_dynquant,
}

criterion_main! {
    bench_tract_smbert,
    bench_tract_qambert,
    bench_onnx_smbert,
    bench_onnx_qambert,
}
