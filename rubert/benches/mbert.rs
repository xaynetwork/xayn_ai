//! Run as `cargo bench --bench mbert --features bench`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{s, Array1, Axis};
use onnxruntime::{environment::Environment, GraphOptimizationLevel};

use rubert::{
    kinds::{QAMBert, SMBert},
    AveragePooler,
    Builder,
    Embedding2,
    FirstPooler,
    NonePooler,
};
use rubert_tokenizer::{Builder as TokenizerBuilder, Padding, Truncation};

const TOKEN_SIZE: usize = 64;
const SEQUENCE: &'static str = "This is a sequence.";

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
            let pipeline = Builder::<_, _, $kind, _>::from_files($vocab, $model)
                .unwrap()
                .with_accents(false)
                .with_lowercase(true)
                .with_token_size(TOKEN_SIZE)
                .unwrap()
                .with_pooling($pooler)
                .build()
                .unwrap();
            group.bench_function($name, |bencher| {
                bencher.iter(|| pipeline.run(black_box(SEQUENCE)).unwrap())
            });
        )+
    };
}

macro_rules! bench_onnx {
    ($manager:expr, $name:expr, $vocab:expr, $model:expr $(,)?) => {
        let tokenizer = TokenizerBuilder::from_file($vocab)
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
            .with_model_from_file($model)
            .unwrap();

        $manager.bench_function($name, |bencher| {
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
    };
}

fn bench_tract_smbert_nonquant(manager: &mut Criterion) {
    bench_tract!(
        manager,
        "Tract SMBert" => SMBert,
        "../data/smbert_v0000/vocab.txt",
        "../data/smbert_v0000/smbert.onnx",
        [
            "None Pooler" => NonePooler,
            "First Pooler" => FirstPooler,
            "Average Pooler" => AveragePooler,
        ],
    );
}

fn bench_tract_smbert_quant(manager: &mut Criterion) {
    bench_tract!(
        manager,
        "Tract SMBert Quantized" => SMBert,
        "../data/smbert_v0000/vocab.txt",
        "../data/smbert_v0000/smbert-quant.onnx",
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
        "../data/qambert_v0001/vocab.txt",
        "../data/qambert_v0001/qambert.onnx",
        [
            "None Pooler" => NonePooler,
            "First Pooler" => FirstPooler,
            "Average Pooler" => AveragePooler,
        ],
    );
}

fn bench_tract_qambert_quant(manager: &mut Criterion) {
    bench_tract!(
        manager,
        "Tract QAMBert Quantized" => QAMBert,
        "../data/qambert_v0001/vocab.txt",
        "../data/qambert_v0001/qambert-quant.onnx",
        [
            "None Pooler" => NonePooler,
            "First Pooler" => FirstPooler,
            "Average Pooler" => AveragePooler,
        ],
    );
}

fn bench_onnx_smbert_nonquant(manager: &mut Criterion) {
    bench_onnx!(
        manager,
        "Onnx SMBert",
        "../data/smbert_v0000/vocab.txt",
        "../data/smbert_v0000/smbert.onnx",
    );
}

fn bench_onnx_smbert_quant(manager: &mut Criterion) {
    bench_onnx!(
        manager,
        "Onnx SMBert Quantized",
        "../data/smbert_v0000/vocab.txt",
        "../data/smbert_v0000/smbert-quant.onnx",
    );
}

fn bench_onnx_qambert_nonquant(manager: &mut Criterion) {
    bench_onnx!(
        manager,
        "Onnx QAMBert",
        "../data/qambert_v0001/vocab.txt",
        "../data/qambert_v0001/qambert.onnx",
    );
}

fn bench_onnx_qambert_quant(manager: &mut Criterion) {
    bench_onnx!(
        manager,
        "Onnx QAMBert Quantized",
        "../data/qambert_v0001/vocab.txt",
        "../data/qambert_v0001/qambert-quant.onnx",
    );
}

criterion_group! {
    name = bench_tract_smbert;
    config = Criterion::default();
    targets = bench_tract_smbert_nonquant, bench_tract_smbert_quant,
}

criterion_group! {
    name = bench_tract_qambert;
    config = Criterion::default();
    targets = bench_tract_qambert_nonquant, bench_tract_qambert_quant,
}

criterion_group! {
    name = bench_onnx_smbert;
    config = Criterion::default();
    targets = bench_onnx_smbert_nonquant, bench_onnx_smbert_quant,
}

criterion_group! {
    name = bench_onnx_qambert;
    config = Criterion::default();
    targets = bench_onnx_qambert_nonquant, bench_onnx_qambert_quant,
}

criterion_main! {
    bench_tract_smbert,
    bench_tract_qambert,
    bench_onnx_smbert,
    bench_onnx_qambert,
}
