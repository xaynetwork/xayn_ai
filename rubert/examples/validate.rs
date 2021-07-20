//! Compare MBert models evaluated by the onnx or the tract runtime.

use std::{
    marker::PhantomPinned,
    ops::{Bound, Deref, RangeBounds},
};

use csv::Reader;
use indicatif::ProgressBar;
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel};

use rubert::{
    kinds::{QAMBert, SMBert},
    Builder as BertBuilder,
    Embedding2,
    NonePooler,
    Pipeline as BertPipeline,
};
use rubert_tokenizer::{Builder as TokenizerBuilder, Padding, Tokenizer, Truncation};

fn main() {
    ValidatorConfig {
        tokenizer: TokenizerConfig {
            accents: false,
            lowercase: true,
            token_size: 90,
        },
        source: ModelConfig {
            kind: ModelKind::OnnxMBert,
            vocab: "../data/smbert_v0000/vocab.txt",
            model: "../data/smbert_v0000/smbert.onnx",
        },
        target: ModelConfig {
            kind: ModelKind::TractSMBert,
            vocab: "../data/smbert_v0000/vocab.txt",
            model: "../data/smbert_v0000/smbert.onnx",
        },
        data: DataConfig {
            talks: "../data/ted_talk_transcripts.csv",
            range: ..100,
        },
    }
    .build()
    .validate()
    .print();
}

/// The available model kinds.
#[allow(dead_code)]
enum ModelKind {
    /// A SMBert or QAMBert model for the onnx runtime.
    OnnxMBert,
    /// A SMBert model for the tract runtime.
    TractSMBert,
    /// A QAMBert model for the tract runtime.
    TractQAMBert,
}

/// Tokenizer configurations.
struct TokenizerConfig {
    /// Whether to keep the accents on characters.
    accents: bool,
    /// Whether to lowercase words.
    lowercase: bool,
    /// The number of tokens for truncation/padding.
    token_size: usize,
}

/// Source or target model configurations.
struct ModelConfig {
    /// The model kind.
    kind: ModelKind,
    /// The path to the vocabulary.
    vocab: &'static str,
    /// The path to the model.
    model: &'static str,
}

/// Ted talks data configurations.
struct DataConfig<R: RangeBounds<usize>> {
    /// The path to the talks.
    talks: &'static str,
    /// The range of talks to use for validation.
    range: R,
}

/// Combined validation configurations.
struct ValidatorConfig<R: RangeBounds<usize>> {
    tokenizer: TokenizerConfig,
    source: ModelConfig,
    target: ModelConfig,
    data: DataConfig<R>,
}

impl<R: RangeBounds<usize>> ValidatorConfig<R> {
    /// Builds a validator from this configuration.
    fn build(&self) -> Validator {
        Validator::build(self)
    }
}

/// The available MBert model pipelines.
enum Pipeline {
    /// A SMBert or QAMBert model pipeline for the onnx runtime.
    OnnxMBert {
        tokenizer: Tokenizer<i64>,
        environment: Environment,
        session: Option<Session<'static>>,
        _pinned: PhantomPinned,
    },
    /// A SMBert model pipeline for the tract runtime.
    TractSMBert(BertPipeline<SMBert, NonePooler>),
    /// A QAMBert model pipeline for the tract runtime.
    TractQAMBert(BertPipeline<QAMBert, NonePooler>),
}

impl Pipeline {
    /// Builds a pipeline from a tokenizer and model configuration.
    fn build(tokenizer: &TokenizerConfig, model: &ModelConfig) -> Self {
        match model.kind {
            ModelKind::OnnxMBert => {
                let tokenizer = TokenizerBuilder::from_file(model.vocab)
                    .unwrap()
                    .with_normalizer(true, false, tokenizer.accents, tokenizer.lowercase)
                    .with_model("[UNK]", "##", 100)
                    .with_post_tokenizer("[CLS]", "[SEP]")
                    .with_truncation(Truncation::fixed(tokenizer.token_size, 0))
                    .with_padding(Padding::fixed(tokenizer.token_size, "[PAD]"))
                    .build()
                    .unwrap();
                let environment = Environment::builder().build().unwrap();
                let mut pipeline = Self::OnnxMBert {
                    tokenizer,
                    environment,
                    session: None,
                    _pinned: PhantomPinned,
                };
                if let Self::OnnxMBert {
                    ref environment,
                    ref mut session,
                    ..
                } = pipeline
                {
                    *session = unsafe {
                        // Safety: environment is pinned and lives as long as session
                        &*(environment as *const Environment)
                    }
                    .new_session_builder()
                    .unwrap()
                    .with_optimization_level(GraphOptimizationLevel::DisableAll)
                    .unwrap()
                    .with_model_from_file(model.model)
                    .unwrap()
                    .into()
                }

                pipeline
            }
            ModelKind::TractSMBert => {
                let pipeline = BertBuilder::from_files(model.vocab, model.model)
                    .unwrap()
                    .with_accents(tokenizer.accents)
                    .with_lowercase(tokenizer.lowercase)
                    .with_token_size(tokenizer.token_size)
                    .unwrap()
                    .with_pooling(NonePooler)
                    .build()
                    .unwrap();

                Self::TractSMBert(pipeline)
            }
            ModelKind::TractQAMBert => {
                let pipeline = BertBuilder::from_files(model.vocab, model.model)
                    .unwrap()
                    .with_accents(tokenizer.accents)
                    .with_lowercase(tokenizer.lowercase)
                    .with_token_size(tokenizer.token_size)
                    .unwrap()
                    .with_pooling(NonePooler)
                    .build()
                    .unwrap();

                Self::TractQAMBert(pipeline)
            }
        }
    }

    /// Runs the model pipeline to infer the embedding of a sequence.
    fn run(&mut self, sequence: impl AsRef<str>) -> Embedding2 {
        match self {
            Self::OnnxMBert {
                tokenizer, session, ..
            } => {
                let encoding = tokenizer.encode(sequence);
                let (token_ids, type_ids, _, _, _, _, attention_mask, _) = encoding.into();
                let inputs = vec![
                    Array1::<i64>::from(token_ids).insert_axis(Axis(0)),
                    Array1::<i64>::from(attention_mask).insert_axis(Axis(0)),
                    Array1::<i64>::from(type_ids).insert_axis(Axis(0)),
                ];
                let outputs = session.as_mut().unwrap().run(inputs).unwrap();

                outputs[0].slice(s![0, .., ..]).to_owned().into()
            }
            Self::TractSMBert(pipeline) => pipeline.run(sequence).unwrap(),
            Self::TractQAMBert(pipeline) => pipeline.run(sequence).unwrap(),
        }
    }
}

/// A validator to compare two models based on a set of Ted talks.
struct Validator {
    talks: &'static str,
    skip: usize,
    take: usize,
    source: Pipeline,
    target: Pipeline,
    errors: Array1<f32>,
}

impl Validator {
    /// Builds a validator from a configuration.
    fn build<R: RangeBounds<usize>>(config: &ValidatorConfig<R>) -> Self {
        let talks = config.data.talks;
        let skip = match config.data.range.start_bound() {
            Bound::Included(start) => *start,
            Bound::Excluded(start) => start + 1,
            Bound::Unbounded => 0,
        };
        let take = match config.data.range.end_bound() {
            Bound::Included(end) => end + 1,
            Bound::Excluded(end) => *end,
            Bound::Unbounded => 2467,
        } - skip;
        let source = Pipeline::build(&config.tokenizer, &config.source);
        let target = Pipeline::build(&config.tokenizer, &config.target);
        let errors = Array1::zeros(11);

        Self {
            talks,
            skip,
            take,
            source,
            target,
            errors,
        }
    }

    /// Computes the l1-norm of the difference between source and target.
    fn mean_absolute_error(source: &Embedding2, target: &Embedding2) -> f32 {
        (source.deref() - target.deref())
            .mapv(|v| v.abs())
            .mean()
            .unwrap_or_default()
    }

    /// Computes the l1-norm of the difference between source and target relative to the source.
    fn mean_relative_error(source: &Embedding2, target: &Embedding2) -> f32 {
        ((source.deref() - target.deref()) / source.deref())
            .mapv(|v| v.is_finite().then(|| v.abs()).unwrap_or_default())
            .mean()
            .unwrap_or_default()
    }

    /// Computes the l2-norm of the difference between source and target.
    fn mean_squared_absolute_error(source: &Embedding2, target: &Embedding2) -> f32 {
        (source.deref() - target.deref())
            .mapv(|v| v.powi(2))
            .mean()
            .unwrap_or_default()
            .sqrt()
    }

    /// Computes the l2-norm of the difference between source and target relative to the source.
    fn mean_squared_relative_error(source: &Embedding2, target: &Embedding2) -> f32 {
        ((source.deref() - target.deref()) / source.deref())
            .mapv(|v| v.is_finite().then(|| v.powi(2)).unwrap_or_default())
            .mean()
            .unwrap_or_default()
            .sqrt()
    }

    /// Computes the cosine similarity between source and target.
    fn cosine_similarity(source: &Embedding2, target: &Embedding2) -> f32 {
        let norm = source.mapv(|v| v.powi(2)).mean().unwrap_or_default().sqrt()
            * target.mapv(|v| v.powi(2)).mean().unwrap_or_default().sqrt();
        (norm.is_finite() && norm != 0.0)
            .then(|| {
                (source.deref() * target.deref() / norm)
                    .sum()
                    .clamp(-1.0, 1.0)
            })
            .unwrap_or_default()
    }

    /// Computes various errors between source and target embeddings based on the chosen ted talks.
    fn validate(&mut self) -> &mut Self {
        let mut reader = Reader::from_path(self.talks).unwrap();
        let progress = ProgressBar::new(self.take as u64);
        let mut errors = Array2::<f32>::zeros((330644, 5));
        let mut idx = 0;

        for record in reader.records().skip(self.skip).take(self.take) {
            for sequence in record.unwrap()[0].split_inclusive(&['.', '!', '?'] as &[char]) {
                let source = self.source.run(sequence);
                let target = self.target.run(sequence);
                errors.slice_mut(s![idx, ..]).assign(&ArrayView1::from(&[
                    Self::mean_absolute_error(&source, &target),
                    Self::mean_relative_error(&source, &target),
                    Self::mean_squared_absolute_error(&source, &target),
                    Self::mean_squared_relative_error(&source, &target),
                    Self::cosine_similarity(&source, &target),
                ]));
                idx += 1;
            }
            progress.inc(1);
        }
        progress.finish();

        self.errors[0] = idx as f32;
        self.errors
            .slice_mut(s![1..6])
            .assign(&errors.slice(s![..idx, ..]).mean_axis(Axis(0)).unwrap());
        self.errors
            .slice_mut(s![6..])
            .assign(&errors.slice(s![..idx, ..]).std_axis(Axis(0), 1.0));

        self
    }

    /// Prints the validation results to stdout.
    fn print(&self) {
        println!(
            "Validated models on {} talks with {} sentences.",
            self.take, self.errors[0],
        );
        println!(
            "  Mean absolute error: μ = {}, σ = {}",
            self.errors[1], self.errors[6],
        );
        println!(
            "  Mean relative error: μ = {}, σ = {}",
            self.errors[2], self.errors[7],
        );
        println!(
            "  Mean squared absolute error: μ = {}, σ = {}",
            self.errors[3], self.errors[8],
        );
        println!(
            "  Mean squared relative error: μ = {}, σ = {}",
            self.errors[4], self.errors[9],
        );
        println!(
            "  Cosine similarities: μ = {}, σ = {}",
            self.errors[5], self.errors[10],
        );
    }
}
