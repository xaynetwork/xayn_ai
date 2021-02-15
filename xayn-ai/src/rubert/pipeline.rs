use std::io::{BufRead, Read};

use tokenizers::tokenizer::EncodeInput;
use tract_onnx::prelude::TVec;

use crate::rubert::{
    anyhow::{anyhow, Context, Result},
    model::RuBertModel,
    pooler::RuBertPooler,
    tokenizer::RuBertTokenizer,
    utils::ArcArrayD,
};

/// A builder to create a [`RuBert`] model.
pub struct RuBertBuilder<V, M> {
    vocab: V,
    model: M,
    strip_accents: bool,
    lowercase: bool,
    batch_size: usize,
    tokens_size: usize,
    pooler: RuBertPooler,
}

impl<V, M> RuBertBuilder<V, M>
where
    V: BufRead,
    M: Read,
{
    /// Creates a new model builder.
    ///
    /// The default settings are:
    /// - Strips accents and makes lower case.
    /// - Supports batch size of 10 and tokens size of 128.
    /// - Applies no additional pooling.
    pub fn new(vocab: V, model: M) -> Self {
        Self {
            vocab,
            model,
            strip_accents: true,
            lowercase: true,
            batch_size: 10,
            tokens_size: 128,
            pooler: RuBertPooler::None,
        }
    }

    /// Toggles accent stripping for the tokenizer.
    pub fn with_strip_accents(mut self, toggle: bool) -> Self {
        self.strip_accents = toggle;
        self
    }

    /// Toggles lower casing for the tokenizer.
    pub fn with_lowercase(mut self, toggle: bool) -> Self {
        self.lowercase = toggle;
        self
    }

    /// Sets the batch size for the model.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Sets the tokens size for the tokenizer and the model.
    pub fn with_tokens_size(mut self, size: usize) -> Self {
        self.tokens_size = size;
        self
    }

    /// Sets pooling for the model.
    pub fn with_pooling(mut self, pooler: RuBertPooler) -> Self {
        self.pooler = pooler;
        self
    }

    /// Creates a model from the builder.
    ///
    /// # Errors
    /// Fails on invalid tokenizer or model settings.
    pub fn build(self) -> Result<RuBert> {
        if self.batch_size == 0 {
            return Err(anyhow!("batch size must be greater than 0"));
        }
        if self.tokens_size < 2 {
            return Err(anyhow!(
                "tokens size must be greater than 2 to allow for special tokens"
            ));
        }

        let tokenizer = RuBertTokenizer::new(
            self.vocab,
            self.strip_accents,
            self.lowercase,
            self.tokens_size,
        )
        .map_err(|_| anyhow!("building tokenizer failed"))?;

        let model = RuBertModel::new(self.model, self.batch_size, self.tokens_size)
            .context("building model failed")?;
        let pooler = self.pooler;

        Ok(RuBert {
            tokenizer,
            model,
            pooler,
        })
    }
}

/// A Bert model pipeline.
///
/// Can be created via the [`RuBertBuilder`] and consists of a tokenizer, a model and optionally a
/// pooler.
pub struct RuBert {
    tokenizer: RuBertTokenizer,
    model: RuBertModel,
    pooler: RuBertPooler,
}

impl RuBert {
    /// Runs prediction on the `sentences` including tokenization and optional pooling.
    ///
    /// The output dimensionality depends on the type of Bert model loaded from onnx.
    ///
    /// # Errors
    /// - The tokenization fails if any of the normalization, tokenization, pre- or post-processing
    /// steps fails.
    /// - The prediction fails on dimensionality mismatches for the tokenized sentences regarding
    /// the loaded onnx model.
    /// - The pooling fails on dimensionality mismatches from the predictions regarding the loaded
    /// onnx model.
    pub fn predict<'s>(
        &self,
        sentences: Vec<impl Into<EncodeInput<'s>> + Send>,
    ) -> Result<ArcArrayD<f32>> {
        let (input_ids, attention_masks, token_type_ids) = self
            .tokenizer
            .encode(sentences)
            .map_err(|_| anyhow!("tokenization failed"))?;
        let predictions = self
            .model
            .predict(input_ids, attention_masks.view(), token_type_ids)
            .context("prediction failed")?;
        let pooling = self
            .pooler
            .pool(predictions, attention_masks)
            .context("pooling failed")?;

        Ok(pooling)
    }

    /// Returns the batch size of the model pipeline.
    pub fn batch_size(&self) -> usize {
        self.model.batch_size()
    }

    /// Returns the tokens size of the model pipeline.
    pub fn tokens_size(&self) -> usize {
        self.model.tokens_size()
    }

    /// Returns the embedding size of the model pipeline.
    ///
    /// # Panics
    /// This assumes that the model output is not empty.
    pub fn embedding_size(&self) -> usize {
        self.model.embedding_size()
    }

    /// Returns the dimensionality of the model pipeline output.
    ///
    /// # Panics
    /// This assumes that the model output is not empty.
    pub fn output_rank(&self) -> usize {
        let rank = self.model.output_rank();
        match self.pooler {
            RuBertPooler::None => rank,
            RuBertPooler::Average | RuBertPooler::First => rank - 1,
        }
    }

    /// Returns the shape of the model pipeline output.
    ///
    /// # Panics
    /// This assumes that the model output is not empty.
    pub fn output_shape(&self) -> TVec<usize> {
        let mut shape = self.model.output_shape();
        match self.pooler {
            RuBertPooler::None => shape,
            RuBertPooler::Average | RuBertPooler::First => {
                shape.remove(1);
                shape
            }
        }
    }
}
