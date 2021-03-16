#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]
//! A Bert tokenizer which converts sequences into encodings.
//!
//! This is a very condensed and heavily refactored version of [huggingface's `tokenizers`] crate.
//!
//! The tokenizer is based on a word piece vocabulary and consists of a Bert normalizer, a Bert
//! pre-tokenizer, a Bert word piece model and a Bert post-tokenizer including truncation and
//! padding strategies. The encodings can be of any numerical data type which implements
//! [`Num`]` + `[`FromPrimitive`]` + `[`Copy`].
//!
//! The normalizer is configurable by:
//! - Cleans any control characters and replaces all sorts of whitespace by ` `.
//! - Separates Chinese characters by whitespace so they get split.
//! - Keeps accents of characters.
//! - Lowercases characters.
//!
//! The pre-tokenizer is not configurable.
//!
//! The word piece model is configurable by:
//! - The unknown token.
//! - The continuing subword prefix.
//! - The maximum number of characters per word.
//!
//! The post-tokenizer is configurable by:
//! - The class token.
//! - The separation token.
//! - A truncation strategy.
//! - A padding strategy.
//!
//! ```no_run
//! use rubert_tokenizer::{Builder, Padding, Truncation};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let tokenizer = Builder::<u32>::from_file("vocab.txt")?
//!         .with_normalizer(true, true, false, true)
//!         .with_model("[UNK]", "##", 100)
//!         .with_post_tokenizer("[CLS]", "[SEP]")
//!         .with_truncation(Truncation::fixed(128, 0))
//!         .with_padding(Padding::fixed(128, "[PAD]"))
//!         .build()?;
//!
//!     let encoding = tokenizer.encode("This is a sequence.");
//!     let encoding = tokenizer.encode_batch(&["This is a sequence.", "And another one!"]);
//!
//!     Ok(())
//! }
//! ```
//!
//! [huggingface's `tokenizers`]: https://crates.io/crates/tokenizers
//! [`Num`]: num_traits::Num
//! [`FromPrimitive`]: num_traits::FromPrimitive

mod builder;
mod model;
mod normalizer;
mod post_tokenizer;
mod pre_tokenizer;
mod tokenizer;

pub use crate::{
    builder::{Builder, BuilderError},
    model::ModelError,
    post_tokenizer::{
        encoding::Encoding,
        padding::{Padding, PaddingError},
        truncation::{Truncation, TruncationError},
        PostTokenizerError,
    },
    tokenizer::Tokenizer,
};

/// A stack allocated string with a maximum length of eight bytes.
type SmallString = smallstr::SmallString<[u8; 8]>;
