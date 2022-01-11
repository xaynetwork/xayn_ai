//! Personalized document that is returned from [`Engine`](crate::engine::Engine).

use std::convert::TryFrom;

use derive_more::Display;
use displaydoc::Display as DisplayDoc;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::embedding::utils::Embedding;

/// Errors that could happen when constructing a [`Document`].
#[derive(Error, Debug, DisplayDoc)]
pub enum Error {
    /// Failed to parse Uuid: {0}.
    Parse(#[from] uuid::Error),
}

/// Unique identifier of the [`Document`].
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, Display)]
#[cfg_attr(test, derive(Default))]
pub struct Id(pub Uuid);

impl Id {
    /// Creates a [`Id`] from a 128bit value in big-endian order.
    pub fn from_u128(id: u128) -> Self {
        Id(Uuid::from_u128(id))
    }
}

impl TryFrom<&[u8]> for Id {
    type Error = Error;

    fn try_from(id: &[u8]) -> Result<Self, Self::Error> {
        Ok(Id(Uuid::from_slice(id)?))
    }
}

/// Represents a result from a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier of the document.
    pub id: Id,

    /// Stack from which the document has been taken.
    // pub stack_id: StackId,

    /// Position of the document from the source.
    pub rank: usize,

    /// Text title of the document.
    pub title: String,

    /// Text snippet of the document.
    pub snippet: String,

    /// URL of the document.
    pub url: String,

    /// Domain of the document.
    pub domain: String,

    /// Embedding from smbert.
    pub smbert_embedding: Embedding,
}
