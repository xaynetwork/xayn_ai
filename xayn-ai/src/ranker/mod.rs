mod context;
mod document;
mod public;
mod system;

pub use self::{
    document::Document,
    public::{Builder, Ranker},
};
pub use crate::{
    coi::{
        config::{Config as CoiSystemConfig, Error as CoiSystemConfigError},
        key_phrase::KeyPhrase,
    },
    embedding::utils::{cosine_similarity, pairwise_cosine_similarity, ArcEmbedding, Embedding},
    DocumentId,
};
pub use rubert::AveragePooler;
