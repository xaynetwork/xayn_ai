pub mod document;
pub(crate) mod document_data;

use derive_more::From;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::embedding::utils::Embedding;

// Hint: We use this id new-type in FFI so repr(transparent) needs to be kept
#[repr(transparent)]
#[derive(
    Debug, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub struct CoiId(Uuid);

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct PositiveCoi {
    pub id: CoiId,
    pub point: Embedding,
    pub alpha: f32,
    pub beta: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct NegativeCoi {
    pub id: CoiId,
    pub point: Embedding,
}

impl PositiveCoi {
    pub fn new(id: CoiId, point: Embedding) -> Self {
        Self {
            id,
            point,
            alpha: 1.,
            beta: 1.,
        }
    }
}

impl NegativeCoi {
    pub fn new(id: CoiId, point: Embedding) -> Self {
        Self { id, point }
    }
}

pub(crate) trait CoiPoint {
    fn new(id: CoiId, embedding: Embedding) -> Self;
    fn merge(self, other: Self, id: CoiId) -> Self;
    fn id(&self) -> CoiId;
    fn set_id(&mut self, id: CoiId);
    fn point(&self) -> &Embedding;
    fn set_point(&mut self, embedding: Embedding);
}

macro_rules! impl_coi_point {
    ($type:ty) => {
        impl CoiPoint for $type {
            fn new(id: CoiId, embedding: Embedding) -> Self {
                <$type>::new(id, embedding)
            }

            fn merge(self, other: Self, id: CoiId) -> Self {
                self.merge(other, id)
            }

            fn id(&self) -> CoiId {
                self.id
            }

            fn set_id(&mut self, id: CoiId) {
                self.id = id;
            }

            fn point(&self) -> &Embedding {
                &self.point
            }

            fn set_point(&mut self, embedding: Embedding) {
                self.point = embedding;
            }
        }
    };
}

impl_coi_point!(PositiveCoi);
impl_coi_point!(NegativeCoi);

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct UserInterests {
    pub positive: Vec<PositiveCoi>,
    pub negative: Vec<NegativeCoi>,
}

impl UserInterests {
    pub(crate) const fn new() -> Self {
        Self {
            positive: Vec::new(),
            negative: Vec::new(),
        }
    }
}

impl Default for UserInterests {
    fn default() -> Self {
        UserInterests::new()
    }
}
