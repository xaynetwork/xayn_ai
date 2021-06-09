pub mod document;
pub(crate) mod document_data;

use serde::{Deserialize, Serialize};

use crate::embedding::smbert::Embedding;

// Hint: We use this id new-type in FFI so repr(transparent) needs to be kept
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize)]
pub struct CoiId(pub usize);

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
    pub fn new(id: usize, point: Embedding) -> Self {
        Self {
            id: CoiId(id),
            point,
            alpha: 1.,
            beta: 1.,
        }
    }
}

impl NegativeCoi {
    pub fn new(id: usize, point: Embedding) -> Self {
        Self {
            id: CoiId(id),
            point,
        }
    }
}

pub(crate) trait CoiPoint {
    fn new(id: usize, embedding: Embedding) -> Self;
    fn point(&self) -> &Embedding;
    fn set_point(&mut self, embedding: Embedding);
}

macro_rules! impl_coi_point {
    ($type:ty) => {
        impl CoiPoint for $type {
            fn new(id: usize, embedding: Embedding) -> Self {
                <$type>::new(id, embedding)
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
