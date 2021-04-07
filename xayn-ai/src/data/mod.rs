pub mod document;
pub mod document_data;

use serde::{Deserialize, Serialize};

use crate::bert::Embedding;

#[repr(transparent)]
#[cfg_attr(test, derive(Debug))]
#[derive(PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize)]
pub struct CoiId(pub usize);

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
pub struct Coi {
    pub id: CoiId,
    pub point: Embedding,
    pub alpha: f32,
    pub beta: f32,
}

impl Coi {
    pub fn new(id: usize, point: Embedding) -> Self {
        Self {
            id: CoiId(id),
            point,
            alpha: 1.,
            beta: 1.,
        }
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
pub struct UserInterests {
    pub positive: Vec<Coi>,
    pub negative: Vec<Coi>,
}

impl UserInterests {
    pub const fn new() -> Self {
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
