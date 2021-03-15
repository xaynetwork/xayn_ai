#![allow(dead_code)]

pub mod document;
pub mod document_data;

use rubert::Embeddings;

#[repr(transparent)]
#[cfg_attr(test, derive(Debug))]
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct CoiId(pub usize);

#[cfg_attr(test, derive(Clone))]
pub struct Coi {
    pub id: CoiId,
    pub point: Embeddings,
    pub alpha: f32,
    pub beta: f32,
}

impl Coi {
    pub fn new(id: usize, point: Embeddings) -> Self {
        Self {
            id: CoiId(id),
            point,
            alpha: 1.,
            beta: 1.,
        }
    }
}

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

pub struct Analytics {}
