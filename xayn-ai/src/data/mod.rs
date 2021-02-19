#![allow(dead_code)]

pub mod document;
pub mod document_data;

#[repr(transparent)]
#[cfg_attr(not(test), derive(Clone))]
#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct EmbeddingPoint(pub Vec<f32>);

#[repr(transparent)]
#[cfg_attr(not(test), derive(Clone))]
#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct CoiId(pub usize);

pub struct Coi {
    pub point: EmbeddingPoint,
}

pub struct UserInterests {
    positive: Vec<Coi>,
    negative: Vec<Coi>,
}

pub struct Analytics {}
