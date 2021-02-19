#![allow(dead_code)]

pub mod document;
pub mod document_data;

#[repr(transparent)]
#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct EmbeddingPoint(pub Vec<f32>);

#[repr(transparent)]
#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct CoiId(pub usize);

pub struct Coi {
    pub point: EmbeddingPoint,
}

pub struct UserInterests {
    positive: Vec<Coi>,
    negative: Vec<Coi>,
}

pub struct Analytics {}
