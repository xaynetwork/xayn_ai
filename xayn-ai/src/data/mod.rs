#![allow(dead_code)]

pub mod document;
pub mod document_data;

use rubert::Embeddings;

#[repr(transparent)]
#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct CoiId(pub usize);

pub struct Coi {
    pub point: Embeddings,
}

pub struct UserInterests {
    positive: Vec<Coi>,
    negative: Vec<Coi>,
}

pub struct Analytics {}
