#![allow(dead_code)]

pub mod document;
pub mod document_data;

#[derive(Clone)]
#[repr(transparent)]
#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct EmbeddingPoint(pub Vec<f32>);

#[derive(Clone)]
#[repr(transparent)]
#[cfg_attr(test, derive(Debug, PartialEq, Clone))]
pub struct CenterOfInterestId(pub usize);

pub struct CenterOfInterest {
    pub point: EmbeddingPoint,
}

pub struct CentersOfInterest {
    positive: Vec<CentersOfInterest>,
    negative: Vec<CentersOfInterest>,
}

pub struct Analytics {}
