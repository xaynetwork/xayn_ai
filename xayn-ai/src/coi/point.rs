use serde::{Deserialize, Serialize};

use crate::{coi::CoiId, embedding::utils::Embedding};

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.0.1"))]
#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) struct PositiveCoi {
    #[obake(cfg(">=0.0.0"))]
    pub id: CoiId,
    #[obake(cfg(">=0.0.0"))]
    pub point: Embedding,
    #[obake(cfg(">=0.0.0"))]
    pub alpha: f32,
    #[obake(cfg(">=0.0.0"))]
    pub beta: f32,
}

impl From<PositiveCoi_v0_0_0> for PositiveCoi {
    fn from(coi: PositiveCoi_v0_0_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
            alpha: coi.alpha,
            beta: coi.beta,
        }
    }
}

#[cfg(test)]
impl PositiveCoi_v0_0_0 {
    pub fn new(id: CoiId, point: Embedding) -> Self {
        Self {
            id,
            point,
            alpha: 1.,
            beta: 1.,
        }
    }
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

#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) struct NegativeCoi {
    pub id: CoiId,
    pub point: Embedding,
}

impl NegativeCoi {
    pub fn new(id: CoiId, point: Embedding) -> Self {
        Self { id, point }
    }
}

pub(crate) trait CoiPoint {
    fn new(id: CoiId, embedding: Embedding) -> Self;
    fn id(&self) -> CoiId;
    fn set_id(&mut self, id: CoiId);
    fn point(&self) -> &Embedding;
    fn set_point(&mut self, embedding: Embedding);
}

macro_rules! impl_coi_point {
    ($($(#[$attribute:meta])* $type:ty),+ $(,)?) => {
        $(
            $(#[$attribute])*
            impl CoiPoint for $type {
                fn new(id: CoiId, embedding: Embedding) -> Self {
                    <$type>::new(id, embedding)
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
        )+
    };
}

impl_coi_point! {
    #[cfg(test)] PositiveCoi_v0_0_0,
    PositiveCoi,
    NegativeCoi,
}

pub(crate) trait CoiPointExt {
    fn merge(self, other: Self, id: CoiId) -> Self;
}

macro_rules! impl_coi_point_ext {
    ($($(#[$attribute:meta])* $type:ty),+ $(,)?) => {
        $(
            $(#[$attribute])*
            impl CoiPointExt for $type {
                fn merge(self, other: Self, id: CoiId) -> Self {
                    self.merge(other, id)
                }
            }
        )+
    };
}

impl_coi_point_ext! {
    PositiveCoi,
    NegativeCoi,
}

// generic types can't be versioned, but aliasing and propper naming in the proc macro call works
#[allow(non_camel_case_types)]
pub(crate) type PositiveCois_v0_0_0 = Vec<PositiveCoi_v0_0_0>;
#[allow(non_camel_case_types)]
pub(crate) type PositiveCois_v0_0_1 = Vec<PositiveCoi>;
