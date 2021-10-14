use serde::{Deserialize, Serialize};

use crate::{coi::CoiId, embedding::utils::Embedding};

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) struct PositiveCoi {
    #[obake(cfg(">=0.0"))]
    pub id: CoiId,
    #[obake(cfg(">=0.0"))]
    pub point: Embedding,

    // removed fields go below this line
    #[obake(cfg(">=0.0, <0.2"))]
    pub alpha: f32,
    #[obake(cfg(">=0.0, <0.2"))]
    pub beta: f32,
}

impl From<PositiveCoi_v0_0_0> for PositiveCoi_v0_1_0 {
    fn from(coi: PositiveCoi_v0_0_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
            alpha: coi.alpha,
            beta: coi.beta,
        }
    }
}

impl From<PositiveCoi_v0_0_0> for PositiveCoi {
    fn from(coi: PositiveCoi_v0_0_0) -> Self {
        PositiveCoi_v0_1_0::from(coi).into()
    }
}

impl From<PositiveCoi_v0_1_0> for PositiveCoi {
    fn from(coi: PositiveCoi_v0_1_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
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

#[cfg(test)]
impl PositiveCoi_v0_1_0 {
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
        Self { id, point }
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
    #[cfg(test)] PositiveCoi_v0_1_0,
    PositiveCoi,
    NegativeCoi,
}

pub(crate) trait CoiPointMerge {
    fn merge(self, other: Self, id: CoiId) -> Self;
}

macro_rules! impl_coi_point_merge {
    ($($(#[$attribute:meta])* $type:ty),+ $(,)?) => {
        $(
            $(#[$attribute])*
            impl CoiPointMerge for $type {
                fn merge(self, other: Self, id: CoiId) -> Self {
                    self.merge(other, id)
                }
            }
        )+
    };
}

impl_coi_point_merge! {
    PositiveCoi,
    NegativeCoi,
}

// generic types can't be versioned, but aliasing and proper naming in the proc macro call works
#[allow(non_camel_case_types)]
type PositiveCois_v0_0_0 = Vec<PositiveCoi_v0_0_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_1_0 = Vec<PositiveCoi_v0_1_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_2_0 = Vec<PositiveCoi>;

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[derive(Clone, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) struct UserInterests {
    #[obake(inherit)]
    #[obake(cfg(">=0.0"))]
    pub positive: PositiveCois,
    #[obake(cfg(">=0.0"))]
    pub negative: Vec<NegativeCoi>,
}

impl From<UserInterests_v0_0_0> for UserInterests_v0_1_0 {
    fn from(ui: UserInterests_v0_0_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<UserInterests_v0_0_0> for UserInterests {
    fn from(ui: UserInterests_v0_0_0) -> Self {
        UserInterests_v0_1_0::from(ui).into()
    }
}

impl From<UserInterests_v0_1_0> for UserInterests {
    fn from(ui: UserInterests_v0_1_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}
