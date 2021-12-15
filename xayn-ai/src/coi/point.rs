use std::{collections::HashSet, time::Duration};

use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::{
    coi::{key_phrase::KeyPhrase, stats::CoiStats, CoiId},
    embedding::utils::Embedding,
};

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[obake(version("0.3.0"))]
#[derive(Clone, Debug, Derivative, Deserialize, Serialize)]
#[derivative(PartialEq)]
pub(crate) struct PositiveCoi {
    #[obake(cfg(">=0.0"))]
    pub(super) id: CoiId,
    #[obake(cfg(">=0.0"))]
    pub(super) point: Embedding,
    #[obake(cfg(">=0.3"))]
    pub(super) key_phrases: HashSet<KeyPhrase>,
    #[obake(cfg(">=0.3"))]
    #[derivative(PartialEq = "ignore")]
    pub(super) stats: CoiStats,

    // removed fields go below this line
    #[obake(cfg(">=0.0, <0.2"))]
    pub(super) alpha: f32,
    #[obake(cfg(">=0.0, <0.2"))]
    pub(super) beta: f32,
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

impl From<PositiveCoi_v0_1_0> for PositiveCoi_v0_2_0 {
    fn from(coi: PositiveCoi_v0_1_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
        }
    }
}

impl From<PositiveCoi_v0_2_0> for PositiveCoi {
    fn from(coi: PositiveCoi_v0_2_0) -> Self {
        Self {
            id: coi.id,
            point: coi.point,
            key_phrases: HashSet::default(),
            stats: CoiStats::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct NegativeCoi {
    pub id: CoiId,
    pub point: Embedding,
}

pub(crate) trait CoiPoint {
    fn new(
        id: CoiId,
        point: Embedding,
        key_phrases: HashSet<KeyPhrase>,
        viewed: Option<Duration>,
    ) -> Self;

    fn id(&self) -> CoiId;

    fn set_id(&mut self, id: CoiId);

    fn point(&self) -> &Embedding;

    fn set_point(&mut self, embedding: Embedding);
}

macro_rules! coi_point_default_impls {
    () => {
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
    };
}

#[cfg(test)]
impl CoiPoint for PositiveCoi_v0_0_0 {
    fn new(
        id: CoiId,
        point: Embedding,
        _key_phrases: HashSet<KeyPhrase>,
        _viewed: Option<Duration>,
    ) -> Self {
        Self {
            id,
            point,
            alpha: 1.,
            beta: 1.,
        }
    }

    coi_point_default_impls! {}
}

#[cfg(test)]
impl CoiPoint for PositiveCoi_v0_1_0 {
    fn new(
        id: CoiId,
        point: Embedding,
        _key_phrases: HashSet<KeyPhrase>,
        _viewed: Option<Duration>,
    ) -> Self {
        Self {
            id,
            point,
            alpha: 1.,
            beta: 1.,
        }
    }

    coi_point_default_impls! {}
}

#[cfg(test)]
impl CoiPoint for PositiveCoi_v0_2_0 {
    fn new(
        id: CoiId,
        point: Embedding,
        _key_phrases: HashSet<KeyPhrase>,
        _viewed: Option<Duration>,
    ) -> Self {
        Self { id, point }
    }

    coi_point_default_impls! {}
}

impl CoiPoint for PositiveCoi {
    fn new(
        id: CoiId,
        point: Embedding,
        key_phrases: HashSet<KeyPhrase>,
        viewed: Option<Duration>,
    ) -> Self {
        Self {
            id,
            point,
            key_phrases,
            stats: CoiStats::new(viewed),
        }
    }

    coi_point_default_impls! {}
}

impl CoiPoint for NegativeCoi {
    fn new(
        id: CoiId,
        point: Embedding,
        _key_phrases: HashSet<KeyPhrase>,
        _viewed: Option<Duration>,
    ) -> Self {
        Self { id, point }
    }

    coi_point_default_impls! {}
}

// generic types can't be versioned, but aliasing and proper naming in the proc macro call works
#[allow(non_camel_case_types)]
type PositiveCois_v0_0_0 = Vec<PositiveCoi_v0_0_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_1_0 = Vec<PositiveCoi_v0_1_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_2_0 = Vec<PositiveCoi_v0_2_0>;
#[allow(non_camel_case_types)]
type PositiveCois_v0_3_0 = Vec<PositiveCoi>;

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[obake(version("0.3.0"))]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
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

impl From<UserInterests_v0_1_0> for UserInterests_v0_2_0 {
    fn from(ui: UserInterests_v0_1_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<UserInterests_v0_2_0> for UserInterests {
    fn from(ui: UserInterests_v0_2_0) -> Self {
        Self {
            positive: ui.positive.into_iter().map(Into::into).collect(),
            negative: ui.negative.into_iter().map(Into::into).collect(),
        }
    }
}
