use std::{
    mem::swap,
    time::{Duration, SystemTime},
};

#[cfg(test)]
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::{coi::CoiId, embedding::utils::Embedding, utils::system_time_now};

#[derive(Clone, Copy, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug))]
pub(crate) struct CoiStats {
    pub(crate) view_count: usize,
    pub(crate) view_time: Duration,
    pub(crate) last_view: SystemTime,
}

impl CoiStats {
    pub(crate) fn new(viewed: Option<Duration>) -> Self {
        Self {
            view_count: 1,
            view_time: viewed.unwrap_or_default(),
            last_view: system_time_now(),
        }
    }

    pub(crate) fn update(&mut self, viewed: Option<Duration>) {
        self.view_count += 1;
        if let Some(viewed) = viewed {
            self.view_time += viewed;
        }
        self.last_view = system_time_now();
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            view_count: self.view_count + other.view_count,
            view_time: self.view_time + other.view_time,
            last_view: self.last_view.max(other.last_view),
        }
    }
}

impl Default for CoiStats {
    fn default() -> Self {
        Self {
            view_count: 1,
            view_time: Duration::ZERO,
            last_view: SystemTime::UNIX_EPOCH,
        }
    }
}

#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[cfg_attr(test, derive(Debug))]
pub(crate) struct KeyPhrase {
    pub(super) words: String,
    pub(super) point: Embedding,
}

impl KeyPhrase {
    pub(super) fn is_valid(&self) -> bool {
        !self.words.is_empty() && self.point.iter().copied().all(f32::is_finite)
    }

    pub(super) fn is_unique(key_phrases: &[KeyPhrase]) -> bool {
        key_phrases
            .iter()
            .enumerate()
            .all(|(idx, this)| key_phrases.iter().skip(idx + 1).all(|other| this != other))
    }
}

#[obake::versioned]
#[obake(version("0.0.0"))]
#[obake(version("0.1.0"))]
#[obake(version("0.2.0"))]
#[obake(version("0.3.0"))]
#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, Derivative), derivative(PartialEq))]
pub(crate) struct PositiveCoi {
    #[obake(cfg(">=0.0"))]
    pub(super) id: CoiId,
    #[obake(cfg(">=0.0"))]
    pub(super) point: Embedding,
    #[obake(cfg(">=0.3"))]
    pub(super) key_phrases: Vec<KeyPhrase>, // invariant: key phrases must be unique
    #[obake(cfg(">=0.3"))]
    #[cfg_attr(test, derivative(PartialEq = "ignore"))]
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
            key_phrases: Vec::default(),
            stats: CoiStats::default(),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) struct NegativeCoi {
    pub id: CoiId,
    pub point: Embedding,
}

pub(crate) trait CoiPoint {
    fn new(
        id: CoiId,
        point: Embedding,
        key_phrases: Vec<KeyPhrase>,
        viewed: Option<Duration>,
    ) -> Self;

    fn id(&self) -> CoiId;

    fn set_id(&mut self, id: CoiId);

    fn point(&self) -> &Embedding;

    fn set_point(&mut self, embedding: Embedding);
}

pub(crate) trait CoiPointKeyPhrases {
    fn key_phrases(&self) -> &[KeyPhrase];

    fn swap_key_phrases(&mut self, candidates: Vec<KeyPhrase>) -> Vec<KeyPhrase>;
}

pub(crate) trait CoiPointStats {
    fn stats(&self) -> CoiStats;

    fn update_stats(&mut self, viewed: Option<Duration>);
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
        _key_phrases: Vec<KeyPhrase>,
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
        _key_phrases: Vec<KeyPhrase>,
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
        _key_phrases: Vec<KeyPhrase>,
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
        key_phrases: Vec<KeyPhrase>,
        viewed: Option<Duration>,
    ) -> Self {
        debug_assert!(key_phrases.iter().all(KeyPhrase::is_valid));
        debug_assert!(KeyPhrase::is_unique(&key_phrases));
        Self {
            id,
            point,
            key_phrases,
            stats: CoiStats::new(viewed),
        }
    }

    coi_point_default_impls! {}
}

impl CoiPointKeyPhrases for PositiveCoi {
    fn key_phrases(&self) -> &[KeyPhrase] {
        self.key_phrases.as_slice()
    }

    fn swap_key_phrases(&mut self, mut candidates: Vec<KeyPhrase>) -> Vec<KeyPhrase> {
        debug_assert!(candidates.iter().all(KeyPhrase::is_valid));
        debug_assert!(KeyPhrase::is_unique(&candidates));
        swap(&mut self.key_phrases, &mut candidates);
        candidates
    }
}

impl CoiPointStats for PositiveCoi {
    fn stats(&self) -> CoiStats {
        self.stats
    }

    fn update_stats(&mut self, viewed: Option<Duration>) {
        self.stats.update(viewed);
    }
}

impl CoiPoint for NegativeCoi {
    fn new(
        id: CoiId,
        point: Embedding,
        _key_phrases: Vec<KeyPhrase>,
        _viewed: Option<Duration>,
    ) -> Self {
        Self { id, point }
    }

    coi_point_default_impls! {}
}

impl CoiPointKeyPhrases for NegativeCoi {
    fn key_phrases(&self) -> &[KeyPhrase] {
        &[]
    }

    fn swap_key_phrases(&mut self, _candidates: Vec<KeyPhrase>) -> Vec<KeyPhrase> {
        Vec::new()
    }
}

impl CoiPointStats for NegativeCoi {
    fn stats(&self) -> CoiStats {
        CoiStats::default()
    }

    fn update_stats(&mut self, _viewed: Option<Duration>) {}
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
