pub mod document;
pub(crate) mod document_data;

use itertools::izip;
use serde::{Deserialize, Serialize};

use crate::embedding::utils::{mean, Embedding};

// Hint: We use this id new-type in FFI so repr(transparent) needs to be kept
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize, PartialOrd)]
pub struct CoiId(pub usize);

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct PositiveCoi {
    pub id: CoiId,
    pub point: Embedding,
    pub alpha: f32,
    pub beta: f32,
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct NegativeCoi {
    pub id: CoiId,
    pub point: Embedding,
}

impl PositiveCoi {
    pub fn new(id: usize, point: Embedding) -> Self {
        Self {
            id: CoiId(id),
            point,
            alpha: 1.,
            beta: 1.,
        }
    }

    pub(crate) fn merge(self, other: Self, id: usize) -> Self {
        let point = mean(&self.point, &other.point);
        let (alpha, beta) = merge_params(self.alpha, self.beta, other.alpha, other.beta);
        Self {
            id: CoiId(id),
            point,
            alpha,
            beta,
        }
    }
}

/// Calculates an "average" beta distribution ~B(a, b) from the given two ~B(`a1`, `b1`), ~B(`a2`, `b2`).
///
/// <https://xainag.atlassian.net/wiki/spaces/XAY/pages/2029944833/CoI+synchronisation>
/// outlines the calculation.
fn merge_params(a1: f32, b1: f32, a2: f32, b2: f32) -> (f32, f32) {
    let mean = |a, b| a / (a + b);
    let var = |a, b| a * b / (f32::powi(a + b, 2) * (a + b + 1.));

    // geometric average of the mean and variance
    let avg_mean = f32::sqrt(mean(a1, b1) * mean(a2, b2));
    let avg_var = f32::sqrt(var(a1, b1) * var(a2, b2));

    let factor = avg_mean * (1. - avg_mean) / avg_var - 1.;
    (avg_mean * factor, (1. - avg_mean) * factor)
}

impl NegativeCoi {
    pub fn new(id: usize, point: Embedding) -> Self {
        Self {
            id: CoiId(id),
            point,
        }
    }

    pub fn merge(self, other: Self, id: usize) -> Self {
        let id = CoiId(id);
        let point = mean(&self.point, &other.point);
        Self { id, point }
    }
}

pub(crate) trait CoiPoint {
    fn new(id: usize, embedding: Embedding) -> Self;
    fn merge(self, other: Self, id: usize) -> Self;
    fn id(&self) -> CoiId;
    fn set_id(&mut self, id: usize);
    fn point(&self) -> &Embedding;
    fn set_point(&mut self, embedding: Embedding);
}

macro_rules! impl_coi_point {
    ($type:ty) => {
        impl CoiPoint for $type {
            fn new(id: usize, embedding: Embedding) -> Self {
                <$type>::new(id, embedding)
            }

            fn merge(self, other: Self, id: usize) -> Self {
                self.merge(other, id)
            }

            fn id(&self) -> CoiId {
                self.id
            }

            fn set_id(&mut self, id: usize) {
                self.id = CoiId(id);
            }

            fn point(&self) -> &Embedding {
                &self.point
            }

            fn set_point(&mut self, embedding: Embedding) {
                self.point = embedding;
            }
        }
    };
}

impl_coi_point!(PositiveCoi);
impl_coi_point!(NegativeCoi);

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct UserInterests {
    pub positive: Vec<PositiveCoi>,
    pub negative: Vec<NegativeCoi>,
}

impl UserInterests {
    pub(crate) const fn new() -> Self {
        Self {
            positive: Vec::new(),
            negative: Vec::new(),
        }
    }

    /// Moves all user interests of `other` into `Self`.
    pub(crate) fn append(&mut self, mut other: Self) {
        // TODO drop dupes in other
        append_cois(&mut self.positive, &mut other.positive);
        append_cois(&mut self.negative, &mut other.negative);
    }

    /// Re-assigns CoI ids for normalization.
    pub(crate) fn reassign_ids(&mut self) {
        reassign_coi_ids(&mut self.positive);
        reassign_coi_ids(&mut self.negative);
    }
}

fn append_cois<C>(locals: &mut Vec<C>, remotes: &mut Vec<C>)
where
    C: CoiPoint,
{
    // shift remote ids to avoid clashes with local ids
    let max_local_id = locals.iter().map(|coi| coi.id().0).max().unwrap_or(0);
    remotes
        .iter_mut()
        .for_each(|coi| coi.set_id(max_local_id + coi.id().0));

    locals.append(remotes);
}

fn reassign_coi_ids(cois: &mut Vec<impl CoiPoint>) {
    for (id, coi) in izip!(1..cois.len() + 1, cois) {
        coi.set_id(id)
    }
}

impl Default for UserInterests {
    fn default() -> Self {
        UserInterests::new()
    }
}
