pub mod document;
pub(crate) mod document_data;

use serde::{Deserialize, Serialize};

use crate::embedding::utils::Embedding;

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

    #[allow(dead_code)]
    pub(crate) fn merge(self, _id: usize, _other: Self) -> Self {
        todo!()
    }

    pub(crate) fn _merge_into(self, other: &mut Self) {
        let Self {
            point, alpha, beta, ..
        } = self;

        merge_point(point, &mut other.point);
        let (alpha, beta) = merge_params(alpha, beta, other.alpha, other.beta);
        other.alpha = alpha;
        other.beta = beta;
    }

    pub(crate) fn from_merge(&self, other: &Self, id: usize) -> Self {
        let point = average_point(&self.point, &other.point);
        let (alpha, beta) = merge_params(self.alpha, self.beta, other.alpha, other.beta);
        Self {
            id: CoiId(id),
            point,
            alpha,
            beta,
        }
    }
}

/// Merges two points by taking their pointwise arithmetic mean.
fn average_point(_p1: &Embedding, _p2: &Embedding) -> Embedding {
    todo!()
}

#[allow(dead_code)]
fn merge_point(_p1: Embedding, _p2: &mut Embedding) {
    todo!()
}

/// Merges two beta distributions X ~ B(a1, b1), Y ~ B(a2, b2) into an "average" Z ~ B(a, b).
fn merge_params(a1: f32, b1: f32, a2: f32, b2: f32) -> (f32, f32) {
    let mean = |a, b| a / (a + b);
    let var = |a, b| a * b / (f32::powi(a + b, 2) * (a + b + 1.));

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
}

pub(crate) trait CoiPoint {
    fn new(id: usize, embedding: Embedding) -> Self;
    fn point(&self) -> &Embedding;
    fn set_point(&mut self, embedding: Embedding);
}

macro_rules! impl_coi_point {
    ($type:ty) => {
        impl CoiPoint for $type {
            fn new(id: usize, embedding: Embedding) -> Self {
                <$type>::new(id, embedding)
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
        self.positive.append(&mut other.positive);
        self.negative.append(&mut other.negative);
        // TODO drop dupe points, scheme for clashing ids
    }
}

impl Default for UserInterests {
    fn default() -> Self {
        UserInterests::new()
    }
}
