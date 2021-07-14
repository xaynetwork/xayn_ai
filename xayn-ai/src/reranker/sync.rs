use crate::{
    data::{CoiPoint, PositiveCoi, UserInterests},
    embedding::utils::l2_distance,
    error::Error,
    utils::nan_safe_f32_cmp_high,
    CoiId,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

const SOCIAL_DIST: f32 = 8.0;

/// Synchronizable data of the reranker.
#[cfg_attr(test, derive(Clone, PartialEq, Debug))]
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct SyncData {
    pub(crate) user_interests: UserInterests,
}

impl SyncData {
    /// Deserializes a `SyncData` from `bytes`.
    pub(crate) fn deserialize(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }

        Ok(bincode::deserialize(&bytes)?)
    }

    /// Serializes a `SyncData` to a byte representation.
    pub(crate) fn serialize(&self) -> Result<Vec<u8>, Error> {
        Ok(bincode::serialize(self)?)
    }

    /// Merge another `SyncData` into the current one.
    pub(crate) fn merge(&mut self, other: SyncData) {
        let Self { user_interests } = other;
        self.user_interests.append(user_interests);

        let cois_iter = self.user_interests.positive.iter();
        let mut coiples = cois_iter
            .clone()
            .cartesian_product(cois_iter)
            .filter_map(|(coi1, coi2)| {
                (coi1.id < coi2.id).then(|| Coiple::new(coi1, coi2, dist(coi1, coi2)))
            })
            .filter(|coiple| coiple.dist < SOCIAL_DIST)
            .collect_vec();

        while !coiples.is_empty() {
            let min_coiple = coiples
                .iter()
                .cloned()
                .min_by(|cpl1, cpl2| nan_safe_f32_cmp_high(&cpl1.dist, &cpl2.dist))
                .unwrap(); // safe: nonempty coiples

            let merged_coi = min_coiple.cois.merge();

            self.user_interests
                .positive
                .retain(|coi| coi.id != min_coiple.cois.0.id || coi.id != min_coiple.cois.1.id);

            coiples.retain(|cpl| {
                !cpl.contains(min_coiple.cois.0.id) && !cpl.contains(min_coiple.cois.1.id)
            });

            let mut new_coiples = self
                .user_interests
                .positive
                .iter()
                .filter_map(|coi| {
                    let dist = dist(&merged_coi, coi);
                    (dist < SOCIAL_DIST).then(|| Coiple::new(&merged_coi, coi, dist))
                })
                .collect_vec();
            coiples.append(&mut new_coiples);

            self.user_interests.positive.push(merged_coi);
        }
    }
}

#[derive(Clone)]
struct CoiPair(PositiveCoi, PositiveCoi); // TODO impl Eq

impl CoiPair {
    /// Merges the CoI pair, assigning it the smaller of the two ids.
    fn merge(&self) -> PositiveCoi {
        let min_id = self.0.id.0.min(self.1.id.0);
        self.0.from_merge(&self.1, min_id)
    }
}

/// A `Coiple` is a pair of CoIs and the distance between them.
#[derive(Clone)]
struct Coiple {
    cois: CoiPair,
    dist: f32,
}

impl Coiple {
    fn new(coi1: &PositiveCoi, coi2: &PositiveCoi, dist: f32) -> Self {
        let cois = CoiPair(coi1.clone(), coi2.clone());
        Self { cois, dist }
    }

    fn contains(&self, id: CoiId) -> bool {
        self.cois.0.id == id || self.cois.1.id == id
    }
}

/// Computes the l2 distance between two CoI points.
fn dist<C>(coi1: &C, coi2: &C) -> f32
where
    C: CoiPoint,
{
    l2_distance(coi1.point(), coi2.point())
}
