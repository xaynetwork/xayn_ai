use crate::{
    data::{CoiPoint, UserInterests},
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

    /// Synchronizes with another `SyncData`.
    ///
    /// <https://xainag.atlassian.net/wiki/spaces/XAY/pages/2029944833/CoI+synchronisation>
    /// outlines the algorithm.
    pub(crate) fn synchronize(&mut self, other: SyncData) {
        let Self { user_interests } = other;
        self.user_interests.append(user_interests);

        reduce_cois(&mut self.user_interests.positive);
        reduce_cois(&mut self.user_interests.negative);
    }
}

/// A pair of CoIs.
#[derive(Clone)]
struct CoiPair<C>(C, C);

impl<C> CoiPair<C>
where
    C: CoiPoint,
{
    /// Merges the CoI pair, assigning it the smaller of the two ids.
    fn merge(&self) -> C {
        let CoiId(id1) = self.0.id();
        let CoiId(id2) = self.1.id();
        self.0.merge(&self.1, id1.min(id2))
    }

    /// True iff either CoI has the given id.
    fn contains(&self, id: CoiId) -> bool {
        self.0.id() == id || self.1.id() == id
    }

    /// True iff either CoI is one of the given pair.
    fn contains_any(&self, other: &Self) -> bool {
        self.contains(other.0.id()) || self.contains(other.1.id())
    }
}

/// A `Coiple` is a pair of CoIs and the distance between them.
#[derive(Clone)]
struct Coiple<C> {
    cois: CoiPair<C>,
    dist: f32,
}

impl<C> Coiple<C> {
    fn new(coi1: C, coi2: C, dist: f32) -> Self {
        let cois = CoiPair(coi1, coi2);
        Self { cois, dist }
    }
}

/// Computes the l2 distance between two CoI points.
fn dist<C>(coi1: &C, coi2: &C) -> f32
where
    C: CoiPoint,
{
    l2_distance(coi1.point(), coi2.point())
}

/// Reduces the given collection of CoIs by successively merging the pair in closest
/// proximity to each other.
fn reduce_cois<C>(cois: &mut Vec<C>)
where
    C: CoiPoint + Clone,
{
    // initialize collection of close coiples
    let cois_iter = cois.iter();
    let mut coiples = cois_iter
        .clone()
        .cartesian_product(cois_iter)
        .filter(|(coi1, coi2)| coi1.id() < coi2.id())
        .filter_map(|(coi1, coi2)| {
            let dist = dist(coi1, coi2);
            (dist < SOCIAL_DIST).then(|| Coiple::new(coi1.clone(), coi2.clone(), dist))
        })
        .collect_vec();

    while !coiples.is_empty() {
        // find the minimum i.e. closest coiple
        let min_coiple = coiples
            .iter()
            .cloned()
            .min_by(|cpl1, cpl2| nan_safe_f32_cmp_high(&cpl1.dist, &cpl2.dist))
            .unwrap(); // safe: nonempty coiples

        let merged_coi = min_coiple.cois.merge();

        // discard component cois
        cois.retain(|coi| !min_coiple.cois.contains(coi.id()));
        coiples.retain(|cpl| !cpl.cois.contains_any(&min_coiple.cois));

        // record close coiples featuring the merged coi
        let mut new_coiples = cois
            .iter()
            .filter_map(|coi| {
                let dist = dist(&merged_coi, coi);
                (dist < SOCIAL_DIST).then(|| Coiple::new(merged_coi.clone(), coi.clone(), dist))
            })
            .collect_vec();
        coiples.append(&mut new_coiples);

        cois.push(merged_coi);
    }
}
