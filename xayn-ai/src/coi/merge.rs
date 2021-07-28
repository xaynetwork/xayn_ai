use std::cmp::Ordering;

use itertools::Itertools;

use crate::{
    data::{CoiPoint, NegativeCoi, PositiveCoi},
    embedding::utils::{l2_distance, mean},
    utils::nan_safe_f32_cmp_high,
    CoiId,
};

const MERGE_THRESHOLD_DIST: f32 = 4.5;

impl PositiveCoi {
    pub fn merge(self, other: Self, id: CoiId) -> Self {
        let point = mean(&self.point, &other.point);
        let (alpha, beta) = merge_params(self.alpha, self.beta, other.alpha, other.beta);
        Self {
            id,
            point,
            alpha,
            beta,
        }
    }
}

impl NegativeCoi {
    pub fn merge(self, other: Self, id: CoiId) -> Self {
        let point = mean(&self.point, &other.point);
        Self { id, point }
    }
}

/// A pair of CoIs.
#[derive(Clone)]
struct CoiPair<C>(C, C);

impl<C: CoiPoint> CoiPair<C> {
    /// Creates a new CoI pair.
    ///
    /// The CoI with the smaller id (or `coi0` if the ids are equal) occupies position 0.
    fn new(coi0: C, coi1: C) -> Self {
        if coi0.id() <= coi1.id() {
            Self(coi0, coi1)
        } else {
            Self(coi1, coi0)
        }
    }

    /// Merges the CoI pair, assigning it the smaller of the two ids.
    fn merge_min(self) -> C {
        let min_id = self.0.id();
        self.0.merge(self.1, min_id)
    }

    /// True iff either CoI has the given id.
    fn contains(&self, id: CoiId) -> bool {
        self.0.id() == id || self.1.id() == id
    }

    /// True iff either CoI is one of the given pair.
    fn contains_any(&self, other: &Self) -> bool {
        self.contains(other.0.id()) || self.contains(other.1.id())
    }

    /// Returns the `Ordering` between `self` and `other`.
    ///
    /// Produces a lexicographic ordering based on the component CoI ids.
    fn compare(&self, other: &Self) -> Ordering {
        match self.0.id().cmp(&other.0.id()) {
            Ordering::Equal => self.1.id().cmp(&other.1.id()),
            lt_or_gt => lt_or_gt,
        }
    }
}

/// A `Coiple` is a pair of CoIs and the distance between them.
#[derive(Clone)]
struct Coiple<C> {
    cois: CoiPair<C>,
    dist: f32,
}

impl<C: CoiPoint> Coiple<C> {
    /// Creates a new coiple.
    fn new(coi1: C, coi2: C, dist: f32) -> Self {
        let cois = CoiPair::new(coi1, coi2);
        Self { cois, dist }
    }

    /// Returns the `Ordering` between `self` and `other`.
    ///
    /// Produces a lexicographic ordering based on the distance, then the CoI pair.
    fn compare(&self, other: &Self) -> Ordering {
        match nan_safe_f32_cmp_high(&self.dist, &other.dist) {
            Ordering::Equal => self.cois.compare(&other.cois),
            lt_or_gt => lt_or_gt,
        }
    }
}

/// Computes the l2 distance between two CoI points.
fn dist<C: CoiPoint>(coi1: &C, coi2: &C) -> f32 {
    l2_distance(coi1.point(), coi2.point())
}

/// Reduces the given collection of CoIs by successively merging the pair in closest
/// proximity to each other.
///
/// <https://xainag.atlassian.net/wiki/spaces/XAY/pages/2029944833/CoI+synchronisation>
/// outlines the core of the algorithm.
pub(crate) fn reduce_cois<C>(cois: &mut Vec<C>)
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
            (dist < MERGE_THRESHOLD_DIST).then(|| Coiple::new(coi1.clone(), coi2.clone(), dist))
        })
        .collect_vec();

    while !coiples.is_empty() {
        // find the minimum i.e. closest pair of cois
        let min_pair = coiples
            .iter()
            .cloned()
            .min_by(|cpl1, cpl2| cpl1.compare(&cpl2))
            .unwrap() // safe: nonempty coiples
            .cois;

        // discard pair from collections
        cois.retain(|coi| !min_pair.contains(coi.id()));
        coiples.retain(|cpl| !cpl.cois.contains_any(&min_pair));

        let merged_coi = min_pair.merge_min();

        // record close coiples featuring the merged coi
        let mut new_coiples = cois
            .iter()
            .filter_map(|coi| {
                let dist = dist(&merged_coi, coi);
                (dist < MERGE_THRESHOLD_DIST)
                    .then(|| Coiple::new(merged_coi.clone(), coi.clone(), dist))
            })
            .collect_vec();
        coiples.append(&mut new_coiples);

        cois.push(merged_coi);
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
