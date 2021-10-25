use std::cmp::Ordering;

use itertools::Itertools;

use crate::{
    coi::{
        point::{CoiPoint, CoiPointMerge, NegativeCoi, PositiveCoi},
        CoiId,
    },
    embedding::utils::{l2_distance, mean},
    utils::nan_safe_f32_cmp_high,
};

const MERGE_THRESHOLD_DIST: f32 = 4.5;

impl PositiveCoi {
    pub fn merge(self, other: Self, id: CoiId) -> Self {
        let point = mean(&self.point, &other.point);
        Self { id, point }
    }
}

impl NegativeCoi {
    pub fn merge(self, other: Self, id: CoiId) -> Self {
        let point = mean(&self.point, &other.point);
        Self { id, point }
    }
}

/// A pair of CoIs.
#[cfg_attr(test, derive(Debug, PartialEq))]
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
    fn merge_min(self) -> C
    where
        C: CoiPointMerge,
    {
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
    C: CoiPoint + CoiPointMerge + Clone,
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
            .min_by(|cpl1, cpl2| cpl1.compare(cpl2))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        coi::CoiId,
        tests::{mocked_smbert_system, pos_cois_from_words, pos_cois_from_words_with_ids},
    };

    impl PositiveCoi {
        fn from_word(word: &str, id: usize) -> Self {
            pos_cois_from_words_with_ids(&[word], mocked_smbert_system(), id)
                .pop()
                .unwrap()
        }

        fn mock() -> Self {
            Self::from_word("", 0)
        }
    }

    #[test]
    fn test_coipair_new_distinct() {
        let cois = pos_cois_from_words(&["a", "b"], mocked_smbert_system());
        assert_eq!(cois.len(), 2);

        let pair = CoiPair::new(cois[0].clone(), cois[1].clone());
        assert_eq!(pair.0, cois[0]);
        assert_eq!(pair.1, cois[1]);

        let pair_rev = CoiPair::new(cois[1].clone(), cois[0].clone());
        assert_eq!(pair_rev, pair);
    }

    #[test]
    fn test_coipair_new_dupes() {
        let coi_a = PositiveCoi::from_word("a", 0);
        let coi_b = PositiveCoi::from_word("b", 0);

        let pair = CoiPair::new(coi_a.clone(), coi_b.clone());
        assert_eq!(pair.0, coi_a);
        assert_eq!(pair.1, coi_b);

        let pair_rev = CoiPair::new(coi_b.clone(), coi_a.clone());
        assert_eq!(pair_rev.0, coi_b);
        assert_eq!(pair_rev.1, coi_a);
    }

    #[test]
    fn test_coipair_contains() {
        let coi_a = PositiveCoi::from_word("a", 0);
        let coi_b = PositiveCoi::from_word("b", 1);
        let coi_c = PositiveCoi::from_word("c", 2);
        let coi_d = PositiveCoi::from_word("d", 3);

        let pair_ab = CoiPair::new(coi_a.clone(), coi_b.clone());
        let pair_ac = CoiPair::new(coi_a, coi_c.clone());
        let pair_bc = CoiPair::new(coi_b, coi_c.clone());
        let pair_cd = CoiPair::new(coi_c, coi_d);

        assert!(pair_ab.contains(CoiId::mocked(0)));
        assert!(pair_ab.contains(CoiId::mocked(1)));
        assert!(!pair_ab.contains(CoiId::mocked(2)));
        assert!(!pair_ab.contains(CoiId::mocked(3)));

        assert!(pair_ab.contains_any(&pair_ab));
        assert!(pair_ab.contains_any(&pair_ac));
        assert!(pair_ab.contains_any(&pair_bc));
        assert!(!pair_ab.contains_any(&pair_cd));
    }

    #[test]
    fn test_merge() {
        let coi_a = PositiveCoi::from_word("a", 0);
        let coi_z = PositiveCoi::from_word("z", 1);
        let pair_az = CoiPair::new(coi_a.clone(), coi_z.clone());
        let merged = pair_az.merge_min();
        assert_eq!(merged.id, coi_a.id);

        let dist_az = dist(&coi_a, &coi_z);
        assert!(dist(&coi_a, &merged) < dist_az);
        assert!(dist(&merged, &coi_z) < dist_az);
    }

    #[test]
    fn test_coipair_compare() {
        let coi_a = PositiveCoi::from_word("a", 0);
        let coi_b = PositiveCoi::from_word("b", 1);
        let coi_c = PositiveCoi::from_word("c", 2);
        let coi_d = PositiveCoi::from_word("d", 3);

        let pair_ac = CoiPair::new(coi_a, coi_c.clone());
        let pair_bc = CoiPair::new(coi_b.clone(), coi_c.clone());
        let pair_cd = CoiPair::new(coi_c, coi_d.clone());
        let pair_bd = CoiPair::new(coi_b, coi_d);

        assert_eq!(pair_bc.compare(&pair_bc), Ordering::Equal);
        assert_eq!(pair_bc.compare(&pair_ac), Ordering::Greater);
        assert_eq!(pair_bc.compare(&pair_cd), Ordering::Less);
        assert_eq!(pair_bc.compare(&pair_bd), Ordering::Less);
    }

    #[test]
    fn test_coiple_compare() {
        let coi_a = PositiveCoi::from_word("a", 0);
        let coi_b = PositiveCoi::from_word("b", 1);
        let coi_c = PositiveCoi::from_word("c", 2);
        let coi_d = PositiveCoi::from_word("d", 3);
        let coiple_bc = Coiple::new(coi_b, coi_c, 5.);

        assert_eq!(coiple_bc.compare(&coiple_bc), Ordering::Equal);
        let close = Coiple::new(PositiveCoi::mock(), PositiveCoi::mock(), 1.);
        assert_eq!(coiple_bc.compare(&close), Ordering::Greater);
        let far = Coiple::new(PositiveCoi::mock(), PositiveCoi::mock(), 10.);
        assert_eq!(coiple_bc.compare(&far), Ordering::Less);

        let coiple_aa = Coiple::new(coi_a.clone(), coi_a, 5.);
        let coiple_dd = Coiple::new(coi_d.clone(), coi_d, 5.);
        assert_eq!(coiple_bc.compare(&coiple_aa), Ordering::Greater);
        assert_eq!(coiple_bc.compare(&coiple_dd), Ordering::Less);
    }

    #[test]
    fn test_reduce_empty() {
        let mut empty: Vec<PositiveCoi> = vec![];
        reduce_cois(&mut empty);
        assert!(empty.is_empty())
    }

    #[test]
    fn test_reduce_distant() {
        let coi_a = PositiveCoi::from_word("a", 0);
        let coi_m = PositiveCoi::from_word("m", 1);
        let coi_z = PositiveCoi::from_word("z", 2);
        let mut cois = vec![coi_a, coi_m, coi_z];
        let cois_before = cois.clone();

        reduce_cois(&mut cois);
        assert_eq!(cois, cois_before);
    }

    #[test]
    fn test_reduce_coiples_idempotent() {
        let coi_a = PositiveCoi::from_word("a", 0);
        let coi_k = PositiveCoi::from_word("k", 1);
        let coi_m = PositiveCoi::from_word("m", 2);
        let coi_n = PositiveCoi::from_word("n", 3);
        let coi_z = PositiveCoi::from_word("z", 4);
        let mut cois = vec![
            coi_a.clone(),
            coi_k.clone(),
            coi_m.clone(),
            coi_n.clone(),
            coi_z.clone(),
        ];

        reduce_cois(&mut cois);
        assert_eq!(cois.len(), 3);
        assert!(cois.contains(&coi_a));
        assert!(cois.contains(&coi_z));

        let coi_mn = CoiPair::new(coi_m, coi_n).merge_min();
        let coi_k_mn = CoiPair::new(coi_k, coi_mn).merge_min();
        assert!(cois.contains(&coi_k_mn));

        let cois_after = cois.clone();
        reduce_cois(&mut cois);
        assert_eq!(cois, cois_after);
    }
}
