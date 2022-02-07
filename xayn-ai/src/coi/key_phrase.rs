use std::{borrow::Borrow, collections::BTreeSet, convert::identity, iter::once, time::Duration};

use derivative::Derivative;
use itertools::izip;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix, Ix2};
use serde::{Deserialize, Serialize};

use crate::{
    coi::{
        point::PositiveCoi,
        relevance::{Relevance, RelevanceMap},
        CoiError,
    },
    embedding::utils::{pairwise_cosine_similarity, ArcEmbedding, Embedding},
    error::Error,
    utils::system_time_now,
};

#[derive(Clone, Debug, Derivative, Deserialize, Serialize)]
#[derivative(Eq, Ord, PartialEq, PartialOrd)]
pub struct KeyPhrase {
    words: String,
    #[derivative(Ord = "ignore", PartialEq = "ignore", PartialOrd = "ignore")]
    point: ArcEmbedding,
}

impl KeyPhrase {
    pub(crate) fn new(
        words: impl Into<String>,
        point: impl Into<ArcEmbedding>,
    ) -> Result<Self, CoiError> {
        let words = words.into();
        let point = point.into();

        if words.is_empty() || point.is_empty() {
            return Err(CoiError::EmptyKeyPhrase);
        }
        if !point.iter().copied().all(f32::is_finite) {
            return Err(CoiError::NonFiniteKeyPhrase(point));
        }

        Ok(Self { words, point })
    }

    pub fn words(&self) -> &str {
        &self.words
    }

    pub fn point(&self) -> &ArcEmbedding {
        &self.point
    }
}

impl Borrow<String> for KeyPhrase {
    fn borrow(&self) -> &String {
        &self.words
    }
}

impl PartialEq<&str> for KeyPhrase {
    fn eq(&self, other: &&str) -> bool {
        self.words.eq(other)
    }
}

impl PositiveCoi {
    pub(crate) fn select_key_phrases(
        &self,
        relevances: &mut RelevanceMap,
        candidates: &[String],
        smbert: impl Fn(&str) -> Result<Embedding, Error>,
        max_key_phrases: usize,
        gamma: f32,
    ) {
        relevances.select_key_phrases(self, candidates, smbert, max_key_phrases, gamma);
    }
}

impl RelevanceMap {
    /// Selects the most relevant key phrases for the positive coi.
    ///
    /// The most relevant key phrases are selected from the set of key phrases of the coi and the
    /// candidates. The computed relevances are a relative score from the interval `[0, 1]`.
    ///
    /// The relevances in the maps are replaced by the key phrase relevances.
    fn select_key_phrases(
        &mut self,
        coi: &PositiveCoi,
        candidates: &[String],
        smbert: impl Fn(&str) -> Result<Embedding, Error>,
        max_key_phrases: usize,
        gamma: f32,
    ) {
        let key_phrases = self.remove(coi.id).unwrap_or_default();
        let key_phrases = unify(key_phrases, candidates, smbert);
        let (similarity, normalized) = similarities(&key_phrases, &coi.point);
        let selected = is_selected(normalized, max_key_phrases, gamma);
        for (relevance, key_phrase) in select(key_phrases, selected, similarity) {
            self.insert(coi.id, relevance, key_phrase);
        }
    }

    /// Selects the top key phrases from the positive cois, sorted in descending relevance.
    ///
    /// The selected key phrases and their relevances are removed from the maps.
    pub(super) fn select_top_key_phrases(
        &mut self,
        cois: &[PositiveCoi],
        top: usize,
        horizon: Duration,
        penalty: &[f32],
    ) -> Vec<KeyPhrase> {
        self.compute_relevances(cois, horizon, system_time_now());

        // TODO: refactor once pop_last() etc are stabilized for BTreeMap
        let mut relevances = self
            .filter(cois.iter().map(|coi| coi.id))
            .flat_map(|(coi_id, relevance, key_phrases)| {
                penalty.iter().zip(key_phrases.iter().rev()).map(move |(&penalty, key_phrase)| {
                    let penalized_relevance = Relevance::coi((f32::from(relevance) * penalty).max(f32::MIN).min(f32::MAX)).unwrap(/* finite by construction */);
                    (coi_id, relevance, penalized_relevance, key_phrase.clone())
                })
            }).collect::<Vec<_>>();
        relevances.sort_unstable_by(|(_, _, this, _), (_, _, other, _)| this.cmp(other).reverse());
        relevances
            .into_iter()
            .take(top)
            .map(|(coi_id, relevance, _, key_phrase)| {
                self.clean(coi_id, relevance, &key_phrase);
                key_phrase
            })
            .collect()
    }
}

/// Unifies the key phrases and candidates.
fn unify<F>(
    mut key_phrases: BTreeSet<KeyPhrase>,
    candidates: &[String],
    smbert: F,
) -> BTreeSet<KeyPhrase>
where
    F: Fn(&str) -> Result<Embedding, Error>,
{
    for candidate in candidates {
        if !key_phrases.contains(candidate) {
            if let Ok(Ok(candidate)) =
                smbert(candidate).map(|point| KeyPhrase::new(candidate, point))
            {
                key_phrases.insert(candidate);
            }
        }
    }

    key_phrases
}

/// Reduces the matrix along the axis while skipping the diagonal elements.
fn reduce_without_diag<S, F, G>(
    a: ArrayBase<S, Ix2>,
    axis: Axis,
    reduce: F,
    finalize: G,
) -> Array2<f32>
where
    S: Data<Elem = f32>,
    F: Fn(f32, f32) -> f32 + Copy,
    G: Fn(f32) -> f32 + Copy,
{
    a.lanes(axis)
        .into_iter()
        .enumerate()
        .map(|(i, lane)| {
            lane.iter()
                .enumerate()
                .filter_map(|(j, element)| (i != j).then(|| *element))
                .reduce(reduce)
                .map(finalize)
                .unwrap_or_default()
        })
        .collect::<Array1<_>>()
        .insert_axis(axis)
}

/// Gets the index of the maximum element.
fn argmax<I, F>(iter: I) -> Option<Ix>
where
    I: IntoIterator<Item = F>,
    F: Borrow<f32>,
{
    iter.into_iter()
        .enumerate()
        .reduce(|(arg, max), (index, element)| {
            if element.borrow() > max.borrow() {
                (index, element)
            } else {
                (arg, max)
            }
        })
        .map(|(arg, _)| arg)
}

/// Computes the pairwise similarity matrix and its normalization of the key phrases.
///
/// The matrices are of shape `(key_phrases_len, key_phrases_len + 1)` where the last column
/// holds the similarities between the key phrases and the coi point.
fn similarities(
    key_phrases: &BTreeSet<KeyPhrase>,
    coi_point: &Embedding,
) -> (Array2<f32>, Array2<f32>) {
    let len = key_phrases.len();
    let similarity = pairwise_cosine_similarity(
        key_phrases
            .iter()
            .map(|key_phrase| key_phrase.point().view())
            .chain(once(coi_point.view())),
    )
    .slice_move(s![..len, ..]);
    debug_assert!(similarity.iter().copied().all(f32::is_finite));

    let min = reduce_without_diag(similarity.view(), Axis(0), f32::min, identity);
    let max = reduce_without_diag(similarity.view(), Axis(0), f32::max, identity);
    let normalized = (&similarity - &min) / (max - min);
    let mean = reduce_without_diag(
        normalized.view(),
        Axis(0),
        |reduced, element| reduced + element,
        |reduced| reduced / (len - 1) as f32,
    );
    let std_dev = reduce_without_diag(
        &normalized - &mean,
        Axis(0),
        |reduced, element| reduced + element.powi(2),
        |reduced| (reduced / (len - 1) as f32).sqrt(),
    );
    let normalized = (normalized - mean) / std_dev + 0.5;
    let normalized = normalized
        .mapv_into(|normalized| normalized.is_finite().then(|| normalized).unwrap_or(0.5));
    debug_assert!(normalized.iter().copied().all(f32::is_finite));

    (similarity, normalized)
}

/// Determines which key phrases should be selected.
fn is_selected(normalized: Array2<f32>, max_key_phrases: usize, gamma: f32) -> Vec<bool> {
    let len = normalized.len_of(Axis(0));
    if len <= max_key_phrases {
        return vec![true; len];
    }

    let candidate =
        argmax(normalized.slice(s![.., -1])).unwrap(/* at least one key phrase is available */);
    let mut selected = vec![false; len];
    selected[candidate] = true;
    for _ in 0..max_key_phrases.min(len) - 1 {
        let candidate = argmax(selected.iter().zip(normalized.rows()).map(
            |(&is_selected, normalized)| {
                if is_selected {
                    f32::MIN
                } else {
                    let max = selected
                        .iter()
                        .zip(normalized)
                        .filter_map(|(is_selected, normalized)| {
                            is_selected.then(|| *normalized)
                        })
                        .reduce(f32::max)
                        .unwrap(/* at least one key phrase is selected */);
                    gamma * normalized.slice(s![-1]).into_scalar() - (1. - gamma) * max
                }
            },
        )).unwrap(/* at least one key phrase is available */);
        selected[candidate] = true;
    }

    selected
}

/// Selects the determined key phrases.
fn select(
    key_phrases: BTreeSet<KeyPhrase>,
    selected: Vec<bool>,
    similarity: Array2<f32>,
) -> impl Iterator<Item = (Relevance, KeyPhrase)> {
    let similarity = similarity.slice_move(s![.., -1]);
    let max = selected
        .iter()
        .zip(similarity.iter())
        .filter_map(|(is_selected, &similarity)| is_selected.then(|| similarity))
        .reduce(f32::max)
        .unwrap_or_default();
    izip!(selected, similarity, key_phrases)
        .filter_map(move |(is_selected, similarity, key_phrase)| {
            is_selected.then(|| {
                let relevance = Relevance::kp((similarity > 0.).then(|| (similarity / max).max(0.).min(1.)).unwrap_or_default()).unwrap(/* finite by construction */);
                (relevance, key_phrase)
            })
        })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use ndarray::arr1;

    use crate::{coi::utils::tests::create_pos_cois, ranker::Config};
    use test_utils::assert_approx_eq;

    use super::*;

    #[test]
    fn test_select_key_phrases_empty() {
        let mut relevances = RelevanceMap::default();
        let cois = create_pos_cois(&[[1., 0., 0.]]);
        let candidates = &[];
        let smbert = |_: &str| unreachable!();
        let config = Config::default();

        relevances.select_key_phrases(
            &cois[0],
            candidates,
            smbert,
            config.max_key_phrases(),
            config.gamma(),
        );
        assert!(relevances.cois_is_empty());
        assert!(relevances.relevances_is_empty());
    }

    #[test]
    fn test_select_key_phrases_no_candidates() {
        let cois = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = [
            KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[1., 1., 1.])).unwrap(),
        ];
        let mut relevances = RelevanceMap::kp([cois[0].id; 2], [0.; 2], key_phrases.to_vec());
        let candidates = &[];
        let smbert = |_: &str| unreachable!();
        let config = Config::default();

        relevances.select_key_phrases(
            &cois[0],
            candidates,
            smbert,
            config.max_key_phrases(),
            config.gamma(),
        );
        assert_eq!(relevances.cois_len(), cois.len());
        assert!(relevances[cois[0].id].is_kp());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.8164967, 1.]);
        assert_eq!(relevances.relevances_len(), key_phrases.len());
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(0))],
            key_phrases[1..],
        );
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(1))],
            key_phrases[..1],
        );
    }

    #[test]
    fn test_select_key_phrases_only_candidates() {
        let cois = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = [
            KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[1., 1., 1.])).unwrap(),
        ];
        let mut relevances = RelevanceMap::default();
        let candidates = key_phrases
            .iter()
            .map(|key_phrase| key_phrase.words().to_string())
            .collect::<Vec<_>>();
        let smbert = |words: &str| {
            key_phrases
                .iter()
                .find_map(|key_phrase| {
                    (key_phrase.words() == words).then(|| Ok(key_phrase.point().clone().into()))
                })
                .unwrap()
        };
        let config = Config::default();

        relevances.select_key_phrases(
            &cois[0],
            &candidates,
            smbert,
            config.max_key_phrases(),
            config.gamma(),
        );
        assert_eq!(relevances.cois_len(), cois.len());
        assert!(relevances[cois[0].id].is_kp());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.8164967, 1.]);
        assert_eq!(relevances.relevances_len(), key_phrases.len());
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(0))],
            key_phrases[1..],
        );
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(1))],
            key_phrases[..1],
        );
    }

    #[test]
    fn test_select_key_phrases_max() {
        let cois = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = [
            KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[2., 1., 1.])).unwrap(),
            KeyPhrase::new("test", arr1(&[1., 1., 1.])).unwrap(),
            KeyPhrase::new("words", arr1(&[2., 1., 0.])).unwrap(),
        ];
        let mut relevances = RelevanceMap::kp([cois[0].id; 2], [0.; 2], key_phrases[..2].to_vec());
        let candidates = key_phrases[2..]
            .iter()
            .map(|key_phrase| key_phrase.words().to_string())
            .collect::<Vec<_>>();
        let smbert = |words: &str| {
            key_phrases
                .iter()
                .find_map(|key_phrase| {
                    (key_phrase.words() == words).then(|| Ok(key_phrase.point().clone().into()))
                })
                .unwrap()
        };
        let config = Config::default();

        relevances.select_key_phrases(
            &cois[0],
            &candidates,
            smbert,
            config.max_key_phrases(),
            config.gamma(),
        );
        assert_eq!(relevances.cois_len(), cois.len());
        assert!(relevances[cois[0].id].is_kp());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.7905694, 0.91287094, 1.]);
        assert_eq!(relevances.relevances_len(), config.max_key_phrases());
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(0))],
            key_phrases[..1],
        );
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(1))],
            key_phrases[1..2],
        );
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(2))],
            key_phrases[3..],
        );
    }

    #[test]
    fn test_select_key_phrases_duplicate() {
        let cois = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = [
            KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[1., 1., 1.])).unwrap(),
        ];
        let mut relevances = RelevanceMap::kp([cois[0].id], [0.], key_phrases[..1].to_vec());
        let candidates = key_phrases[1..]
            .iter()
            .map(|key_phrase| key_phrase.words().to_string())
            .cycle()
            .take(2)
            .collect::<Vec<_>>();
        let smbert = |words: &str| {
            key_phrases
                .iter()
                .find_map(|key_phrase| {
                    (key_phrase.words() == words).then(|| Ok(key_phrase.point().clone().into()))
                })
                .unwrap()
        };
        let config = Config::default();

        relevances.select_key_phrases(
            &cois[0],
            &candidates,
            smbert,
            config.max_key_phrases(),
            config.gamma(),
        );
        assert_eq!(relevances.cois_len(), cois.len());
        assert!(relevances[cois[0].id].is_kp());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.8164967, 1.]);
        assert_eq!(relevances.relevances_len(), key_phrases.len());
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(0))],
            key_phrases[1..],
        );
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(1))],
            key_phrases[..1],
        );
    }

    #[test]
    fn test_select_key_phrases_orthogonal() {
        let cois = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = [
            KeyPhrase::new("key", arr1(&[0., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[0., 0., 1.])).unwrap(),
        ];
        let mut relevances = RelevanceMap::kp([cois[0].id], [0.], key_phrases[..1].to_vec());
        let candidates = key_phrases[1..]
            .iter()
            .map(|key_phrase| key_phrase.words().to_string())
            .collect::<Vec<_>>();
        let smbert = |words: &str| {
            key_phrases
                .iter()
                .find_map(|key_phrase| {
                    (key_phrase.words() == words).then(|| Ok(key_phrase.point().clone().into()))
                })
                .unwrap()
        };
        let config = Config::default();

        relevances.select_key_phrases(
            &cois[0],
            &candidates,
            smbert,
            config.max_key_phrases(),
            config.gamma(),
        );
        assert_eq!(relevances.cois_len(), cois.len());
        assert!(relevances[cois[0].id].is_kp());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.]);
        assert_eq!(relevances.relevances_len(), 1);
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(0))],
            key_phrases,
        );
    }

    #[test]
    fn test_select_key_phrases_positive_similarity() {
        let cois = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = [
            KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[1., 1., 1.])).unwrap(),
        ];
        let mut relevances = RelevanceMap::kp([cois[0].id], [0.], key_phrases[..1].to_vec());
        let candidates = key_phrases[1..]
            .iter()
            .map(|key_phrase| key_phrase.words().to_string())
            .collect::<Vec<_>>();
        let smbert = |words: &str| {
            key_phrases
                .iter()
                .find_map(|key_phrase| {
                    (key_phrase.words() == words).then(|| Ok(key_phrase.point().clone().into()))
                })
                .unwrap()
        };
        let config = Config::default();

        relevances.select_key_phrases(
            &cois[0],
            &candidates,
            smbert,
            config.max_key_phrases(),
            config.gamma(),
        );
        assert_eq!(relevances.cois_len(), cois.len());
        assert!(relevances[cois[0].id].is_kp());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.8164967, 1.]);
        assert_eq!(relevances.relevances_len(), key_phrases.len());
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(0))],
            key_phrases[1..],
        );
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(1))],
            key_phrases[..1],
        );
    }

    #[test]
    fn test_select_key_phrases_negative_similarity() {
        let cois = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = [
            KeyPhrase::new("key", arr1(&[-1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[-1., 1., 1.])).unwrap(),
        ];
        let mut relevances = RelevanceMap::kp([cois[0].id], [0.], key_phrases[..1].to_vec());
        let candidates = key_phrases[1..]
            .iter()
            .map(|key_phrase| key_phrase.words().to_string())
            .collect::<Vec<_>>();
        let smbert = |words: &str| {
            key_phrases
                .iter()
                .find_map(|key_phrase| {
                    (key_phrase.words() == words).then(|| Ok(key_phrase.point().clone().into()))
                })
                .unwrap()
        };
        let config = Config::default();

        relevances.select_key_phrases(
            &cois[0],
            &candidates,
            smbert,
            config.max_key_phrases(),
            config.gamma(),
        );
        assert_eq!(relevances.cois_len(), cois.len());
        assert!(relevances[cois[0].id].is_kp());
        assert_approx_eq!(f32, relevances[cois[0].id], [0.]);
        assert_eq!(relevances.relevances_len(), 1);
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(0))],
            key_phrases,
        );
    }

    #[test]
    fn test_select_key_phrases_mixed_similarity() {
        let cois = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = [
            KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[-1., 1., 1.])).unwrap(),
        ];
        let mut relevances = RelevanceMap::kp([cois[0].id], [0.], key_phrases[..1].to_vec());
        let candidates = key_phrases[1..]
            .iter()
            .map(|key_phrase| key_phrase.words().to_string())
            .collect::<Vec<_>>();
        let smbert = |words: &str| {
            key_phrases
                .iter()
                .find_map(|key_phrase| {
                    (key_phrase.words() == words).then(|| Ok(key_phrase.point().clone().into()))
                })
                .unwrap()
        };
        let config = Config::default();

        relevances.select_key_phrases(
            &cois[0],
            &candidates,
            smbert,
            config.max_key_phrases(),
            config.gamma(),
        );
        assert_eq!(relevances.cois_len(), cois.len());
        assert!(relevances[cois[0].id].is_kp());
        assert_approx_eq!(f32, relevances[cois[0].id], [0., 1.]);
        assert_eq!(relevances.relevances_len(), key_phrases.len());
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(0))],
            key_phrases[1..],
        );
        assert_eq!(
            relevances[(cois[0].id, relevances[cois[0].id].to_relevance(1))],
            key_phrases[..1],
        );
    }

    #[test]
    fn test_select_top_key_phrases_empty_cois() {
        let cois = create_pos_cois(&[] as &[[f32; 0]]);
        let mut relevances = RelevanceMap::default();
        let config = Config::default();

        let top_key_phrases = relevances.select_top_key_phrases(
            &cois,
            usize::MAX,
            config.horizon(),
            config.penalty(),
        );
        assert!(top_key_phrases.is_empty());
        assert!(relevances.cois_is_empty());
        assert!(relevances.relevances_is_empty());
    }

    #[test]
    fn test_select_top_key_phrases_empty_key_phrases() {
        let cois = create_pos_cois(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        let mut relevances = RelevanceMap::default();
        let config = Config::default();

        let top_key_phrases = relevances.select_top_key_phrases(
            &cois,
            usize::MAX,
            config.horizon(),
            config.penalty(),
        );
        assert!(top_key_phrases.is_empty());
        assert_eq!(relevances.cois_len(), cois.len());
        assert!(relevances[cois[0].id].is_coi());
        assert!(relevances[cois[1].id].is_coi());
        assert!(relevances[cois[2].id].is_coi());
        assert!(relevances.relevances_is_empty());
    }

    #[test]
    fn test_select_top_key_phrases_zero() {
        let cois = create_pos_cois(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        let key_phrases = [
            KeyPhrase::new("key", arr1(&[1., 1., 1.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[2., 1., 1.])).unwrap(),
            KeyPhrase::new("words", arr1(&[3., 1., 1.])).unwrap(),
        ];
        let mut relevances = RelevanceMap::kp(
            [cois[0].id, cois[1].id, cois[2].id],
            [0.; 3],
            key_phrases.to_vec(),
        );
        let config = Config::default();

        let top_key_phrases =
            relevances.select_top_key_phrases(&cois, 0, config.horizon(), config.penalty());
        assert!(top_key_phrases.is_empty());
        assert_eq!(relevances.cois_len(), cois.len());
        assert!(relevances[cois[0].id].is_coi());
        assert!(relevances[cois[1].id].is_coi());
        assert!(relevances[cois[2].id].is_coi());
        assert_eq!(relevances.relevances_len(), key_phrases.len());
    }

    #[test]
    fn test_select_top_key_phrases_all() {
        let mut cois = create_pos_cois(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        cois[0].log_time(Duration::from_secs(11));
        cois[1].log_time(Duration::from_secs(12));
        cois[2].log_time(Duration::from_secs(13));
        let key_phrases = [
            KeyPhrase::new("key", arr1(&[1., 1., 1.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[2., 1., 1.])).unwrap(),
            KeyPhrase::new("words", arr1(&[3., 1., 1.])).unwrap(),
            KeyPhrase::new("and", arr1(&[1., 4., 1.])).unwrap(),
            KeyPhrase::new("more", arr1(&[1., 5., 1.])).unwrap(),
            KeyPhrase::new("stuff", arr1(&[1., 6., 1.])).unwrap(),
            KeyPhrase::new("still", arr1(&[1., 1., 7.])).unwrap(),
            KeyPhrase::new("not", arr1(&[1., 1., 8.])).unwrap(),
            KeyPhrase::new("enough", arr1(&[1., 1., 9.])).unwrap(),
        ];
        let mut relevances = RelevanceMap::kp(
            [
                cois[0].id, cois[0].id, cois[0].id, cois[1].id, cois[1].id, cois[1].id, cois[2].id,
                cois[2].id, cois[2].id,
            ],
            [0.; 9],
            key_phrases.into(),
        );
        let config = Config::default();

        let top_key_phrases = relevances.select_top_key_phrases(
            &cois,
            usize::MAX,
            config.horizon(),
            config.penalty(),
        );
        assert_eq!(
            top_key_phrases,
            ["enough", "stuff", "words", "not", "more", "phrase", "still", "and", "key"],
        );
        assert!(relevances.cois_is_empty());
        assert!(relevances.relevances_is_empty());
    }
}
