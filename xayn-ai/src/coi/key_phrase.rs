use std::{
    borrow::{Borrow, Cow},
    collections::BTreeSet,
    convert::identity,
    iter::once,
};

use derivative::Derivative;
use lazy_static::lazy_static;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix, Ix2};
use serde::{Deserialize, Serialize};

use crate::{
    coi::{
        point::{CoiPoint, NegativeCoi, PositiveCoi},
        CoiError,
    },
    embedding::utils::{pairwise_cosine_similarity, ArcEmbedding, Embedding},
    error::Error,
};

#[derive(Clone, Debug, Derivative, Deserialize, Serialize)]
#[derivative(Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct KeyPhrase {
    words: String,
    #[derivative(Ord = "ignore", PartialEq = "ignore", PartialOrd = "ignore")]
    point: ArcEmbedding,
    #[derivative(Ord = "ignore", PartialEq = "ignore", PartialOrd = "ignore")]
    relevance: f32,
}

lazy_static! {
    // TODO: temporary workaround, remove once positive and negative cois have been split properly
    static ref EMPTY_KEY_PHRASES: BTreeSet<KeyPhrase> = BTreeSet::new();
}

impl KeyPhrase {
    pub(crate) fn new(
        words: impl Into<String>,
        point: impl Into<ArcEmbedding>,
    ) -> Result<Self, CoiError> {
        let words = words.into();
        let point = point.into();
        let relevance = 0.;

        if words.is_empty() || point.is_empty() {
            return Err(CoiError::EmptyKeyPhrase);
        }
        if !point.iter().copied().all(f32::is_finite) {
            return Err(CoiError::NonFiniteKeyPhrase(point));
        }

        Ok(Self {
            words,
            point,
            relevance,
        })
    }

    pub(crate) fn with_relevance(self, relevance: f32) -> Result<Self, CoiError> {
        if (0. ..=1.).contains(&relevance) {
            Ok(Self { relevance, ..self })
        } else {
            Err(CoiError::NonNormalizedKeyPhrase(relevance))
        }
    }

    #[cfg(test)]
    pub(crate) fn words(&self) -> &str {
        &self.words
    }

    pub(crate) fn point(&self) -> &ArcEmbedding {
        &self.point
    }

    #[cfg(test)]
    pub(crate) fn relevance(&self) -> f32 {
        self.relevance
    }
}

impl Borrow<String> for KeyPhrase {
    fn borrow(&self) -> &String {
        &self.words
    }
}

impl Borrow<str> for KeyPhrase {
    fn borrow(&self) -> &str {
        self.words.as_str()
    }
}

pub(crate) trait CoiPointKeyPhrases {
    fn key_phrases(&self) -> &BTreeSet<KeyPhrase>;

    fn set_key_phrases(&mut self, key_phrases: BTreeSet<KeyPhrase>);
}

impl CoiPointKeyPhrases for PositiveCoi {
    fn key_phrases(&self) -> &BTreeSet<KeyPhrase> {
        &self.key_phrases
    }

    fn set_key_phrases(&mut self, key_phrases: BTreeSet<KeyPhrase>) {
        self.key_phrases = key_phrases;
    }
}

impl CoiPointKeyPhrases for NegativeCoi {
    fn key_phrases(&self) -> &BTreeSet<KeyPhrase> {
        &EMPTY_KEY_PHRASES
    }

    fn set_key_phrases(&mut self, _key_phrases: BTreeSet<KeyPhrase>) {}
}

/// Filters the unique candidates wrt the existing key phrases.
fn unique_candidates<CP, F>(coi: &CP, candidates: &[String], smbert: F) -> BTreeSet<KeyPhrase>
where
    CP: CoiPoint + CoiPointKeyPhrases,
    F: Fn(&str) -> Result<Embedding, Error>,
{
    candidates
        .iter()
        .filter_map(|words| {
            (!coi.key_phrases().contains(words))
                .then(|| {
                    smbert(words)
                        .ok()
                        .and_then(|point| KeyPhrase::new(words, point).ok())
                })
                .flatten()
        })
        .collect()
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

/// Computes the pairwise similarity and their normalizations of the key phrases.
///
/// The matrices are of shape `(key_phrases_len + candidates_len, key_phrases_len +
/// candidates_len + 1)` with the following blockwise layout:
/// ```text
/// [[sim(kp, kp),   sim(kp, cand),   sim(kp, coi)  ],
///  [sim(cand, kp), sim(cand, cand), sim(cand, coi)]]
/// ```
fn similarities<CP>(coi: &CP, candidates: &BTreeSet<KeyPhrase>) -> (Array2<f32>, Array2<f32>)
where
    CP: CoiPoint + CoiPointKeyPhrases,
{
    let len = coi.key_phrases().len() + candidates.len();
    let similarity = pairwise_cosine_similarity(
        coi.key_phrases()
            .iter()
            .chain(candidates.iter())
            .map(|key_phrase| key_phrase.point().view())
            .chain(once(coi.point().view())),
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
fn is_selected<S>(normalized: ArrayBase<S, Ix2>, max_key_phrases: usize, gamma: f32) -> Vec<bool>
where
    S: Data<Elem = f32>,
{
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
fn select<CP, S>(
    coi: &mut CP,
    candidates: BTreeSet<KeyPhrase>,
    selected: Vec<bool>,
    similarity: ArrayBase<S, Ix2>,
) where
    CP: CoiPoint + CoiPointKeyPhrases,
    S: Data<Elem = f32>,
{
    let relevance = selected
        .iter()
        .zip(similarity.slice(s![.., -1]))
        .filter_map(|(is_selected, similarity)| is_selected.then(|| similarity))
        .copied();
    let max = relevance.clone().reduce(f32::max).unwrap_or_default();
    let relevance = relevance.map(|relevance| {
        (relevance > 0.)
            .then(|| (relevance / max).clamp(0., 1.))
            .unwrap_or_default()
    });

    let key_phrases = selected
        .iter()
        .zip(
            coi.key_phrases()
                .iter()
                .map(Cow::Borrowed)
                .chain(candidates.into_iter().map(Cow::Owned)),
        )
        .filter_map(|(is_selected, key_phrase)| is_selected.then(|| key_phrase.into_owned()))
        .zip(relevance)
        .filter_map(|(key_phrase, relevance)| key_phrase.with_relevance(relevance).ok())
        .collect();
    coi.set_key_phrases(key_phrases);
}

/// Selects the most relevant key phrases for the coi.
///
/// The most relevant key phrases are selected from the set of key phrases of the coi and the
/// candidates. The computed relevances are a relative score from the interval `[0, 1]`.
pub(super) fn select_key_phrases<CP, F>(
    coi: &mut CP,
    candidates: &[String],
    // TODO: make SMBert available to CoiSystem and remove this argument
    smbert: F,
    max_key_phrases: usize,
    gamma: f32,
) where
    CP: CoiPoint + CoiPointKeyPhrases,
    F: Fn(&str) -> Result<Embedding, Error>,
{
    let candidates = unique_candidates(coi, candidates, smbert);
    let (similarity, normalized) = similarities(coi, &candidates);
    let selected = is_selected(normalized, max_key_phrases, gamma);
    select(coi, candidates, selected, similarity);
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use crate::coi::utils::tests::create_pos_cois;
    use test_utils::assert_approx_eq;

    use super::*;

    #[test]
    fn test_select_key_phrases_empty() {
        let mut coi = create_pos_cois(&[[1., 0., 0.]]);
        let candidates = &[];
        let smbert = |_: &str| unreachable!();
        select_key_phrases(&mut coi[0], candidates, smbert, 3, 0.9);
        assert!(coi[0].key_phrases().is_empty());
    }

    #[test]
    fn test_select_key_phrases_one() {
        let mut coi = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases =
            IntoIterator::into_iter([KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap()])
                .collect::<BTreeSet<_>>();
        coi[0].set_key_phrases(key_phrases.clone());
        let candidates = &[];
        let smbert = |_: &str| unreachable!();
        select_key_phrases(&mut coi[0], candidates, smbert, 3, 0.9);
        assert_eq!(coi[0].key_phrases(), &key_phrases);
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("key").unwrap().relevance(),
            1.,
        );
    }

    #[test]
    fn test_select_key_phrases_no_candidates() {
        let mut coi = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = IntoIterator::into_iter([
            KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[1., 1., 1.])).unwrap(),
        ])
        .collect::<BTreeSet<_>>();
        coi[0].set_key_phrases(key_phrases.clone());
        let candidates = &[];
        let smbert = |_: &str| unreachable!();
        select_key_phrases(&mut coi[0], candidates, smbert, 3, 0.9);
        assert_eq!(coi[0].key_phrases(), &key_phrases);
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("key").unwrap().relevance(),
            1.,
        );
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("phrase").unwrap().relevance(),
            0.8164967,
        );
    }

    #[test]
    fn test_select_key_phrases_only_candidates() {
        let mut coi = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = IntoIterator::into_iter([
            KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[1., 1., 1.])).unwrap(),
        ])
        .collect::<BTreeSet<_>>();
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
        select_key_phrases(&mut coi[0], &candidates, smbert, 3, 0.9);
        assert_eq!(coi[0].key_phrases(), &key_phrases);
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("key").unwrap().relevance(),
            1.,
        );
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("phrase").unwrap().relevance(),
            0.8164967,
        );
    }

    #[test]
    fn test_select_key_phrases_max() {
        let mut coi = create_pos_cois(&[[1., 0., 0.]]);
        let mut key_phrases = IntoIterator::into_iter([
            KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[2., 1., 1.])).unwrap(),
            KeyPhrase::new("test", arr1(&[1., 1., 1.])).unwrap(),
            KeyPhrase::new("words", arr1(&[2., 1., 0.])).unwrap(),
        ])
        .collect::<BTreeSet<_>>();
        coi[0].set_key_phrases(key_phrases.iter().cloned().take(2).collect());
        let candidates = key_phrases
            .iter()
            .skip(2)
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
        select_key_phrases(&mut coi[0], &candidates, smbert, 3, 0.9);
        assert!(key_phrases.remove("test"));
        assert_eq!(coi[0].key_phrases(), &key_phrases);
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("key").unwrap().relevance(),
            0.7905694,
        );
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("phrase").unwrap().relevance(),
            0.91287094,
        );
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("words").unwrap().relevance(),
            1.,
        );
    }

    #[test]
    fn test_select_key_phrases_duplicate() {
        let mut coi = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = IntoIterator::into_iter([
            KeyPhrase::new("key", arr1(&[1., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[1., 1., 1.])).unwrap(),
        ])
        .collect::<BTreeSet<_>>();
        coi[0].set_key_phrases(key_phrases.iter().cloned().take(1).collect());
        let candidates = key_phrases
            .iter()
            .skip(1)
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
        select_key_phrases(&mut coi[0], &candidates, smbert, 3, 0.9);
        assert_eq!(coi[0].key_phrases(), &key_phrases);
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("key").unwrap().relevance(),
            1.,
        );
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("phrase").unwrap().relevance(),
            0.8164967,
        );
    }

    #[test]
    fn test_select_key_phrases_orthogonal() {
        let mut coi = create_pos_cois(&[[1., 0., 0.]]);
        let key_phrases = IntoIterator::into_iter([
            KeyPhrase::new("key", arr1(&[0., 1., 0.])).unwrap(),
            KeyPhrase::new("phrase", arr1(&[0., 0., 1.])).unwrap(),
        ])
        .collect::<BTreeSet<_>>();
        coi[0].set_key_phrases(key_phrases.iter().cloned().take(1).collect());
        let candidates = key_phrases
            .iter()
            .skip(1)
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
        select_key_phrases(&mut coi[0], &candidates, smbert, 3, 0.9);
        assert_eq!(coi[0].key_phrases(), &key_phrases);
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("key").unwrap().relevance(),
            0.,
        );
        assert_approx_eq!(
            f32,
            coi[0].key_phrases().get("phrase").unwrap().relevance(),
            0.,
        );
    }
}
