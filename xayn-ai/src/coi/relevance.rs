use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
};

use derive_more::Into;

use crate::coi::{key_phrase::KeyPhrase, CoiError, CoiId};

/// A finite relevance score.
#[derive(Clone, Copy, Debug, Default, Into, PartialEq, PartialOrd)]
pub(crate) struct Relevance(f32);

impl Relevance {
    /// Creates a relevance score.
    ///
    /// # Errors
    /// Fails if the relevance isn't finite.
    pub fn new(relevance: f32) -> Result<Self, CoiError> {
        if relevance.is_finite() {
            Ok(Relevance(relevance))
        } else {
            Err(CoiError::NonFiniteRelevance)
        }
    }
}

impl PartialEq<f32> for Relevance {
    fn eq(&self, other: &f32) -> bool {
        self.0.eq(other)
    }
}

impl Eq for Relevance {
    // never nan by construction
}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for Relevance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap(/* never nan by construction */)
    }
}

/// Sorted maps from cois to relevances to key phrases.
#[derive(Debug, Default)]
pub(crate) struct Relevances {
    coi_to_relevance: HashMap<CoiId, BTreeSet<Relevance>>,
    relevance_to_key_phrase: BTreeMap<(Relevance, CoiId), Vec<KeyPhrase>>,
}

impl Relevances {
    /// Iterates over all tuples with matching ids in ascending relevance.
    pub(super) fn filter(
        &self,
        coi_ids: impl Clone + Iterator<Item = CoiId>,
    ) -> impl Iterator<Item = (CoiId, Relevance, &[KeyPhrase])> + DoubleEndedIterator {
        self.relevance_to_key_phrase.iter().filter_map(
            move |(&(relevance, coi_id), key_phrases)| {
                coi_ids
                    .clone()
                    .any(|id| id == coi_id)
                    .then(move || (coi_id, relevance, key_phrases.as_slice()))
            },
        )
    }

    /// Inserts the tuple.
    pub(super) fn insert(&mut self, coi_id: CoiId, relevance: Relevance, key_phrase: KeyPhrase) {
        self.coi_to_relevance
            .entry(coi_id)
            .or_default()
            .insert(relevance);
        self.relevance_to_key_phrase
            .entry((relevance, coi_id))
            .or_default()
            .push(key_phrase);
    }

    /// Removes all tuples with a matching id.
    pub(super) fn remove(&mut self, coi_id: CoiId) -> Option<BTreeSet<KeyPhrase>> {
        self.coi_to_relevance
            .remove(&coi_id)
            .map(|relevances| {
                let key_phrases = relevances
                    .into_iter()
                    .map(|relevance| {
                        self.relevance_to_key_phrase
                            .remove(&(relevance, coi_id))
                            .unwrap_or_default()
                    })
                    .flatten()
                    .collect::<BTreeSet<_>>();
                (!key_phrases.is_empty()).then(|| key_phrases)
            })
            .flatten()
    }

    /// Removes the tuple and cleans up empty entries afterwards.
    pub(super) fn clean(&mut self, coi_id: CoiId, relevance: Relevance, key_phrase: &KeyPhrase) {
        if let Some(key_phrases) = self.relevance_to_key_phrase.get_mut(&(relevance, coi_id)) {
            key_phrases.retain(|this| this != key_phrase);
            if key_phrases.is_empty() {
                self.relevance_to_key_phrase.remove(&(relevance, coi_id));
                if let Some(relevances) = self.coi_to_relevance.get_mut(&coi_id) {
                    relevances.remove(&relevance);
                    if relevances.is_empty() {
                        self.coi_to_relevance.remove(&coi_id);
                    }
                }
            }
        }
    }

    /// Replaces the relevances in the tuples with a matching id.
    ///
    /// If old key phrases exist, then their order is preserved under the new relevance.
    pub(super) fn replace(&mut self, coi_id: CoiId, relevance: Relevance) {
        if let Some(old_relevances) = self
            .coi_to_relevance
            .insert(coi_id, IntoIterator::into_iter([relevance]).collect())
        {
            let len = old_relevances
                .iter()
                .filter_map(|&old_relevance| {
                    self.relevance_to_key_phrase
                        .get(&(old_relevance, coi_id))
                        .map(|key_phrases| key_phrases.len())
                })
                .sum::<usize>();
            if len > 0 {
                self.relevance_to_key_phrase
                    .entry((relevance, coi_id))
                    .and_modify(|key_phrases| key_phrases.reserve_exact(len))
                    .or_insert_with(|| Vec::with_capacity(len));
                for old_relevance in old_relevances {
                    if let Some(old_key_phrases) = self
                        .relevance_to_key_phrase
                        .remove(&(old_relevance, coi_id))
                    {
                        self.relevance_to_key_phrase
                            .entry((relevance, coi_id))
                            .or_default()
                            .extend(old_key_phrases);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        iter::{once, repeat},
        ops::Index,
    };

    use itertools::izip;
    use ndarray::Ix;

    use test_utils::ApproxEqIter;

    use super::*;

    impl Relevances {
        pub(crate) fn new<const N: usize>(
            ids: [CoiId; N],
            relevances: [f32; N],
            key_phrases: Vec<KeyPhrase>,
        ) -> Self {
            assert!(IntoIterator::into_iter(relevances).all(f32::is_finite));
            let len = key_phrases.len();
            assert!(len <= N);
            let key_phrases = key_phrases
                .into_iter()
                .map(Some)
                .chain(repeat(None).take(N - len))
                .collect::<Vec<_>>();

            let mut this = Self::default();
            for (coi_id, relevance, key_phrase) in izip!(ids, relevances, key_phrases) {
                let relevance = Relevance::new(relevance).unwrap();
                this.coi_to_relevance
                    .entry(coi_id)
                    .or_default()
                    .insert(relevance);
                if let Some(key_phrase) = key_phrase {
                    this.relevance_to_key_phrase
                        .entry((relevance, coi_id))
                        .or_default()
                        .push(key_phrase);
                }
            }

            this
        }

        pub(crate) fn cois_len(&self) -> usize {
            self.coi_to_relevance.len()
        }

        pub(crate) fn cois_is_empty(&self) -> bool {
            self.coi_to_relevance.is_empty()
        }

        pub(crate) fn relevances_len(&self) -> usize {
            self.relevance_to_key_phrase.len()
        }

        pub(crate) fn relevances_is_empty(&self) -> bool {
            self.relevance_to_key_phrase.is_empty()
        }
    }

    impl Index<CoiId> for Relevances {
        type Output = BTreeSet<Relevance>;

        fn index(&self, coi_id: CoiId) -> &Self::Output {
            &self.coi_to_relevance[&coi_id]
        }
    }

    impl Index<(CoiId, Relevance)> for Relevances {
        type Output = [KeyPhrase];

        fn index(&self, (coi_id, relevance): (CoiId, Relevance)) -> &Self::Output {
            &self.relevance_to_key_phrase[&(relevance, coi_id)]
        }
    }

    impl<'a> ApproxEqIter<'a, f32> for &'a Relevance {
        fn indexed_iter_logical_order(
            self,
            index_prefix: Vec<Ix>,
        ) -> Box<dyn Iterator<Item = (Vec<Ix>, f32)> + 'a> {
            Box::new(once((index_prefix, self.0)))
        }
    }
}
