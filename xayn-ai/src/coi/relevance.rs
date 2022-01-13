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
    /// Iterates over all tuples in ascending relevance.
    pub(super) fn iter(
        &self,
    ) -> impl Iterator<Item = (CoiId, Relevance, &KeyPhrase)> + DoubleEndedIterator {
        self.relevance_to_key_phrase
            .iter()
            .map(|(&(relevance, coi_id), key_phrases)| {
                key_phrases
                    .iter()
                    .map(move |key_phrase| (coi_id, relevance, key_phrase))
            })
            .flatten()
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

    /// Removes all tuples with the given id.
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

    /// Replaces the relevances in the tuples with the given id.
    pub(super) fn replace(&mut self, coi_id: CoiId, mut relevances: Vec<Relevance>) {
        if let Some(old_relevances) = self
            .coi_to_relevance
            .insert(coi_id, relevances.iter().copied().collect())
        {
            relevances.sort_unstable_by(|this, other| this.cmp(other).reverse());
            let key_phrases = old_relevances
                .into_iter()
                .map(|old_relevance| {
                    self.relevance_to_key_phrase
                        .remove(&(old_relevance, coi_id))
                        .unwrap_or_default()
                })
                .flatten()
                .rev()
                .collect::<Vec<_>>();
            for (relevance, key_phrase) in relevances.into_iter().zip(key_phrases) {
                self.relevance_to_key_phrase
                    .entry((relevance, coi_id))
                    .or_default()
                    .push(key_phrase);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{iter::once, ops::Index};

    use ndarray::Ix;

    use test_utils::ApproxEqIter;

    use super::*;

    impl Relevances {
        pub fn cois_len(&self) -> usize {
            self.coi_to_relevance.len()
        }

        pub fn cois_is_empty(&self) -> bool {
            self.coi_to_relevance.is_empty()
        }

        pub fn relevances_len(&self) -> usize {
            self.relevance_to_key_phrase.len()
        }

        pub fn relevances_is_empty(&self) -> bool {
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
