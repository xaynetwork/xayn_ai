use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
};

use derive_more::Into;
use serde::{Deserialize, Serialize};

use crate::coi::{key_phrase::KeyPhrase, CoiError, CoiId};

/// A finite f32.
#[derive(Clone, Copy, Debug, Default, Into, PartialEq, PartialOrd, Serialize, Deserialize)]
// invariant: the wrapped value must always be finite
struct F32(f32);

impl PartialEq<f32> for F32 {
    fn eq(&self, other: &f32) -> bool {
        self.0.eq(other)
    }
}

impl Eq for F32 {
    // never nan by invariant
}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for F32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap(/* never nan by invariant */)
    }
}

/// A relevance score.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub(crate) struct Relevance(Rel);

/// Relevance score variants.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
enum Rel {
    /// A coi relevance score.
    Coi(F32),
    /// A key phrase relevance score.
    Kp(F32),
}

impl Relevance {
    /// Creates a coi relevance.
    ///
    /// # Errors
    /// Fails if the relevance isn't finite.
    pub(crate) fn coi(relevance: f32) -> Result<Self, CoiError> {
        if relevance.is_finite() {
            Ok(Self(Rel::Coi(F32(relevance))))
        } else {
            Err(CoiError::NonFiniteRelevance)
        }
    }

    /// Creates a key phrase relevance.
    ///
    /// # Errors
    /// Fails if the relevance isn't finite.
    pub(crate) fn kp(relevance: f32) -> Result<Self, CoiError> {
        if relevance.is_finite() {
            Ok(Self(Rel::Kp(F32(relevance))))
        } else {
            Err(CoiError::NonFiniteRelevance)
        }
    }
}

impl From<Relevance> for f32 {
    fn from(relevance: Relevance) -> Self {
        match relevance {
            Relevance(Rel::Coi(F32(relevance))) => relevance,
            Relevance(Rel::Kp(F32(relevance))) => relevance,
        }
    }
}

/// Relevance scores.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Relevances(Rels);

/// Relevance scores variants.
#[derive(Debug, Serialize, Deserialize)]
enum Rels {
    /// Coi relevance scores.
    Coi(F32),
    Kps(BTreeSet<F32>),
}

/// Sorted maps from cois to relevances to key phrases.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct RelevanceMap {
    coi_to_relevance: HashMap<CoiId, Relevances>,
    relevance_to_key_phrase: BTreeMap<(Relevance, CoiId), Vec<KeyPhrase>>,
    cleaned: HashMap<CoiId, Vec<(Relevance, KeyPhrase)>>,
}

impl RelevanceMap {
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
        match relevance {
            Relevance(Rel::Coi(relevance)) => {
                self.coi_to_relevance
                    .insert(coi_id, Relevances(Rels::Coi(relevance)));
            }
            Relevance(Rel::Kp(relevance)) => {
                if let Some(Relevances(Rels::Kps(relevances))) =
                    self.coi_to_relevance.get_mut(&coi_id)
                {
                    relevances.insert(relevance);
                } else {
                    self.coi_to_relevance.insert(
                        coi_id,
                        Relevances(Rels::Kps(IntoIterator::into_iter([relevance]).collect())),
                    );
                }
            }
        }
        self.relevance_to_key_phrase
            .entry((relevance, coi_id))
            .or_default()
            .push(key_phrase);
    }

    /// Removes all tuples with a matching id.
    pub(super) fn remove(&mut self, coi_id: CoiId) -> Option<BTreeSet<KeyPhrase>> {
        self.cleaned.remove(&coi_id);
        self.coi_to_relevance
            .remove(&coi_id)
            .and_then(|relevances| {
                let key_phrases = match relevances {
                    Relevances(Rels::Coi(relevance)) => self
                        .relevance_to_key_phrase
                        .remove(&(Relevance(Rel::Coi(relevance)), coi_id))
                        .unwrap_or_default()
                        .into_iter()
                        .collect::<BTreeSet<_>>(),
                    Relevances(Rels::Kps(relevances)) => relevances
                        .into_iter()
                        .filter_map(|relevance| {
                            self.relevance_to_key_phrase
                                .remove(&(Relevance(Rel::Kp(relevance)), coi_id))
                        })
                        .flatten()
                        .collect::<BTreeSet<_>>(),
                };
                (!key_phrases.is_empty()).then(|| key_phrases)
            })
    }

    /// Removes the tuple and cleans up empty entries afterwards.
    pub(super) fn clean(&mut self, coi_id: CoiId, relevance: Relevance, key_phrase: &KeyPhrase) {
        if let Some(key_phrases) = self.relevance_to_key_phrase.get_mut(&(relevance, coi_id)) {
            if let Some(idx) = key_phrases.iter().position(|kp| kp == key_phrase) {
                let key_phrase = key_phrases.remove(idx);

                self.cleaned
                    .entry(coi_id)
                    .or_default()
                    .push((relevance, key_phrase));
            }
            if key_phrases.is_empty() {
                self.relevance_to_key_phrase.remove(&(relevance, coi_id));
            }
        }
        if let Some(relevances) = self.coi_to_relevance.get_mut(&coi_id) {
            match (relevances, relevance) {
                (Relevances(Rels::Coi(relevances)), Relevance(Rel::Coi(ref relevance)))
                    if relevances == relevance =>
                {
                    self.coi_to_relevance.remove(&coi_id);
                }
                (Relevances(Rels::Kps(relevances)), Relevance(Rel::Kp(ref relevance))) => {
                    relevances.remove(relevance);
                    if relevances.is_empty() {
                        self.coi_to_relevance.remove(&coi_id);
                    }
                }
                _ => {}
            }
        }
    }

    /// Replaces the relevances in the tuples with a matching id.
    ///
    /// If old key phrases exist, then their order is preserved under the new relevance.
    pub(super) fn replace(&mut self, coi_id: CoiId, relevance: Relevance) {
        let new_relevances = match relevance {
            Relevance(Rel::Coi(relevance)) => Relevances(Rels::Coi(relevance)),
            Relevance(Rel::Kp(relevance)) => {
                Relevances(Rels::Kps(IntoIterator::into_iter([relevance]).collect()))
            }
        };
        if let Some(old_relevances) = self.coi_to_relevance.insert(coi_id, new_relevances) {
            match old_relevances {
                Relevances(Rels::Coi(old_relevance)) => {
                    if let Some(old_key_phrases) = self
                        .relevance_to_key_phrase
                        .remove(&(Relevance(Rel::Coi(old_relevance)), coi_id))
                    {
                        self.relevance_to_key_phrase
                            .entry((relevance, coi_id))
                            .or_default()
                            .extend(old_key_phrases);
                    }
                }
                Relevances(Rels::Kps(old_relevances)) => {
                    let len = old_relevances
                        .iter()
                        .filter_map(|&old_relevance| {
                            self.relevance_to_key_phrase
                                .get(&(Relevance(Rel::Kp(old_relevance)), coi_id))
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
                                .remove(&(Relevance(Rel::Kp(old_relevance)), coi_id))
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
    }

    pub(crate) fn relevance_for_coi(&self, id: &CoiId) -> Option<f32> {
        match self.coi_to_relevance.get(id) {
            Some(Relevances(Rels::Coi(F32(relevance)))) => Some(*relevance),
            _ => None,
        }
    }

    pub(super) fn insert_cleaned_if_empty(&mut self) {
        if self.relevance_to_key_phrase.is_empty() {
            let to_insert = std::mem::take(&mut self.cleaned);

            for (coi_id, value) in to_insert.into_iter() {
                // we need to insert in reverse order to keep the order in which
                // the key phrases were originally. this is due to the fact that in
                // select_top_key_phrases() we collect and sort the key phrases in
                // descending order wrt the penalized relevances.
                for (relevance, key_phrase) in value.into_iter().rev() {
                    self.insert(coi_id, relevance, key_phrase);
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
        /// Gets the number of relevances.
        pub(crate) fn len(&self) -> usize {
            match self {
                Self(Rels::Coi(_)) => 1,
                Self(Rels::Kps(relevances)) => relevances.len(),
            }
        }

        /// Copies the relevance at the given index.
        ///
        /// # Panics
        /// Panics if the index is out of bounds.
        pub(crate) fn to_relevance(&self, index: usize) -> Relevance {
            assert!(
                index < self.len(),
                "index {} out of bounds for {:?}",
                index,
                self,
            );
            match self {
                Self(Rels::Coi(relevance)) => Relevance(Rel::Coi(*relevance)),
                Self(Rels::Kps(relevances)) => {
                    Relevance(Rel::Kp(*relevances.iter().nth(index).unwrap()))
                }
            }
        }

        /// Checks if this is a coi relevance.
        pub(crate) fn is_coi(&self) -> bool {
            matches!(self, Self(Rels::Coi(_)))
        }

        /// Checks if these are key phrase relevances.
        pub(crate) fn is_kp(&self) -> bool {
            matches!(self, Self(Rels::Kps(_)))
        }
    }

    impl RelevanceMap {
        /// Creates a map with key phrase relevances.
        pub(crate) fn kp<const N: usize>(
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
                let relevance = F32(relevance);
                this.coi_to_relevance
                    .entry(coi_id)
                    .and_modify(|relevances| {
                        if let Relevances(Rels::Kps(relevances)) = relevances {
                            relevances.insert(relevance);
                        }
                    })
                    .or_insert_with(|| {
                        Relevances(Rels::Kps(IntoIterator::into_iter([relevance]).collect()))
                    });
                if let Some(key_phrase) = key_phrase {
                    this.relevance_to_key_phrase
                        .entry((Relevance(Rel::Kp(relevance)), coi_id))
                        .or_default()
                        .push(key_phrase);
                }
            }

            this
        }

        /// Gets the number of cois.
        pub(crate) fn cois_len(&self) -> usize {
            self.coi_to_relevance.len()
        }

        /// Checks if there are no cois.
        pub(crate) fn cois_is_empty(&self) -> bool {
            self.coi_to_relevance.is_empty()
        }

        /// Gets the number of relevances.
        pub(crate) fn relevances_len(&self) -> usize {
            self.relevance_to_key_phrase.len()
        }

        /// Checks if there are no relevances.
        pub(crate) fn relevances_is_empty(&self) -> bool {
            self.relevance_to_key_phrase.is_empty()
        }
    }

    impl Index<CoiId> for RelevanceMap {
        type Output = Relevances;

        fn index(&self, coi_id: CoiId) -> &Self::Output {
            &self.coi_to_relevance[&coi_id]
        }
    }

    impl Index<(CoiId, Relevance)> for RelevanceMap {
        type Output = [KeyPhrase];

        fn index(&self, (coi_id, relevance): (CoiId, Relevance)) -> &Self::Output {
            &self.relevance_to_key_phrase[&(relevance, coi_id)]
        }
    }

    impl<'a> ApproxEqIter<'a, f32> for &'a F32 {
        fn indexed_iter_logical_order(
            self,
            index_prefix: Vec<Ix>,
        ) -> Box<dyn Iterator<Item = (Vec<Ix>, f32)> + 'a> {
            Box::new(once((index_prefix, self.0)))
        }
    }

    impl<'a> ApproxEqIter<'a, f32> for &'a Relevances {
        fn indexed_iter_logical_order(
            self,
            index_prefix: Vec<Ix>,
        ) -> Box<dyn Iterator<Item = (Vec<Ix>, f32)> + 'a> {
            match self {
                Relevances(Rels::Coi(relevance)) => {
                    relevance.indexed_iter_logical_order(index_prefix)
                }
                Relevances(Rels::Kps(relevances)) => {
                    relevances.indexed_iter_logical_order(index_prefix)
                }
            }
        }
    }
}
