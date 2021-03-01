use std::collections::HashMap;

use crate::{
    normalizer::{NormalizedString, OffsetReferential, Offsets},
    Error,
};

/// Various possible types of offsets
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OffsetType {
    Byte,
    Char,
}

/// The `PreTokenizedString` is in charge of splitting an underlying string,
/// making sure everything is fine while doing so, and providing ways to normalize
/// and tokenize these splits.
/// Once everything has been normalized and tokenized, the `PreTokenizedString` is able
/// to build an `Encoding` with all the relevant offsets and word ids, relative to the
/// original string.
pub struct PreTokenizedString {
    pub original: String,
    pub splits: Vec<NormalizedString>,
}

impl From<NormalizedString> for PreTokenizedString {
    fn from(normalized: NormalizedString) -> Self {
        Self {
            original: normalized.original.clone(),
            splits: vec![normalized],
        }
    }
}

impl PreTokenizedString {
    /// Split the `PreTokenizedString` by providing a `split_fn` in charge of splitting
    /// each substring (`NormalizedString`) into multiple parts.
    ///
    /// `split_fn` takes a `NormalizedString` and is in charge of returning an iterator
    /// over the produced `NormalizedString`. `split_fn` is free of modifying these
    /// `NormalizedString` as relevant, as long as it respects the constraint stated below.
    ///
    /// There are only one constraint that *MUST* be respected:
    /// > The produced `NormalizedString`, if combined back together, must have the
    /// same `original` string as the original one given to `split_fn`. This concretely
    /// means that for the offset tracking to work as expected, `split_fn` must produce
    /// "splits" of the original string.
    pub(crate) fn split<F, U>(mut self, split_fn: F) -> Result<Self, Error>
    where
        F: Fn(usize, NormalizedString) -> Result<U, Error>,
        U: IntoIterator<Item = NormalizedString>,
    {
        // new_splits is at least as big as self.splits
        let mut new_splits = Vec::with_capacity(self.splits.len());
        for (i, original_split) in self.splits.drain(..).enumerate() {
            new_splits.extend(
                split_fn(i, original_split)?
                    .into_iter()
                    .filter_map(|split| {
                        if split.normalized.is_empty() {
                            None
                        } else {
                            Some(split)
                        }
                    }),
            );
        }
        self.splits = new_splits;

        Ok(self)
    }

    /// Returns a list of splits, each of them being a slice of the normalized
    /// string, the associated offsets either in original or normalized
    /// referential, as well as the potention tokens
    pub(crate) fn get_splits(
        &self,
        offset_ref: OffsetReferential,
        offset_type: OffsetType,
    ) -> Vec<(&str, Offsets)> {
        let offset_converter = match offset_type {
            OffsetType::Char => Some(BytesToCharOffsetConverter::new(&self.original)),
            OffsetType::Byte => None,
        };

        let mut offset = 0;
        self.splits
            .iter()
            .map(|split| {
                let mut offsets = match offset_ref {
                    OffsetReferential::Original => split.offsets_original(),
                    OffsetReferential::Normalized => {
                        let len = split.normalized.len();
                        offset += len;
                        Offsets(offset - len, offset)
                    }
                };

                // Convert to char offsets if relevant
                if let Some(ref converter) = offset_converter {
                    offsets = converter.convert(offsets).unwrap_or(offsets);
                }

                (split.normalized.as_str(), offsets)
            })
            .collect()
    }
}

pub struct BytesToCharOffsetConverter {
    map: HashMap<usize, usize>,
}

impl BytesToCharOffsetConverter {
    pub fn new(sequence: &str) -> Self {
        Self {
            map: sequence
                .char_indices()
                .enumerate()
                .flat_map(|(i, (b, c))| {
                    let mut n = 0;
                    std::iter::repeat_with(move || {
                        let o = (b + n, i);
                        n += 1;
                        o
                    })
                    .take(c.len_utf8())
                })
                .collect(),
        }
    }

    pub fn convert(&self, offsets: Offsets) -> Option<Offsets> {
        match (self.map.get(&offsets.0), self.map.get(&offsets.1)) {
            (Some(start), Some(end)) => Some(Offsets(*start, *end)),
            // If we reached the end, `end` is not in the map
            (Some(start), None) => {
                // But the one just before should be
                let last = self.map.get(&(offsets.1 - 1)).copied().unwrap_or(start + 1);
                Some(Offsets(*start, last + 1))
            }
            _ => None,
        }
    }
}
