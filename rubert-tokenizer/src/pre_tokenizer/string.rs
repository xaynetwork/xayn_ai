use crate::{normalizer::string::NormalizedString, Error};

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
    pub(crate) fn split<F, S>(mut self, split_fn: F) -> Result<Self, Error>
    where
        F: Fn(usize, NormalizedString) -> Result<S, Error>,
        S: IntoIterator<Item = NormalizedString>,
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
}
