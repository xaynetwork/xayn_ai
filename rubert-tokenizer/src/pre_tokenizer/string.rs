use crate::normalizer::string::NormalizedString;

/// A pre-tokenized sequence.
pub struct PreTokenizedString {
    pub original: String,
    pub splits: Vec<NormalizedString>,
}

impl From<NormalizedString> for PreTokenizedString {
    fn from(sequence: NormalizedString) -> Self {
        Self {
            original: sequence.original.clone(),
            splits: vec![sequence],
        }
    }
}

impl PreTokenizedString {
    /// Splits wrt the function.
    ///
    /// The function takes a normalized sequence and returns an iterator over normalized
    /// subsequences. The combined normalized subsequences must have the same original sequence as
    /// the normalized sequence.
    pub fn split<F, S>(mut self, f: F) -> Self
    where
        F: Fn(usize, NormalizedString) -> S,
        S: IntoIterator<Item = NormalizedString>,
    {
        // new_splits is at least as big as self.splits
        let mut new_splits = Vec::with_capacity(self.splits.len());
        for (i, original_split) in self.splits.drain(..).enumerate() {
            new_splits.extend(
                f(i, original_split)
                    .into_iter()
                    .filter(|split| !split.normalized.is_empty()),
            );
        }
        self.splits = new_splits;

        self
    }
}
