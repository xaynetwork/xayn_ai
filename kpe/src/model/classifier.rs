use derive_more::{Deref, From};
use ndarray::{s, Array1, Array2};

use crate::{
    model::{cnn::Features, ModelError},
    tokenizer::encoding::ActiveMask,
};
use layer::{activation::Linear, dense::Dense};

/// A Classifier onnx model.
#[derive(Debug)]
pub struct ClassifierModel {
    dense: Dense<Linear>,
}

/// The inferred scores.
#[derive(Clone, Deref, From)]
pub struct Scores(pub Vec<f32>);

impl ClassifierModel {
    /// Creates a model from an onnx model file.
    ///
    /// Requires the maximum number of words per key phrase and the size of the output channel of
    /// the CNN model.
    ///
    /// # Panics
    /// Panics if the model is empty (due to the way tract implemented the onnx model parsing).
    pub fn new(weights: Array2<f32>, bias: Array1<f32>) -> Result<Self, ModelError> {
        let dense = Dense::new(weights, bias, Linear)?;

        Ok(ClassifierModel { dense })
    }

    /// Runs the model on the convolved features to compute the scores.
    pub fn run(&self, features: Features, active_mask: ActiveMask) -> Result<Scores, ModelError> {
        debug_assert_eq!(features.shape()[1..], active_mask.shape()[..]);
        let (scores, _) = self.dense.run(features.0.slice(s![0, .., ..]), false);
        debug_assert!(scores.iter().all(|v| !v.is_infinite() && !v.is_nan()));

        let scores = active_mask
            .rows()
            .into_iter()
            .map(|active| {
                active
                    .iter()
                    .zip(scores.iter())
                    .filter_map(|(active, score)| active.then(|| score))
                    .copied()
                    .reduce(f32::max)
                    .unwrap(/* active mask must have entries in each row */)
            })
            .collect::<Vec<f32>>();
        debug_assert!(scores.iter().all(|v| !v.is_infinite() && !v.is_nan()));

        Ok(scores.into())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;

    use super::*;

    #[test]
    #[ignore = "check actual weight shapes"]
    fn test_run() {
        let key_phrase_size = 5;
        let channel_out_size = 128;
        let key_phrase_choices = 10;
        let model =
            ClassifierModel::new(Array2::zeros((channel_out_size, 1)), Array1::zeros(1)).unwrap();

        let features = Array3::zeros((1, key_phrase_size, channel_out_size)).into();
        let active_mask =
            Array2::from_elem((key_phrase_choices, key_phrase_choices + 5), false).into();
        let scores = model.run(features, active_mask).unwrap();
        assert_eq!(scores.len(), key_phrase_choices);
    }
}
