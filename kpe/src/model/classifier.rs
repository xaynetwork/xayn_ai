use derive_more::{Deref, From};
use ndarray::s;

use crate::{
    model::{cnn::Features, ModelError},
    tokenizer::encoding::ActiveMask,
};
use layer::{activation::Linear, dense::Dense, io::BinParams};

/// A Classifier onnx model.
#[derive(Debug)]
pub struct ClassifierModel {
    layer: Dense<Linear>,
}

/// The inferred scores.
#[derive(Clone, Deref, From)]
pub struct Scores(pub Vec<f32>);

impl ClassifierModel {
    /// Creates a model from a binary parameters file.
    pub fn new(mut params: BinParams) -> Result<Self, ModelError> {
        let layer = Dense::load(params.with_scope("dense"), Linear)?;
        if !params.is_empty() {
            return Err(ModelError::UnusedParams(
                params.keys().map(Into::into).collect(),
            ));
        }

        Ok(Self { layer })
    }

    /// Runs the model on the convolved features to compute the scores.
    pub fn run(&self, features: Features, active_mask: ActiveMask) -> Result<Scores, ModelError> {
        debug_assert_eq!(features.shape()[1..], active_mask.shape()[..]);
        let (scores, _) = self.layer.run(features.0.slice(s![0, .., ..]), false);
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
    use ndarray::{Array2, Array3};

    use super::*;
    use test_utils::kpe::classifier;

    #[test]
    #[ignore = "check actual weight shapes"]
    fn test_run() {
        let model =
            ClassifierModel::new(BinParams::deserialize_from_file(classifier().unwrap()).unwrap())
                .unwrap();

        let key_phrase_size = 5;
        let channel_out_size = 512;
        let key_phrase_choices = 10;
        let features = Array3::zeros((1, key_phrase_size, channel_out_size)).into();
        let active_mask =
            Array2::from_elem((key_phrase_choices, key_phrase_choices + 5), false).into();

        let scores = model.run(features, active_mask).unwrap();
        assert_eq!(scores.len(), key_phrase_choices);
    }
}
