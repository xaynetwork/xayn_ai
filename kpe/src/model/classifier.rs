use derive_more::{Deref, From};

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
///
/// The scores are of shape `(len(key_phrase_choices),)`.
#[derive(Clone, Debug, Deref, From)]
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
        debug_assert_eq!(features.shape()[1], active_mask.shape()[1]);
        debug_assert!(active_mask
            .rows()
            .into_iter()
            .all(|row| row.iter().any(|active| *active)));
        let (scores, _) = self.layer.run(features.t(), false);
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
    use ndarray::Array2;

    use super::*;
    use test_utils::kpe::classifier;

    #[test]
    fn test_model_shapes() {
        let model =
            ClassifierModel::new(BinParams::deserialize_from_file(classifier().unwrap()).unwrap())
                .unwrap();
        assert_eq!(model.layer.weights().shape(), [512, 1],);
        assert_eq!(model.layer.bias().shape(), [1]);
    }

    #[test]
    fn test_model_empty() {
        matches!(
            ClassifierModel::new(BinParams::default()).unwrap_err(),
            ModelError::Classifier(_),
        );
    }

    #[test]
    fn test_run_unique() {
        let channel_out_size = 512;
        let output_size = 42;

        let model =
            ClassifierModel::new(BinParams::deserialize_from_file(classifier().unwrap()).unwrap())
                .unwrap();
        let features = Array2::zeros((channel_out_size, output_size)).into();
        let active_mask = Array2::from_elem((output_size, output_size), true).into();

        let scores = model.run(features, active_mask).unwrap();
        assert_eq!(scores.len(), output_size);
    }

    #[test]
    fn test_run_duplicate() {
        let channel_out_size = 512;
        let output_size = 42;

        let model =
            ClassifierModel::new(BinParams::deserialize_from_file(classifier().unwrap()).unwrap())
                .unwrap();
        let features = Array2::zeros((channel_out_size, output_size)).into();
        let active_mask = Array2::from_elem((output_size / 2, output_size), true).into();

        let scores = model.run(features, active_mask).unwrap();
        assert_eq!(scores.len(), output_size / 2);
    }
}
