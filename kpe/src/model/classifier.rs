use std::io::Read;

use derive_more::{Deref, From};
use ndarray::s;
use tract_onnx::prelude::{
    tvec,
    Datum,
    Framework,
    InferenceFact,
    InferenceModelExt,
    IntoTensor,
    TypedModel,
    TypedSimplePlan,
};

use crate::{
    model::{cnn::Features, ModelError},
    tokenizer::encoding::ActiveMask,
};

/// A Classifier onnx model.
#[derive(Debug)]
pub struct ClassifierModel {
    plan: TypedSimplePlan<TypedModel>,
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
    pub fn new(
        mut model: impl Read,
        key_phrase_size: usize,
        cnn_out_channel_size: usize,
    ) -> Result<Self, ModelError> {
        let input_fact = InferenceFact::dt_shape(
            f32::datum_type(),
            &[1, key_phrase_size, cnn_out_channel_size],
        );
        let plan = tract_onnx::onnx()
            .model_for_read(&mut model)?
            .with_input_fact(0, input_fact)? // convolved features
            .into_optimized()?
            .into_runnable()?;

        Ok(ClassifierModel { plan })
    }

    /// Runs the model on the convolved features to compute the scores.
    pub fn run(&self, features: Features, active_mask: ActiveMask) -> Result<Scores, ModelError> {
        debug_assert_eq!(features.shape()[1..], active_mask.shape()[..]);
        let inputs = tvec!(features.0.into_tensor());
        let outputs = self.plan.run(inputs)?;
        debug_assert!(outputs[0]
            .to_array_view::<f32>()?
            .iter()
            .all(|v| !v.is_infinite() && !v.is_nan()));

        // TODO: check if the shapes match once we have the models loaded
        let scores = outputs[0].to_array_view::<f32>()?;
        let scores = scores.slice(s![0, .., 0]);
        let scores = active_mask
            .rows()
            .into_iter()
            .map(|active| {
                active
                    .iter()
                    .zip(scores)
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
    use std::{fs::File, io::BufReader};

    use ndarray::{Array2, Array3};
    use tract_onnx::prelude::IntoArcTensor;

    use super::*;
    use test_utils::smbert::model;

    #[test]
    fn test_model_empty() {
        assert!(matches!(
            ClassifierModel::new(Vec::new().as_slice(), 5, 128).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    fn test_model_invalid() {
        assert!(matches!(
            ClassifierModel::new([0].as_ref(), 5, 128).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    #[ignore = "missing classifier model asset"]
    fn test_key_phrase_size_invalid() {
        let model = BufReader::new(File::open(model().unwrap()).unwrap());
        assert!(matches!(
            ClassifierModel::new(model, 0, 128).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    #[ignore = "missing classifier model asset"]
    fn test_cnn_out_channel_size_invalid() {
        let model = BufReader::new(File::open(model().unwrap()).unwrap());
        assert!(matches!(
            ClassifierModel::new(model, 5, 0).unwrap_err(),
            ModelError::Tract(_),
        ));
    }

    #[test]
    #[ignore = "missing classifier model asset"]
    fn test_run() {
        let key_phrase_size = 5;
        let cnn_out_channel_size = 128;
        let key_phrase_choices = 10;
        let model = BufReader::new(File::open(model().unwrap()).unwrap());
        let model = ClassifierModel::new(model, key_phrase_size, cnn_out_channel_size).unwrap();

        let features = Array3::from_elem((1, key_phrase_size, cnn_out_channel_size), 0)
            .into_arc_tensor()
            .into();
        let active_mask =
            Array2::from_elem((key_phrase_choices, key_phrase_choices + 5), false).into();
        let scores = model.run(features, active_mask).unwrap();
        assert_eq!(scores.len(), key_phrase_choices);
    }
}
