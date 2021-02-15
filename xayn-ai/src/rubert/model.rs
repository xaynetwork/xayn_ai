use std::io::Read;

use tract_onnx::prelude::{
    tvec,
    Datum,
    Framework,
    InferenceFact,
    InferenceModelExt,
    TVec,
    TractError,
    TractResult,
    TypedModel,
    TypedSimplePlan,
};

use crate::rubert::{
    ndarray::{s, Array2, Data},
    utils::{ArcArrayD, ArrayBase2},
};

/// A Bert model based on an onnx definition.
pub struct RuBertModel {
    plan: TypedSimplePlan<TypedModel>,
    batch_size: usize,
    tokens_size: usize,
}

impl RuBertModel {
    pub fn new(mut model: impl Read, batch_size: usize, tokens_size: usize) -> TractResult<Self> {
        let input_shape = tvec!(batch_size, tokens_size);
        // the exported onnx model which we currently use expects i64 as input type
        // TODO: generalize this for other input types
        let input_fact = InferenceFact::dt_shape(i64::datum_type(), input_shape);

        let plan = tract_onnx::onnx()
            .model_for_read(&mut model)?
            .with_input_fact(0, input_fact.clone())?
            .with_input_fact(1, input_fact.clone())?
            .with_input_fact(2, input_fact)?
            .into_optimized()?
            .into_runnable()?;

        Ok(RuBertModel {
            plan,
            batch_size,
            tokens_size,
        })
    }

    /// Runs prediction on the tokenized inputs.
    ///
    /// The inputs will be padded or truncated to the shape `(batch_size, tokens_size)`. The row
    /// dimension (0) of the output will be the minimum between `batch_size` and the row dimension
    /// of the inputs.
    ///
    /// The output dimensionality depends on the type of Bert model loaded from onnx.
    ///
    /// # Errors
    /// The constraint `input_ids.shape() == attention_masks.shape() == token_type_ids.shape()` must
    /// hold.
    pub fn predict<S1, S2, S3>(
        &self,
        input_ids: ArrayBase2<S1>,
        attention_masks: ArrayBase2<S2>,
        token_type_ids: ArrayBase2<S3>,
    ) -> TractResult<ArcArrayD<f32>>
    where
        S1: Data<Elem = u32>,
        S2: Data<Elem = u32>,
        S3: Data<Elem = u32>,
    {
        if input_ids.shape() != attention_masks.shape()
            || input_ids.shape() != token_type_ids.shape()
        {
            return Err(TractError::msg("mismatched Bert input shapes"));
        }

        let input_shape = (self.batch_size, self.tokens_size);
        let output_rows = std::cmp::min(input_ids.dim().0, self.batch_size);

        let inputs = tvec!(
            pad_or_truncate(input_ids, input_shape).into(),
            pad_or_truncate(attention_masks, input_shape).into(),
            pad_or_truncate(token_type_ids, input_shape).into()
        );
        let outputs = self.plan.run(inputs)?;

        let outputs = outputs[0].to_array_view()?;
        let outputs = match outputs.ndim() {
            2 => outputs.slice(s!(..output_rows, ..)).into_dyn(),
            3 => outputs.slice(s!(..output_rows, .., ..)).into_dyn(),
            _ => unimplemented!("unsupported Bert output dimensionality"),
        };
        Ok(outputs.to_shared())
    }

    /// Returns the batch size of the model.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the tokens size of the model.
    pub fn tokens_size(&self) -> usize {
        self.tokens_size
    }

    /// Returns the embedding size of the model.
    ///
    /// # Panics
    /// This assumes that the model output is not empty.
    pub fn embedding_size(&self) -> usize {
        self.output_shape().pop().unwrap()
    }

    /// Returns the dimensionality of the model output.
    ///
    /// # Panics
    /// This assumes that the model output is not empty.
    pub fn output_rank(&self) -> usize {
        self.plan.model().output_fact(0).unwrap().rank()
    }

    /// Returns the shape of the model output.
    ///
    /// # Panics
    /// This assumes that the model output is not empty.
    pub fn output_shape(&self) -> TVec<usize> {
        self.plan
            .model()
            .output_fact(0)
            .unwrap()
            .shape
            .as_finite()
            .unwrap()
    }
}

/// Pads or truncates the `array` to the `shape`.
fn pad_or_truncate<S>(array: ArrayBase2<S>, shape: (usize, usize)) -> Array2<i64>
where
    S: Data<Elem = u32>,
{
    Array2::from_shape_fn(shape, |coords| *array.get(coords).unwrap_or(&0) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_truncate_same_dim() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones(dim);
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![.., ..]).iter().all(|e| *e == 1));
    }

    #[test]
    fn test_pad_truncate_bigger() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((7, 7));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![.., ..]).iter().all(|e| *e == 1));
    }

    #[test]
    fn test_pad_truncate_bigger_rows_same_cols() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((7, 5));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![.., ..]).iter().all(|e| *e == 1));
    }

    #[test]
    fn test_pad_truncate_bigger_rows_smaller_cols() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((7, 3));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![.., ..3]).iter().all(|e| *e == 1));
        assert!(r.slice(s![.., 3..]).iter().all(|e| *e == 0));
    }

    #[test]
    fn test_pad_truncate_bigger_cols_same_rows() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((5, 7));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![.., ..]).iter().all(|e| *e == 1));
    }

    #[test]
    fn test_pad_truncate_bigger_cols_smaller_rows() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((3, 7));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![..3, ..]).iter().all(|e| *e == 1));
        assert!(r.slice(s![3.., ..]).iter().all(|e| *e == 0));
    }

    #[test]
    fn test_pad_truncate_smaller() {
        let dim = (5, 5);

        let a = Array2::<u32>::ones((3, 3));
        let r = pad_or_truncate(a, dim);

        assert_eq!(r.dim(), dim);
        assert!(r.slice(s![..3, ..3]).iter().all(|e| *e == 1));
        assert!(r.slice(s![3.., ..]).iter().all(|e| *e == 0));
        assert!(r.slice(s![.., 3..]).iter().all(|e| *e == 0));
    }
}
