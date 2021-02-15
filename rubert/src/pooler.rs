use crate::{
    anyhow::{anyhow, Result},
    ndarray::{s, Axis, Data},
    utils::{ArcArray2, ArcArrayD, ArrayBase2, ArrayBase3, ArrayBaseD},
};

/// Averages the `predictions` along the tokens dimension (1) discarding any padding.
///
/// The `predictions` are of shape `(batch_size x tokens_size x embedding_size)`.
///
/// # Errors
/// The `attention_masks` must be binary and the first two dimensions of the `predictions` and
/// `attention_masks` must coincide.
fn average<S1, S2>(
    predictions: ArrayBase3<S1>,
    attention_masks: ArrayBase2<S2>,
) -> Result<ArcArray2<f32>>
where
    S1: Data<Elem = f32>,
    S2: Data<Elem = u32>,
{
    if &predictions.shape()[0..2] != attention_masks.shape() {
        return Err(anyhow!("mismatched input shapes"));
    }
    if !attention_masks.iter().all(|e| e == &0 || e == &1) {
        return Err(anyhow!("invalid attention masks"));
    }

    let masks = attention_masks.mapv(|v| v as f32);
    let tokens_count = masks.sum_axis(Axis(1)).insert_axis(Axis(1));
    let masks = masks.insert_axis(Axis(2));

    let averaged = predictions.into_owned() * masks;
    let averaged = averaged.sum_axis(Axis(1)) / tokens_count;

    Ok(averaged.into_shared())
}

/// Picks the first element of the tokens dimension (`[CLS]`) of the `predictions`.
///
/// The `predictions` are of shape `(batch_size x tokens_size x embedding_size)`.
fn first<S>(predictions: ArrayBase3<S>) -> ArcArray2<f32>
where
    S: Data<Elem = f32>,
{
    predictions.slice(s!(.., 0, ..)).to_shared()
}

/// Pooling strategies for [`RuBert`] models.
///
/// [`RuBert`]: struct.RuBert.html
pub enum RuBertPooler {
    /// No pooling, for example if the model already has a pooling layer.
    None,
    /// Pooling over the first token of the model output.
    First,
    /// Pooling over the averaged tokens dimension of the model output.
    Average,
}

impl RuBertPooler {
    /// Pools the predictions according to the pooling strategy.
    ///
    /// The `predictions` are of shape `(batch_size x tokens_size x embedding_size)`.
    ///
    /// Pooling stategies:
    /// - None: The `predictions` are just passed through.
    /// - First: The first element of the tokens dimension (`[CLS]`) of the `predictions` is picked.
    /// - Average: The `predictions` are averaged along the tokens dimension (1) discarding any
    /// padding. The `attention_masks` must be binary and the first two dimensions of the
    /// `predictions` and `attention_masks` must coincide.
    pub(crate) fn pool<S1, S2>(
        &self,
        predictions: ArrayBaseD<S1>,
        attention_masks: ArrayBase2<S2>,
    ) -> Result<ArcArrayD<f32>>
    where
        S1: Data<Elem = f32>,
        S2: Data<Elem = u32>,
    {
        Ok(match self {
            Self::None => predictions.into_owned().into_shared(),
            Self::First => first(predictions.into_dimensionality()?).into_dyn(),
            Self::Average => {
                average(predictions.into_dimensionality()?, attention_masks)?.into_dyn()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::ndarray::{arr2, arr3};

    use super::*;

    #[test]
    fn test_average() {
        let predictions = arr3(&[
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        ]);
        let attention_masks = arr2(&[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]);
        assert_eq!(
            average(predictions, attention_masks).unwrap(),
            arr2(&[
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [2.5, 3.5, 4.5],
                [4., 5., 6.],
                [5.5, 6.5, 7.5],
                [4., 5., 6.]
            ]),
        );
    }

    #[test]
    fn test_first() {
        let predictions = arr3(&[
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[4., 5., 6.], [7., 8., 9.], [1., 2., 3.]],
            [[7., 8., 9.], [1., 2., 3.], [4., 5., 6.]],
        ]);
        assert_eq!(
            first(predictions),
            arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        );
    }
}
