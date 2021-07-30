//! This module contains utility for loading storing ndarray arrays

use bincode::Options;
use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
    io::{self, Read},
};

#[cfg(test)]
use std::{fs::File, io::BufReader, path::Path};

use thiserror::Error;

use ndarray::{ArrayBase, DataOwned, Dim, Dimension, IntoDimension, Ix, Ix1, IxDyn};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use ndarray::Array;

/// Deserialization helper representing a flattened array.
///
/// The flattened array is in row-major order.
#[derive(Serialize)]
#[cfg_attr(test, derive(Debug, Default, PartialEq))]
pub(crate) struct FlattenedArray<A> {
    shape: Vec<Ix>,
    /// There is a invariant that the length of data is
    /// equal to the product of all values in shape.
    data: Vec<A>,
}

#[cfg(test)]
impl<A, D> From<Array<A, D>> for FlattenedArray<A>
where
    A: Copy,
    D: Dimension,
{
    fn from(array: Array<A, D>) -> Self {
        //only used in tests so we don't care about the
        //unnecessary addition allocation, if used outside
        //of tests consider using `is_standard_layout()` and
        //`.into_raw_vec()`.
        let shape = array.shape().to_owned();
        let data = array.iter().copied().collect();

        FlattenedArray { shape, data }
    }
}

impl<'de, A> Deserialize<'de> for FlattenedArray<A>
where
    A: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = FlattenedArrayDeserializationHelper::<A>::deserialize(deserializer)?;

        let expected_data_len = helper.shape.iter().product::<usize>();
        if helper.data.len() != expected_data_len {
            return Err(<D::Error as serde::de::Error>::custom(
                UnexpectedNumberOfDimensions {
                    got: helper.data.len(),
                    expected: expected_data_len,
                },
            ));
        } else {
            return Ok(Self {
                shape: helper.shape,
                data: helper.data,
            });
        };

        /// Helper to get a post serialization invariant check.
        #[derive(Deserialize)]
        struct FlattenedArrayDeserializationHelper<A> {
            shape: Vec<Ix>,
            data: Vec<A>,
        }
    }
}

#[derive(Debug, Error)]
#[error("Unexpected number of dimensions: got={got}, expected={expected}")]
pub struct UnexpectedNumberOfDimensions {
    got: usize,
    expected: usize,
}

#[derive(Debug, Error)]
pub enum FailedToRetrieveParams {
    #[error(transparent)]
    UnexpectedNumberOfDimensions(#[from] UnexpectedNumberOfDimensions),

    #[error("Missing parameters for {name}.")]
    MissingParameters { name: String },
}

impl<S, D> TryFrom<FlattenedArray<S::Elem>> for ArrayBase<S, D>
where
    D: Dimension + TryIntoDimension,
    S: DataOwned,
{
    type Error = UnexpectedNumberOfDimensions;

    fn try_from(array: FlattenedArray<S::Elem>) -> Result<Self, Self::Error> {
        let shape = D::try_from(&array.shape)?;

        let flattend = ArrayBase::<S, Ix1>::from(array.data);
        let output = flattend.into_shape(shape);
        // This can only fail if the FlattenedArray invariant is violated, which
        // we do check when deserializing it!
        Ok(output.unwrap_or_else(|_| unreachable!()))
    }
}

/// Helper trait to allow us to create various `Dim` instances from a slice.
///
/// The serialization format for `Dim`,`ArrayBase` and similar is not fixed,
/// so we must deserialize it as a `Vec<usize>` (or similar) and then convert
/// it. But `ndarray` only ships with conversion methods from `Vec<Ix>`/`&[Ix]`
/// to `IxDyn` but not to the various specific dims.
pub(crate) trait TryIntoDimension: Sized {
    fn try_from(slice: &[Ix]) -> Result<Self, UnexpectedNumberOfDimensions>;
}

impl<const N: usize> TryIntoDimension for Dim<[Ix; N]>
where
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
{
    fn try_from(slice: &[Ix]) -> Result<Self, UnexpectedNumberOfDimensions> {
        <[Ix; N]>::try_from(slice)
            .map(IntoDimension::into_dimension)
            .map_err(|_| UnexpectedNumberOfDimensions {
                got: slice.len(),
                expected: N,
            })
    }
}

impl TryIntoDimension for IxDyn {
    fn try_from(slice: &[Ix]) -> Result<Self, UnexpectedNumberOfDimensions> {
        Ok(slice.into_dimension())
    }
}
#[derive(Serialize, Deserialize)]
#[cfg_attr(test, derive(Debug, Default, PartialEq))]
pub(crate) struct BinParams {
    params: HashMap<String, FlattenedArray<f32>>,
}

impl BinParams {
    #[cfg(test)]
    pub(crate) fn load_from_file(file: impl AsRef<Path>) -> Result<Self, LoadingBinParamsFailed> {
        let file = File::open(file)?;
        let source = BufReader::new(file);
        Self::load(source)
    }

    pub(crate) fn load(source: impl Read) -> Result<Self, LoadingBinParamsFailed> {
        let bincode = Self::setup_bincode();
        bincode.deserialize_from(source).map_err(Into::into)
    }

    fn setup_bincode() -> impl bincode::Options {
        // we explicitly set some default options to
        // convey exactly which options we use.
        bincode::DefaultOptions::new()
            .with_little_endian()
            .with_fixint_encoding()
            .reject_trailing_bytes()
    }

    /// True if this instance is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// List the keys contained in this instance.
    pub(crate) fn keys(&self) -> impl Iterator<Item = &str> {
        self.params.keys().map(|s| &**s)
    }

    pub(crate) fn take<A>(&mut self, name: &str) -> Result<A, FailedToRetrieveParams>
    where
        FlattenedArray<f32>: TryInto<A, Error = UnexpectedNumberOfDimensions>,
    {
        self.params
            .remove(name)
            .ok_or_else(|| FailedToRetrieveParams::MissingParameters {
                name: name.to_owned(),
            })?
            .try_into()
            .map_err(Into::into)
    }

    /// Creates a new `BinParamsWithScope` instance.
    ///
    /// The name prefix will be  scope + '/'. Passing a empty
    /// scope in is possible.
    pub(crate) fn with_scope<'b>(&'b mut self, scope: &str) -> BinParamsWithScope<'b> {
        BinParamsWithScope {
            params: self,
            prefix: scope.to_owned() + "/",
        }
    }
}

#[derive(Debug, Error)]
pub enum LoadingBinParamsFailed {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    DeserializationFailed(#[from] bincode::Error),
}

/// A wrapper embedding a prefix with the bin params.
//Note: In the future we might have some Loader trait but given
//that we currently only use it at one place that would be
//overkill
pub(crate) struct BinParamsWithScope<'a> {
    params: &'a mut BinParams,
    prefix: String,
}

impl<'a> BinParamsWithScope<'a> {
    pub(crate) fn take<A>(&mut self, name: &str) -> Result<A, FailedToRetrieveParams>
    where
        FlattenedArray<f32>: TryInto<A, Error = UnexpectedNumberOfDimensions>,
    {
        let name = self.prefix.clone() + name;
        self.params.take(&name)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array1, Array2};

    use super::*;

    #[test]
    fn ix_is_usize() {
        let _a: Ix = 12usize;
    }

    #[rustfmt::skip]
    const BIN_PARAMS_MOCK_DATA_1: &[u8] = &[
        0x2,0x0,0x0,0x0,0x0,0x0,0x0,0x0, // map len 2
        0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x0, // string(key) len 1
        0x61, // "a"
        0x2,0x0,0x0,0x0,0x0,0x0,0x0,0x0, // dimensions len 2
        0x2,0x0,0x0,0x0,0x0,0x0,0x0,0x0, // first dim len 2
        0x2,0x0,0x0,0x0,0x0,0x0,0x0,0x0, // second dim len 2
        0x4,0x0,0x0,0x0,0x0,0x0,0x0,0x0, // data nr elements 4
        0x0,0x0,0x80,0x3f, // a f32
        0x0,0x0,0x0,0x40, // another f32
        0x0,0x0,0x40,0x40, // ...
        0x0,0x0,0x80,0x40, // ...
        0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x0, // string(key) len 1
        0x62, // "b"
        0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x0, // dims len 1
        0x4,0x0,0x0,0x0,0x0,0x0,0x0,0x0, // first dim len 4
        0x4,0x0,0x0,0x0,0x0,0x0,0x0,0x0, // data nr elements 4
        0x0,0x0,0x40,0x40, // a f32
        0x0,0x0,0x0,0x40, // another f32
        0x0,0x0,0x80,0x3f, // another one
        0x0,0x0,0x80,0x40 // another one
    ];

    fn bin_params_mock_outcome_1() -> BinParams {
        let mut params = HashMap::default();
        params.insert(
            "a".to_owned(),
            FlattenedArray::from(arr2(&[[1.0f32, 2.], [3., 4.]])),
        );
        params.insert(
            "b".to_owned(),
            FlattenedArray::from(arr1(&[3.0f32, 2., 1., 4.])),
        );
        BinParams { params }
    }

    #[test]
    fn bin_params_can_load_bin_params() {
        let loaded = BinParams::load(BIN_PARAMS_MOCK_DATA_1).unwrap();
        assert_eq!(loaded, bin_params_mock_outcome_1());
    }

    #[test]
    fn bin_params_can_load_arrays_of_specific_dimensions() {
        let mut loaded = BinParams::load(BIN_PARAMS_MOCK_DATA_1).unwrap();
        let array1 = loaded.take::<Array2<f32>>("a").unwrap();
        let array2 = loaded.take::<Array1<f32>>("b").unwrap();

        assert_eq!(array1, arr2(&[[1.0f32, 2.], [3., 4.]]));
        assert_eq!(array2, arr1(&[3.0f32, 2., 1., 4.]));
    }
}
