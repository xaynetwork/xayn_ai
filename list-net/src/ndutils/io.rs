//! This module contains utility for loading storing ndarray arrays
use bincode::Options;
use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};
use thiserror::Error;

use ndarray::{ArrayBase, Data, DataOwned, Dim, Dimension, IntoDimension, Ix, Ix1, IxDyn};
use serde::{Deserialize, Serialize};
/// Deserialization helper representing a flattened array.
///
/// The flattened array is in `C` format, i.e. it's
/// row first (instead of column first, `F` format).
#[derive(Serialize)]
#[serde(transparent)]
#[cfg_attr(test, derive(Debug, Default, PartialEq))]
pub(crate) struct FlattenedArray<A> {
    inner: InnerFlattenedArray<A>,
}

/// Helper to get a post serialization invariant check.
#[derive(Serialize, Deserialize)]
#[cfg_attr(test, derive(Debug, Default, PartialEq))]
struct InnerFlattenedArray<A> {
    shape: Vec<Ix>,
    /// There is a invariant that the length of data is
    /// equal to the product of all values in shape
    data: Vec<A>,
}

impl<S, D> From<ArrayBase<S, D>> for FlattenedArray<S::Elem>
where
    S: Data,
    S::Elem: Clone,
    D: Dimension,
{
    fn from(array: ArrayBase<S, D>) -> Self {
        let shape = array.shape().to_owned();
        let n_elements = array.len();
        // if we would know array is not a view and in memory
        // order we could use into_raw_vec...
        let data = array.into_shape((n_elements,)).unwrap().to_vec();

        FlattenedArray {
            inner: InnerFlattenedArray { shape, data },
        }
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
        let inner = InnerFlattenedArray::<A>::deserialize(deserializer)?;
        if inner.data.len() != inner.shape.iter().product::<usize>() {
            Err(<D::Error as serde::de::Error>::custom(
                UnexpectedNumberOfDimensions,
            ))
        } else {
            Ok(Self { inner })
        }
    }
}

#[derive(Debug, Error)]
#[error("Expected and found number of dimensions do not match")]
pub struct UnexpectedNumberOfDimensions;

#[derive(Debug, Error)]
pub enum FailedToRetrieveParams {
    #[error(transparent)]
    UnexpectedNumberOfDimensions(#[from] UnexpectedNumberOfDimensions),

    #[error("Missing parameters for {name}.")]
    MissingParameters { name: String },
}

impl<S, D> TryFrom<FlattenedArray<S::Elem>> for ArrayBase<S, D>
where
    D: Dimension + DimensionTryFromSliceHelper,
    S: DataOwned,
{
    type Error = UnexpectedNumberOfDimensions;

    fn try_from(array: FlattenedArray<S::Elem>) -> Result<Self, Self::Error> {
        let shape = D::try_from(&array.inner.shape)?;

        let flattend = ArrayBase::<S, Ix1>::from(array.inner.data);
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
pub(crate) trait DimensionTryFromSliceHelper: Sized {
    fn try_from(slice: &[Ix]) -> Result<Self, UnexpectedNumberOfDimensions>;
}

impl<const N: usize> DimensionTryFromSliceHelper for Dim<[Ix; N]>
where
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
{
    fn try_from(slice: &[Ix]) -> Result<Self, UnexpectedNumberOfDimensions> {
        <[Ix; N]>::try_from(slice)
            .map(IntoDimension::into_dimension)
            .map_err(|_| UnexpectedNumberOfDimensions)
    }
}

impl DimensionTryFromSliceHelper for IxDyn {
    fn try_from(slice: &[Ix]) -> Result<Self, UnexpectedNumberOfDimensions> {
        Ok(slice.into_dimension())
    }
}
#[derive(Serialize, Deserialize)]
#[cfg_attr(test, derive(Debug, Default, PartialEq))]
pub(crate) struct BinParams {
    // for now limited to f32
    params: HashMap<String, FlattenedArray<f32>>,
}

impl BinParams {
    pub(crate) fn load_from_file(file: impl AsRef<Path>) -> Result<Self, LoadingBinParamsFailed> {
        let file = File::open(file)?;
        let source = BufReader::new(file);
        Self::load(source)
    }

    pub(crate) fn load(mut source: impl Read) -> Result<Self, LoadingBinParamsFailed> {
        Self::load_check_version(&mut source)?;
        let bincode = Self::setup_bincode();
        Ok(bincode.deserialize_from(source)?)
    }

    fn setup_bincode() -> impl bincode::Options {
        // we explicitly set some default options to
        // convey exactly which options we use.
        bincode::DefaultOptions::new()
            .with_little_endian()
            // 500MiB input limit,
            // way bigger then we could ever use
            // (as we run on phones)
            .with_limit(500 * 1024 * 1024)
            .with_fixint_encoding()
            .reject_trailing_bytes()
    }

    fn load_check_version(source: &mut impl Read) -> Result<(), LoadingBinParamsFailed> {
        let mut version_buf = [0u8; 1];
        source.read_exact(&mut version_buf)?;

        if version_buf[0] == 0x1 {
            Ok(())
        } else {
            Err(LoadingBinParamsFailed::UnsupportedVersion {
                version: version_buf[0],
            })
        }
    }

    pub(crate) fn take<A>(&mut self, name: &str) -> Result<A, FailedToRetrieveParams>
    where
        FlattenedArray<f32>: TryInto<A, Error = UnexpectedNumberOfDimensions>,
    {
        let result = self
            .params
            .remove(name)
            .ok_or_else(|| FailedToRetrieveParams::MissingParameters {
                name: name.to_owned(),
            })?
            .try_into()?;

        Ok(result)
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

    #[error("Loading failed: Unsupported version: {version}")]
    UnsupportedVersion { version: u8 },

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

    /// Returns a instance where the nem prefix is extended by the given scope.
    ///
    /// The new prefix is the old prefix + the scope + '/'.
    pub(crate) fn with_scope<'b>(&'b mut self, scope: &str) -> BinParamsWithScope<'b> {
        BinParamsWithScope {
            params: self.params,
            prefix: format!("{}{}/", self.prefix, scope),
        }
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
        0x1, // Version
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
