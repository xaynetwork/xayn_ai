//! This module contains utility for loading storing ndarray arrays

use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
    fs::File,
    io::{BufReader, BufWriter, Error as IoError, Read, Write},
    path::Path,
};

use bincode::{DefaultOptions, Error as BinError, ErrorKind, Options};
use displaydoc::Display;
use ndarray::{Array, ArrayBase, DataOwned, Dim, Dimension, IntoDimension, Ix, Ix1, IxDyn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::utils::IncompatibleMatrices;

/// Deserialization helper representing a flattened array.
///
/// The flattened array is in row-major order.
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(Default, PartialEq))]
pub struct FlattenedArray<A> {
    shape: Vec<Ix>,
    /// There is a invariant that the length of data is
    /// equal to the product of all values in shape.
    data: Vec<A>,
}

impl<A, D> From<Array<A, D>> for FlattenedArray<A>
where
    A: Copy,
    D: Dimension,
{
    fn from(array: Array<A, D>) -> Self {
        let shape = array.shape().to_owned();

        let data = if array.is_standard_layout() {
            array.into_raw_vec()
        } else {
            array.iter().copied().collect()
        };

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

/// Unexpected number of dimensions: got={got}, expected={expected}"
#[derive(Debug, Display, Error)]
pub struct UnexpectedNumberOfDimensions {
    got: usize,
    expected: usize,
}

/// Irretrievable parameters error.
#[derive(Debug, Display, Error)]
pub enum FailedToRetrieveParams {
    /// Unexpected dimensionality.
    #[displaydoc("{0}")]
    UnexpectedNumberOfDimensions(#[from] UnexpectedNumberOfDimensions),

    /// Missing parameters for {name}
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
pub trait TryIntoDimension: Sized {
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
#[derive(Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct BinParams {
    params: HashMap<String, FlattenedArray<f32>>,
}

impl BinParams {
    pub fn deserialize_from_file(file: impl AsRef<Path>) -> Result<Self, LoadingBinParamsFailed> {
        let file = File::open(file)?;
        let source = BufReader::new(file);
        Self::deserialize_from(source)
    }

    pub fn deserialize_from(source: impl Read) -> Result<Self, LoadingBinParamsFailed> {
        let bincode = Self::setup_bincode();
        bincode.deserialize_from(source).map_err(Into::into)
    }

    pub fn serialize_into_file(&self, path: impl AsRef<Path>) -> Result<(), Box<ErrorKind>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.serialize_into(writer)
    }

    pub fn serialize_into(&self, writer: impl Write) -> Result<(), Box<ErrorKind>> {
        let bincode = Self::setup_bincode();
        bincode.serialize_into(writer, self)
    }

    fn setup_bincode() -> impl Options {
        // we explicitly set some default options to
        // convey exactly which options we use.
        DefaultOptions::new()
            .with_little_endian()
            .with_fixint_encoding()
            .reject_trailing_bytes()
    }

    /// True if this instance is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// List the keys contained in this instance.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
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

    /// Insert an array of given name (replacing any array previously set for that name).
    pub(crate) fn insert(
        &mut self,
        name: impl Into<String>,
        array: impl Into<FlattenedArray<f32>>,
    ) {
        self.params.insert(name.into(), array.into());
    }

    /// Creates a new `BinParamsWithScope` instance.
    ///
    /// The name prefix will be  scope + '/'. Passing a empty
    /// scope in is possible.
    pub fn with_scope<'b>(&'b mut self, scope: &str) -> BinParamsWithScope<'b> {
        BinParamsWithScope {
            params: self,
            prefix: scope.to_owned() + "/",
        }
    }
}

impl IntoIterator for BinParams {
    type Item = (String, FlattenedArray<f32>);

    type IntoIter = BinParamsIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        BinParamsIntoIter(self.params.into_iter())
    }
}

pub struct BinParamsIntoIter(std::collections::hash_map::IntoIter<String, FlattenedArray<f32>>);

impl Iterator for BinParamsIntoIter {
    type Item = (String, FlattenedArray<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

/// Loading of binary parameters failed
#[derive(Debug, Display, Error)]
#[prefix_enum_doc_attributes]
pub enum LoadingBinParamsFailed {
    /// {0}
    Io(#[from] IoError),
    /// {0}
    DeserializationFailed(#[from] BinError),
}

/// A wrapper embedding a prefix with the bin params.
pub struct BinParamsWithScope<'a> {
    params: &'a mut BinParams,
    prefix: String,
}

impl BinParamsWithScope<'_> {
    pub fn take<A>(&mut self, name: &str) -> Result<A, FailedToRetrieveParams>
    where
        FlattenedArray<f32>: TryInto<A, Error = UnexpectedNumberOfDimensions>,
    {
        let name = self.create_name(name);
        self.params.take(&name)
    }

    /// Insert a array under given name combined with the current prefix (replacing any array previously set for that name).
    pub fn insert(&mut self, name: &str, array: impl Into<FlattenedArray<f32>>) {
        let name = self.create_name(name);
        self.params.insert(name, array);
    }

    /// Create a name based on a suffix and the current prefix.
    pub fn create_name(&self, suffix: &str) -> String {
        self.prefix.clone() + suffix
    }

    pub fn with_scope(&mut self, scope: &str) -> BinParamsWithScope {
        BinParamsWithScope {
            params: &mut *self.params,
            prefix: self.prefix.clone() + scope + "/",
        }
    }
}

/// Failed to load the layer
#[derive(Debug, Display, Error)]
#[prefix_enum_doc_attributes]
pub enum LoadingLayerFailed {
    /// Incompatible matrices.
    #[displaydoc("{0}")]
    IncompatibleMatrices(#[from] IncompatibleMatrices),

    /// Mismatched dimensions.
    #[displaydoc("{0}")]
    DimensionMismatch(#[from] UnexpectedNumberOfDimensions),

    /// Irretrievable parameters.
    #[displaydoc("{0}")]
    FailedToRetrieveParams(#[from] FailedToRetrieveParams),

    /// Some parameters are invalid (e.g. nan, infinite, etc.)
    InvalidParams,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use ndarray::{arr1, arr2, Array1, Array2};
    use rand::{thread_rng, Rng};

    use crate::utils::he_normal_weights_init;
    use test_utils::assert_approx_eq;

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
        let loaded = BinParams::deserialize_from(BIN_PARAMS_MOCK_DATA_1).unwrap();
        assert_eq!(loaded, bin_params_mock_outcome_1());
    }

    #[test]
    fn bin_params_can_load_arrays_of_specific_dimensions() {
        let mut loaded = BinParams::deserialize_from(BIN_PARAMS_MOCK_DATA_1).unwrap();
        let array1 = loaded.take::<Array2<f32>>("a").unwrap();
        let array2 = loaded.take::<Array1<f32>>("b").unwrap();

        assert_eq!(array1, arr2(&[[1.0f32, 2.], [3., 4.]]));
        assert_eq!(array2, arr1(&[3.0f32, 2., 1., 4.]));
    }

    #[test]
    fn serialize_deserialize_random_bin_params() {
        let mut rng = thread_rng();
        for nr_params in 0..4 {
            let mut bin_params = BinParams::default();

            let name = format!("{}", nr_params);
            let nr_rows = rng.gen_range(0..100);
            let nr_columns = rng.gen_range(0..100);
            let matrix = he_normal_weights_init(&mut rng, (nr_rows, nr_columns));

            bin_params.params.insert(name, matrix.into());

            let mut buffer = Vec::new();
            bin_params.serialize_into(&mut buffer).unwrap();
            let bin_params2 = BinParams::deserialize_from(&*buffer).unwrap();

            for (key, fla1) in bin_params.params.iter() {
                let fla2 = bin_params2.params.get(key).unwrap();
                assert_eq!(fla1.shape, fla2.shape);
                assert_approx_eq!(f32, &fla1.data, &fla2.data, ulps = 0);
            }
        }
    }

    const EMPTY_BIN_PARAMS: &[u8] = &[0u8; 8];

    const BIN_PARAMS_WITH_EMPTY_ARRAY_AND_KEY: &[u8] = &[
        1u8, 0, 0, 0, 0, 0, 0, 0, // 1 entry
        0, 0, 0, 0, 0, 0, 0, 0, // empty string key
        1, 0, 0, 0, 0, 0, 0, 0, // 1 dimensional array
        0, 0, 0, 0, 0, 0, 0, 0, // the dimension is 0
        0, 0, 0, 0, 0, 0, 0, 0, // and we have 0 bytes of data data
    ];

    const BIN_PARAMS_WITH_SOME_KEYS: &[u8] = &[
        2u8, 0, 0, 0, 0, 0, 0, 0, // 2 entries
        3, 0, 0, 0, 0, 0, 0, 0, b'f', b'o', b'o', // key 1
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // shape [0]
        0, 0, 0, 0, 0, 0, 0, 0, // and data
        3, 0, 0, 0, 0, 0, 0, 0, b'b', b'a', b'r', // key 2
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // shape [0]
        0, 0, 0, 0, 0, 0, 0, 0, // and data
    ];

    #[test]
    fn test_bin_params_is_empty() {
        let params = BinParams::deserialize_from(EMPTY_BIN_PARAMS).unwrap();
        assert!(params.is_empty());

        let mut params = BinParams::deserialize_from(BIN_PARAMS_WITH_EMPTY_ARRAY_AND_KEY).unwrap();
        assert!(!params.is_empty());

        let array: Array<f32, IxDyn> = params.take("").unwrap();
        assert_eq!(array.shape(), &[0]);
    }

    #[test]
    fn test_bin_params_keys() {
        let params = BinParams::deserialize_from(BIN_PARAMS_WITH_SOME_KEYS).unwrap();
        let mut expected = HashSet::default();
        expected.insert("foo");
        expected.insert("bar");
        assert_eq!(params.keys().collect::<HashSet<_>>(), expected);
    }
}
