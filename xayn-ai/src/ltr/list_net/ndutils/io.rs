//! This module contains utility for loading storing ndarray arrays

use bincode::Options;
use std::{
    collections::HashMap,
    convert::{TryFrom, TryInto},
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    path::Path,
};

use thiserror::Error;

use ndarray::{Array, ArrayBase, DataOwned, Dim, Dimension, IntoDimension, Ix, Ix1, IxDyn};
use serde::{Deserialize, Serialize};

/// Deserialization helper representing a flattened array.
///
/// The flattened array is in row-major order.
#[derive(Serialize)]
#[cfg_attr(test, derive(Debug, Default, PartialEq))]
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
#[derive(Default, Serialize, Deserialize)]
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

    pub fn serialize_into_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<bincode::ErrorKind>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        self.serialize_into(writer)
    }

    pub fn serialize_into(&self, writer: impl Write) -> Result<(), Box<bincode::ErrorKind>> {
        let bincode = Self::setup_bincode();
        bincode.serialize_into(writer, self)
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
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// List the keys contained in this instance.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.params.keys().map(|s| &**s)
    }

    pub fn take<A>(&mut self, name: &str) -> Result<A, FailedToRetrieveParams>
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

    /// Insert a array under given name (replacing any array previously set for that name).
    pub fn insert<A>(&mut self, name: impl Into<String>, array: A)
    where
        FlattenedArray<f32>: From<A>,
    {
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
pub struct BinParamsWithScope<'a> {
    params: &'a mut BinParams,
    prefix: String,
}

impl<'a> BinParamsWithScope<'a> {
    pub fn take<A>(&mut self, name: &str) -> Result<A, FailedToRetrieveParams>
    where
        FlattenedArray<f32>: TryInto<A, Error = UnexpectedNumberOfDimensions>,
    {
        let name = self.create_name(name);
        self.params.take(&name)
    }

    /// Insert a array under given name combined with the current prefix (replacing any array previously set for that name).
    pub fn insert<A>(&mut self, name: &str, array: A)
    where
        FlattenedArray<f32>: From<A>,
    {
        let name = self.create_name(name);
        self.params.insert(name, array);
    }

    /// Create a name based on a suffix and the current prefix.
    fn create_name(&self, suffix: &str) -> String {
        self.prefix.clone() + suffix
    }

    #[cfg(test)]
    pub fn with_scope(&mut self, scope: &str) -> BinParamsWithScope {
        BinParamsWithScope {
            params: &mut *self.params,
            prefix: self.prefix.clone() + scope + "/",
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array1, Array2};
    use rand::{thread_rng, Rng};

    use super::super::he_normal_weights_init;

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
}
