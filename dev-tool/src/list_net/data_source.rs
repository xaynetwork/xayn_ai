#![cfg(not(tarpaulin))]

//FIXME[follow up PR]: Move modified parts of this module into the `ltr::list_net` module to re-use them for in-app training.
use std::{
    error::Error as StdError,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    u64,
};

use anyhow::{bail, Error};
use bincode::Options;
use displaydoc::Display;
use itertools::Itertools;
use log::debug;
use ndarray::{Array1, Array2, ArrayBase, ArrayView, Data, Dimension};
use rand::{prelude::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use xayn_ai::list_net::{self, ListNet, SampleOwned, SampleView};

/// A [`xayn_ai::list_net::DataSource`] implementation.
pub(crate) struct DataSource<S>
where
    S: Storage,
{
    /// The storage containing all samples.
    storage: S,
    /// The batch size.
    batch_size: usize,
    /// A container with all ids of all training samples, returning them in randomized order.
    training_data_order: DataLookupOrder,
    /// A container with all ids of all evaluation samples, returning them in randomized order.
    evaluation_data_order: DataLookupOrder,
}

impl<S> DataSource<S>
where
    S: Storage,
{
    /// Creates a new `DataSource` from given storage and evaluation split.
    ///
    /// # Errors
    ///
    /// - If `evaluation_split` is less than 0, greater then 1.0 or not a normal float.
    /// - If there are no training samples with the given `evaluation_split`, which is not exactly 1.0.
    /// - If there are no evaluation samples with the given `evaluation_split`, which is not exactly 0.0.
    ///   but the split is not `0`.
    /// - If the `batch_size` is larger then the number of training samples.
    pub(crate) fn new(
        storage: S,
        evaluation_split: f32,
        mut batch_size: usize,
    ) -> Result<Self, DataSourceError<S::Error>> {
        let nr_all_samples = storage.number_of_samples();
        let (nr_training_samples, _nr_evaluation_samples) =
            Self::calculate_evaluation_split(evaluation_split, nr_all_samples)?;

        if batch_size > nr_training_samples {
            return Err(DataSourceError::TooLargeBatchSize {
                batch_size,
                nr_training_samples,
            });
        } else if batch_size == 0 {
            batch_size = nr_training_samples;
        }
        let evaluation_ids = (nr_training_samples..nr_all_samples).collect();
        let training_ids = (0..nr_training_samples).collect();

        Ok(Self {
            storage,
            batch_size,
            training_data_order: DataLookupOrder::new(training_ids),
            evaluation_data_order: DataLookupOrder::new(evaluation_ids),
        })
    }

    /// Create multiple `DataSources` by splitting the storage into chunks.
    ///
    /// The `DataSources` in the `Vec` will only have training samples and
    /// the additional `DataSource` only contains evaluation samples.
    ///
    /// This will first split the storage and then for each chunk create
    /// a `DataSource`.
    ///
    /// # Errors
    ///
    /// Has the same errors as [`DataSource.new()`].
    #[allow(dead_code, clippy::type_complexity)]
    pub(crate) fn new_split(
        storage: S,
        evaluation_split: f32,
        batch_size: usize,
        chunk_size: usize,
    ) -> Result<(Vec<Self>, Option<Self>), DataSourceError<S::Error>> {
        let (_, nr_evaluation_samples) =
            Self::calculate_evaluation_split(evaluation_split, storage.number_of_samples())?;

        let (training_chunks, eval_storage) =
            storage.split_storage_into_chunks(chunk_size, batch_size, nr_evaluation_samples)?;

        let training_data_sources = training_chunks
            .into_iter()
            .map(|storage| Self::new(storage, 0.0, batch_size))
            .collect::<Result<_, _>>()?;

        let eval_data_source = (eval_storage.number_of_samples() > 0)
            .then(|| Self::new(eval_storage, 1.0, 0))
            .transpose()?;
        Ok((training_data_sources, eval_data_source))
    }

    /// Calculates `(nr_training_samples, nr_evaluation_samples)` and checks for consistency/invariants.
    fn calculate_evaluation_split(
        evaluation_split: f32,
        nr_all_samples: usize,
    ) -> Result<(usize, usize), DataSourceError<S::Error>> {
        if nr_all_samples == 0 {
            return Err(DataSourceError::EmptyDatabase);
        }
        if !(0.0..=1.0).contains(&evaluation_split)
            || (evaluation_split != 0.0 && !evaluation_split.is_normal())
        {
            return Err(DataSourceError::BadEvaluationSplit(evaluation_split));
        }
        let nr_evaluation_samples = (nr_all_samples as f32 * evaluation_split).round() as usize;
        if nr_evaluation_samples == nr_all_samples && evaluation_split < 1. {
            return Err(DataSourceError::TooLargeEvaluationSplit(evaluation_split));
        }
        if nr_all_samples > 0 && nr_evaluation_samples == 0 && evaluation_split > 0. {
            return Err(DataSourceError::NoEvaluationSamples(evaluation_split));
        }
        let nr_training_samples = nr_all_samples - nr_evaluation_samples;
        Ok((nr_training_samples, nr_evaluation_samples))
    }
}

#[derive(Error, Debug, Display)]
pub enum DataSourceError<SE>
where
    SE: StdError + 'static,
{
    /// Unusable evaluation split: Assertion `split >= 0 && split.is_normal()` failed (split = {0}).
    BadEvaluationSplit(f32),
    /// Unusable evaluation split: With evaluation split {0} no samples are left for training.
    TooLargeEvaluationSplit(f32),
    /// Unusable evaluation split: Assertion `nr_evaluation_samples > 0 || split == 0` failed (split = {0}).
    NoEvaluationSamples(f32),
    /// The batch size ({batch_size}) is larger than the number of samples ({nr_training_samples}).
    TooLargeBatchSize {
        batch_size: usize,
        nr_training_samples: usize,
    },
    /// Empty database cannot be used for training.
    EmptyDatabase,
    /// Fetching sample from storage failed: {0}.
    Storage(#[from] SE),
}

impl<S> list_net::DataSource for DataSource<S>
where
    S: Storage,
{
    type Error = DataSourceError<S::Error>;

    fn reset(&mut self) -> Result<(), Self::Error> {
        let mut rng = rand::thread_rng();
        self.training_data_order.reset(&mut rng);
        self.evaluation_data_order.reset(&mut rng);
        Ok(())
    }

    fn number_of_training_batches(&self) -> usize {
        self.training_data_order.number_of_batches(self.batch_size)
    }

    fn next_training_batch(&mut self) -> Result<Vec<SampleView>, Self::Error> {
        let ids = self.training_data_order.next_batch(self.batch_size);
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        self.storage
            .load_batch(&ids)
            .map_err(DataSourceError::Storage)
    }

    fn number_of_evaluation_samples(&self) -> usize {
        self.evaluation_data_order.number_of_samples()
    }

    fn next_evaluation_sample(&mut self) -> Result<Option<SampleOwned>, Self::Error> {
        if let Some(id) = self.evaluation_data_order.next() {
            match self.storage.load_sample(id) {
                Ok(sample) => Ok(Some(sample.to_owned())),
                Err(error) => Err(DataSourceError::Storage(error)),
            }
        } else {
            Ok(None)
        }
    }
}

/// Abstraction over a storage of samples.
pub(crate) trait Storage: Send {
    type Error: StdError + 'static + Send;

    /// Splits this storage into multiple chunks.
    ///
    /// This will split the storage into multiple chunks by first splitting it
    /// into two segments. The first will be chunked and the second will be a
    /// single storage instance. The the second segment will have a size of
    /// `tail_segment_size` and the first a size of `nr_all_samples-tail_segment_size`.
    ///
    /// (The first is used for chunked training samples, the second for non-chunked
    /// evaluation samples.)
    ///
    /// The chunking of the first segment will be done by splitting it into
    /// chunks with a size of `chunk_size`, through the last chunk might
    /// have a smaller size. If a smaller last chunk has a size less then
    /// `min_remainder_size` it will be discarded.
    ///
    /// # Errors
    ///
    /// An error should be returned if:
    ///
    /// - `chunk_size` is `0`
    /// - `last_chunk_size` is greater then the number of all chunks
    /// - `min_remainder_size` is greater then `chunk_size`
    fn split_storage_into_chunks(
        self,
        chunk_size: usize,
        min_remainder_size: usize,
        tail_segment_size: usize,
    ) -> Result<(Vec<Self>, Self), Self::Error>
    where
        Self: Sized;

    /// Return all sample ids.
    ///
    /// For now this is a range. I.e. `0..nr_of_samples`. We might need
    /// to change this in the future, but for now this is simpler.
    fn number_of_samples(&self) -> usize;

    /// Loads a sample and returns a reference to it.
    ///
    /// This method is `&mut self` as it might change the internal state
    /// (due to loading/caching) and we also want to make sure that there
    /// are no more lend out samples, when `load_sample` or `load_batch` are
    /// called again.
    fn load_sample(&mut self, id: DataId) -> Result<SampleView, Self::Error>;

    /// Loads a batch of samples and returns a vector of references to them.
    ///
    /// See [`Storage.load_batch()`] about why this is `&mut self`.
    fn load_batch<'a>(&'a mut self, ids: &'_ [DataId]) -> Result<Vec<SampleView<'a>>, Self::Error>;
}

/// For now samples ids are always incremental integers from `0` to `nr_samples`.
///
/// - This is only used to handle shuffling and retrieving the samples from the
///   storage.
/// - This is unlikely to ever change.
/// - We do not care about the consistency of the id-to-sample mapping between
///   trainings.
///     - Though we do care about a deterministic evaluation split, as such
///       we currently need some degree of determinism in practice. But that is
///       an implementation detail of the splitting unrelated to the rest of
///       the code.
type DataId = usize;

/// Helper to randomize sample order.
struct DataLookupOrder {
    /// Ids of all samples in this lookup order.
    data_ids: Vec<DataId>,
    /// The offset into `data_ids` at which we should continue returning ids
    /// when `next`/`next_batch` is called.
    data_ids_offset: usize,
}

impl DataLookupOrder {
    fn new(data_ids: Vec<DataId>) -> Self {
        Self {
            data_ids,
            data_ids_offset: 0,
        }
    }

    /// Returns the total number of batches which will be returned with the given batch size.
    fn number_of_batches(&self, batch_size: usize) -> usize {
        self.data_ids.len() / batch_size
    }

    /// Returns the total number of samples which will be returned.
    fn number_of_samples(&self) -> usize {
        self.data_ids.len()
    }

    /// Resets this container.
    ///
    /// This will mark all samples as unused and
    /// shuffles the order of samples.
    fn reset(&mut self, rng: &mut impl Rng) {
        self.data_ids.shuffle(rng);
        self.data_ids_offset = 0;
    }

    /// Returns the next unused sample id.
    fn next(&mut self) -> Option<DataId> {
        if self.data_ids_offset < self.data_ids.len() {
            let idx = self.data_ids_offset;
            self.data_ids_offset += 1;

            Some(self.data_ids[idx])
        } else {
            None
        }
    }

    /// Returns the next `batch_size` many sample ids.
    fn next_batch(&mut self, batch_size: usize) -> Vec<DataId> {
        let end_idx = self.data_ids_offset + batch_size;
        if end_idx <= self.data_ids.len() {
            let start_idx = self.data_ids_offset;
            self.data_ids_offset = end_idx;

            self.data_ids[start_idx..end_idx].to_owned()
        } else {
            Vec::new()
        }
    }
}

/// An "in-memory" sample storage.
///
/// It can also be portably stored to disk using bincode serialization.
///
/// The storage is reasonably compact as ~5.3GiB of soundgarden user
/// data-frames will produce a less than 975MiB serialized `.samples` file.
// While there are many ways to improve on this (e.g. memory-mapped I/O), they are not relevant for our
// use-case for now.
#[derive(Serialize, Deserialize, Default)]
pub struct InMemorySamples {
    /// Vector of concatenated `inputs` and `target_prob_dist`, this slightly improves memory size and
    /// cache locality, we can derive the number of document from the vectors length as the vectors length
    /// is `nr_document * INPUT_NR_FEATURES + nr_document*1`, i.e. `nr_documents * 51`.
    data: Vec<Vec<f32>>,
}

#[derive(Error, Debug, Display)]
pub enum StorageError {
    /// The database was parsed successfully but contains broken invariants at index {at_index}.
    BrokenInvariants { at_index: usize },

    /// The `chunk_size` used to split the storage was `0`.
    ChunkSize0,

    /// The `min_remainder_size` ({min_remainder_size}) used to split the storage was larger then `chunk_size` ({chunk_size}).
    ToLargeMinRemainderSize {
        min_remainder_size: usize,
        chunk_size: usize,
    },

    /// The `tail_segment_size` ({tail_segment_size}) was larger then the total number of samples ({nr_all_samples}).
    ToLargeExtraChunkSize {
        tail_segment_size: usize,
        nr_all_samples: usize,
    },
}

impl InMemorySamples {
    /// Creates a new storage which prepares for a specific number of samples to be added.
    pub fn with_sample_capacity(nr_samples: usize) -> Self {
        Self {
            data: Vec::with_capacity(nr_samples),
        }
    }

    /// Serializes this instance into a file, preferably using the `.samples` file ending.
    //FIXME[follow-up PR] version the file format, it's used to persist data and it's not
    //                    unlikely to slightly change in the future.
    pub fn serialize_into_file(&self, file: impl AsRef<Path>) -> Result<(), Error> {
        self.serialize_into(BufWriter::new(File::create(file)?))
    }

    /// Serializes this instance into the given writer.
    fn serialize_into(&self, writer: impl Write) -> Result<(), Error> {
        bincode::DefaultOptions::new()
            .serialize_into(writer, self)
            .map_err(Into::into)
    }

    /// Deserialize an instance from the given file.
    pub fn deserialize_from_file(file: impl AsRef<Path>) -> Result<Self, Error> {
        Self::deserialize_from(BufReader::new(File::open(file)?))
    }

    /// Deserialize an instance from the given (preferably buffered) reader.
    fn deserialize_from(reader: impl Read) -> Result<Self, Error> {
        let self_: Self = bincode::DefaultOptions::new().deserialize_from(reader)?;
        debug!("Loaded {} samples.", self_.data.len());
        Ok(self_)
    }

    fn load_sample_helper(&self, id: DataId) -> Result<SampleView, StorageError> {
        let raw = &self.data[id];

        // len == nr_document * nr_features + nr_documents * 1
        let nr_documents = raw.len() / (ListNet::INPUT_NR_FEATURES + 1);
        let start_of_target_prob_dist = nr_documents * ListNet::INPUT_NR_FEATURES;
        debug_assert_eq!(start_of_target_prob_dist + nr_documents, raw.len());

        let inputs = ArrayView::from_shape(
            (nr_documents, ListNet::INPUT_NR_FEATURES),
            &raw[..start_of_target_prob_dist],
        )
        .map_err(|_| StorageError::BrokenInvariants { at_index: id })?;
        let target_prob_dist =
            ArrayView::from_shape((nr_documents,), &raw[start_of_target_prob_dist..])
                .map_err(|_| StorageError::BrokenInvariants { at_index: id })?;

        Ok(SampleView {
            inputs,
            target_prob_dist,
        })
    }

    /// Prepares samples to be added to this storage.
    ///
    /// The result should then be passed to [`DataSource.add_prepared_samples()`].
    ///
    /// The reason this is not a single function is that it allows the preparation
    /// of the samples to be part of the parallel pipeline creating the samples.
    ///
    /// # Errors
    ///
    /// This will return an error if for any sample:
    ///
    /// - The number of documents in `inputs` doesn't match the number of probabilities in
    /// `target_prob_dist`.
    /// - The the number of features per document is not equal to `[ListNet::INPUT_NR_FEATURES]`.
    pub fn prepare_samples(
        samples: impl IntoIterator<Item = (Array2<f32>, Array1<f32>)>,
    ) -> Result<PreparedSamples, Error> {
        let samples = samples
            .into_iter()
            .map(Self::sample_to_combined_vec)
            .collect::<Result<_, _>>()?;

        Ok(PreparedSamples { samples })
    }

    /// Adds all prepared samples to this storage.
    pub fn add_prepared_samples(&mut self, prepared_samples: PreparedSamples) {
        self.data.extend(prepared_samples.samples)
    }

    /// Adds a single non-prepared sample
    #[cfg(test)]
    pub fn add_sample(
        &mut self,
        inputs: Array2<f32>,
        target_prob_dist: Array1<f32>,
    ) -> Result<(), Error> {
        let prepared_samples = Self::prepare_samples(std::iter::once((inputs, target_prob_dist)))?;
        self.add_prepared_samples(prepared_samples);
        Ok(())
    }

    fn sample_to_combined_vec(
        (inputs, target_prob_dist): (Array2<f32>, Array1<f32>),
    ) -> Result<Vec<f32>, Error> {
        if inputs.shape() != [target_prob_dist.len(), ListNet::INPUT_NR_FEATURES] {
            bail!("Sample with bad array shapes. Expected shapes [{nr_docs}, {nr_feats}] & [{nr_docs}] but got shapes {inputs_shape:?} & {prob_dist_shape:?}.",
                nr_docs=inputs.shape()[0],
                nr_feats=ListNet::INPUT_NR_FEATURES,
                inputs_shape=inputs.shape(),
                prob_dist_shape=target_prob_dist.shape(),
            );
        }
        let mut data = Vec::with_capacity(inputs.len() + target_prob_dist.len());
        extend_vec_with_ndarray(&mut data, inputs);
        extend_vec_with_ndarray(&mut data, target_prob_dist);
        Ok(data)
    }
}

pub struct PreparedSamples {
    samples: Vec<Vec<f32>>,
}

/// Extend a vec with the elements of the given array.
///
/// Elements are added in logical order independent of storage order,
/// i.e. elements are added as if `data` was continuous and in standard
/// storage order (terms are used the way they are defined by ndarray).
fn extend_vec_with_ndarray<T: Clone>(
    data: &mut Vec<T>,
    array: ArrayBase<impl Data<Elem = T>, impl Dimension>,
) {
    if let Some(slice) = array.as_slice() {
        data.extend_from_slice(slice);
    } else {
        data.extend(array.iter().cloned());
    }
}

impl Storage for InMemorySamples {
    type Error = StorageError;

    fn number_of_samples(&self) -> usize {
        self.data.len()
    }

    fn load_sample(&mut self, id: DataId) -> Result<SampleView, Self::Error> {
        self.load_sample_helper(id)
    }

    fn load_batch<'a>(&'a mut self, ids: &'_ [DataId]) -> Result<Vec<SampleView<'a>>, Self::Error> {
        let mut samples = Vec::new();
        for id in ids {
            samples.push(self.load_sample_helper(*id)?)
        }
        Ok(samples)
    }

    fn split_storage_into_chunks(
        self,
        chunk_size: usize,
        min_remainder_size: usize,
        tail_segment_size: usize,
    ) -> Result<(Vec<Self>, Self), Self::Error>
    where
        Self: Sized,
    {
        let Self { mut data } = self;
        let nr_all_samples = data.len();

        if chunk_size == 0 {
            return Err(StorageError::ChunkSize0);
        }
        if min_remainder_size > chunk_size {
            return Err(StorageError::ToLargeMinRemainderSize {
                min_remainder_size,
                chunk_size,
            });
        }
        if tail_segment_size > nr_all_samples {
            return Err(StorageError::ToLargeExtraChunkSize {
                tail_segment_size,
                nr_all_samples,
            });
        }

        let nr_training_samples = nr_all_samples - tail_segment_size;
        let last_chunk = InMemorySamples {
            data: data.split_off(nr_training_samples),
        };
        let chunks = data
            .into_iter()
            .chunks(chunk_size)
            .into_iter()
            .filter_map(|chunk| {
                let data = chunk.collect_vec();
                let number_of_samples = data.len();
                (number_of_samples == chunk_size || number_of_samples >= min_remainder_size)
                    .then(|| InMemorySamples { data })
            })
            .collect_vec();

        Ok((chunks, last_chunk))
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use ndarray::Array;
    use rand::{prelude::StdRng, SeedableRng};
    use xayn_ai::{assert_approx_eq, list_net::DataSource as _};

    use super::*;

    fn dummy_storage() -> InMemorySamples {
        let mut storage = InMemorySamples::default();

        let inputs = Array::zeros((10, 50));
        let target_prob_dist = Array::ones((10,));
        storage.add_sample(inputs, target_prob_dist).unwrap();

        let inputs = Array::ones((10, 50));
        let target_prob_dist = Array::zeros((10,));
        storage.add_sample(inputs, target_prob_dist).unwrap();

        let inputs = Array::from_elem((10, 50), 4.);
        let target_prob_dist = Array::ones((10,));
        storage.add_sample(inputs, target_prob_dist).unwrap();

        storage
    }

    #[test]
    fn test_adding_bad_matrices_fails() {
        let mut storage = InMemorySamples::default();
        let inputs = Array::zeros((10, 48));
        let target_prob_dist = Array::ones((10,));
        let err = storage.add_sample(inputs, target_prob_dist).unwrap_err();

        assert_eq!(&format!("{}", err), "Sample with bad array shapes. Expected shapes [10, 50] & [10] but got shapes [10, 48] & [10]." );

        let inputs = Array::zeros((8, 50));
        let target_prob_dist = Array::ones((11,));
        let err = storage.add_sample(inputs, target_prob_dist).unwrap_err();

        assert_eq!(&format!("{}", err), "Sample with bad array shapes. Expected shapes [8, 50] & [8] but got shapes [8, 50] & [11]." );

        assert!(storage.data.is_empty());
    }

    #[test]
    fn test_runtime_representation_of_in_memory_storage() {
        let storage = dummy_storage();

        assert_eq!(storage.data.len(), 3);

        for f in &storage.data[0][..500] {
            assert_approx_eq!(f32, f, 0.0, ulps = 0);
        }
        for f in &storage.data[0][500..] {
            assert_approx_eq!(f32, f, 1.0, ulps = 0);
        }
        assert_eq!(storage.data[0].len(), 510);

        for f in &storage.data[1][..500] {
            assert_approx_eq!(f32, f, 1.0, ulps = 0);
        }
        for f in &storage.data[1][500..] {
            assert_approx_eq!(f32, f, 0.0, ulps = 0);
        }
        assert_eq!(storage.data[1].len(), 510);

        for f in &storage.data[2][..500] {
            assert_approx_eq!(f32, f, 4.0, ulps = 0);
        }
        for f in &storage.data[2][500..] {
            assert_approx_eq!(f32, f, 1.0, ulps = 0);
        }
        assert_eq!(storage.data[2].len(), 510);
    }

    #[test]
    fn test_get_samples_from_in_memory_storage() {
        let mut storage = dummy_storage();

        {
            let sample = storage.load_sample(0).unwrap();
            assert_eq!(sample.inputs.shape(), &[10, 50]);
            for f in sample.inputs.iter() {
                assert_approx_eq!(f32, f, 0.0, ulps = 0);
            }
            assert_eq!(sample.target_prob_dist.shape(), &[10]);
            for f in sample.target_prob_dist.iter() {
                assert_approx_eq!(f32, f, 1.0, ulps = 0);
            }
        }
        {
            let sample = storage.load_sample(1).unwrap();
            assert_eq!(sample.inputs.shape(), &[10, 50]);
            for f in sample.inputs.iter() {
                assert_approx_eq!(f32, f, 1.0, ulps = 0);
            }
            assert_eq!(sample.target_prob_dist.shape(), &[10]);
            for f in sample.target_prob_dist.iter() {
                assert_approx_eq!(f32, f, 0.0, ulps = 0);
            }
        }
        {
            let sample = storage.load_sample(2).unwrap();
            assert_eq!(sample.inputs.shape(), &[10, 50]);
            for f in sample.inputs.iter() {
                assert_approx_eq!(f32, f, 4.0, ulps = 0);
            }
            assert_eq!(sample.target_prob_dist.shape(), &[10]);
            for f in sample.target_prob_dist.iter() {
                assert_approx_eq!(f32, f, 1.0, ulps = 0);
            }
        }
    }

    #[test]
    fn test_get_batch_from_in_memory_storage() {
        let mut storage = dummy_storage();

        assert_eq!(storage.load_batch(&[]).unwrap().len(), 0);

        {
            let samples = storage.load_batch(&[1, 1, 0, 1]).unwrap();
            assert_eq!(samples.len(), 4);

            assert_approx_eq!(f32, &samples[0].inputs, &samples[1].inputs, ulps = 0);
            assert_approx_eq!(
                f32,
                &samples[0].target_prob_dist,
                &samples[1].target_prob_dist,
                ulps = 0
            );
            assert_approx_eq!(f32, &samples[0].inputs, &samples[3].inputs, ulps = 0);
            assert_approx_eq!(
                f32,
                &samples[0].target_prob_dist,
                &samples[3].target_prob_dist,
                ulps = 0
            );

            assert_approx_eq!(f32, samples[0].inputs[[0, 0]], 1.0, ulps = 0);
            assert_approx_eq!(f32, samples[2].inputs[[0, 0]], 0.0, ulps = 0);
        }
    }

    #[test]
    fn test_serialization_of_in_memory_storage_works() {
        let storage = dummy_storage();

        let mut buffer = Vec::new();
        storage.serialize_into(&mut buffer).unwrap();
        let storage2 = InMemorySamples::deserialize_from(&*buffer).unwrap();

        assert_approx_eq!(f32, storage.data, storage2.data);
    }

    #[test]
    fn test_data_lookup_order_changes_on_reset() {
        // If we don't seed it, than the test might randomly fail as
        // the two shuffles could randomly yield the same result. The
        // `Rng` algorithm might change with updates to "rand" if this
        // happens this test could still fail, but it would be reproducible.
        let mut rng = StdRng::from_seed([2u8; 32]);
        let mut dlo = DataLookupOrder::new((0..40).collect_vec());

        dlo.reset(&mut rng);
        let all = dlo.next_batch(5);
        dlo.reset(&mut rng);
        let all2 = dlo.next_batch(5);
        assert_ne!(all, all2);
    }

    #[test]
    fn test_data_lookup_order_does_not_return_partial_batches() {
        let mut dlo = DataLookupOrder::new(vec![0, 1, 2, 3, 4]);
        assert_eq!(dlo.next_batch(2), [0, 1]);
        assert_eq!(dlo.next_batch(2), [2, 3]);
        assert!(dlo.next_batch(2).is_empty());
        assert!(dlo.next_batch(2).is_empty());
    }

    #[test]
    fn test_data_lookup_order_returns_samples_in_order() {
        let mut dlo = DataLookupOrder::new(vec![0, 1, 4]);
        assert_eq!(dlo.next(), Some(0));
        assert_eq!(dlo.next(), Some(1));
        assert_eq!(dlo.next(), Some(4));
        assert_eq!(dlo.next(), None);
        assert_eq!(dlo.next(), None);
    }

    fn mock_storage() -> (InMemorySamples, usize) {
        let multiplier = ListNet::INPUT_NR_FEATURES + 1;
        let storage = InMemorySamples {
            // Invariant: len= (INPUT_NR_FEATURES+1)*nr_document
            data: vec![
                vec![0.0; multiplier * 10],
                vec![1.2; multiplier * 3],
                vec![2.0; multiplier],
                vec![3.0; multiplier * 12],
                vec![4.0; multiplier * 10],
                vec![5.0; multiplier * 10],
                vec![6.0; multiplier * 10],
                vec![7.0; multiplier * 10],
                vec![4.8; multiplier * 10],
                vec![9.0; multiplier * 6],
                vec![10.0; multiplier * 5],
                vec![11.0; multiplier * 7],
                vec![12.0; multiplier * 3],
                vec![12.2; multiplier * 5],
            ],
        };
        (storage, multiplier)
    }

    #[test]
    fn test_storage_into_chunks() {
        let (storage, multiplier) = mock_storage();
        let (training_chunks, evaluation_chunk) =
            storage.split_storage_into_chunks(3, 1, 4).unwrap();
        assert_approx_eq!(
            f32,
            &training_chunks[0].data,
            vec![
                vec![0.0; multiplier * 10],
                vec![1.2; multiplier * 3],
                vec![2.0; multiplier],
            ]
        );
        assert_approx_eq!(
            f32,
            &training_chunks[1].data,
            vec![
                vec![3.0; multiplier * 12],
                vec![4.0; multiplier * 10],
                vec![5.0; multiplier * 10],
            ]
        );
        assert_approx_eq!(
            f32,
            &training_chunks[2].data,
            vec![
                vec![6.0; multiplier * 10],
                vec![7.0; multiplier * 10],
                vec![4.8; multiplier * 10],
            ]
        );
        assert_approx_eq!(
            f32,
            &training_chunks[3].data,
            vec![vec![9.0; multiplier * 6],]
        );
        assert_eq!(training_chunks.len(), 4);

        assert_approx_eq!(
            f32,
            &evaluation_chunk.data,
            vec![
                vec![10.0; multiplier * 5],
                vec![11.0; multiplier * 7],
                vec![12.0; multiplier * 3],
                vec![12.2; multiplier * 5],
            ]
        );
        assert_eq!(evaluation_chunk.number_of_samples(), 4);
    }

    #[test]
    fn test_storage_into_chunks_error_cases() {
        assert!(mock_storage().0.split_storage_into_chunks(0, 1, 4).is_err());
        assert!(mock_storage().0.split_storage_into_chunks(1, 2, 4).is_err());
        assert!(mock_storage()
            .0
            .split_storage_into_chunks(2, 2, 15)
            .is_err());
        mock_storage()
            .0
            .split_storage_into_chunks(2, 1, 14)
            .unwrap();
        mock_storage().0.split_storage_into_chunks(1, 1, 4).unwrap();
        mock_storage()
            .0
            .split_storage_into_chunks(2, 2, 10)
            .unwrap();
    }

    #[test]
    fn test_storage_into_chunks_empty_chunks() {
        let (chunked, extra) = mock_storage()
            .0
            .split_storage_into_chunks(2, 2, 14)
            .unwrap();
        assert_eq!(chunked.len(), 0);
        assert_eq!(extra.number_of_samples(), 14);
    }

    #[test]
    fn test_storage_into_chunks_empty_tail_segment() {
        let (chunked, extra) = mock_storage().0.split_storage_into_chunks(3, 2, 0).unwrap();
        assert_eq!(chunked.len(), 5);
        assert_eq!(extra.number_of_samples(), 0);
    }

    #[test]
    fn test_data_source_new_split_no_eval_samples() {
        let (storage, _) = mock_storage();
        let (train_sources, eval_source) = DataSource::new_split(storage, 0.0, 2, 5).unwrap();
        assert!(eval_source.is_none());
        assert_eq!(train_sources.len(), 3)
    }

    #[test]
    fn test_data_source_new_split() {
        let (storage, multiplier) = mock_storage();
        let (mut train_sources, eval_source) = DataSource::new_split(storage, 0.286, 2, 3).unwrap();
        let mut eval_source = eval_source.unwrap();

        assert_approx_eq!(
            f32,
            &train_sources[0].storage.data,
            vec![
                vec![0.0; multiplier * 10],
                vec![1.2; multiplier * 3],
                vec![2.0; multiplier],
            ]
        );
        train_sources[0].reset().unwrap();
        assert_eq!(train_sources[0].next_training_batch().unwrap().len(), 2);
        assert_eq!(train_sources[0].next_training_batch().unwrap().len(), 0);

        assert_approx_eq!(
            f32,
            &train_sources[1].storage.data,
            vec![
                vec![3.0; multiplier * 12],
                vec![4.0; multiplier * 10],
                vec![5.0; multiplier * 10],
            ]
        );
        train_sources[1].reset().unwrap();
        assert_eq!(train_sources[1].next_training_batch().unwrap().len(), 2);
        assert_eq!(train_sources[1].next_training_batch().unwrap().len(), 0);

        assert_approx_eq!(
            f32,
            &train_sources[2].storage.data,
            vec![
                vec![6.0; multiplier * 10],
                vec![7.0; multiplier * 10],
                vec![4.8; multiplier * 10],
            ]
        );
        train_sources[2].reset().unwrap();
        assert_eq!(train_sources[2].next_training_batch().unwrap().len(), 2);
        assert_eq!(train_sources[2].next_training_batch().unwrap().len(), 0);

        assert_eq!(train_sources.len(), 3);

        assert_approx_eq!(
            f32,
            &eval_source.storage.data,
            vec![
                vec![10.0; multiplier * 5],
                vec![11.0; multiplier * 7],
                vec![12.0; multiplier * 3],
                vec![12.2; multiplier * 5],
            ]
        );
        eval_source.reset().unwrap();
        assert!(eval_source.next_evaluation_sample().unwrap().is_some());
        assert!(eval_source.next_evaluation_sample().unwrap().is_some());
        assert!(eval_source.next_evaluation_sample().unwrap().is_some());
        assert!(eval_source.next_evaluation_sample().unwrap().is_some());
        assert!(eval_source.next_evaluation_sample().unwrap().is_none());
    }
}
