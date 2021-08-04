#![cfg(not(tarpaulin))]

//FIXME[follow up PR]: Move modified parts of this module into the `ltr::list_net` module to re-use them for in-app training.
use std::{
    error::Error as StdError,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    ops::RangeTo,
    path::Path,
    u64,
};

use anyhow::{bail, Error};
use bincode::Options;
use displaydoc::Display;
use log::debug;
use ndarray::{ArrayBase, ArrayView, ArrayView1, ArrayView2, Data, Dimension};
use rand::{prelude::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use xayn_ai::list_net::{self, ListNet, Sample};

/// A [`xayn_ai::list_net::DataSource`] implementation.
pub(crate) struct DataSource<S>
where
    S: Storage,
{
    /// The storage containing all samples.
    storage: S,
    /// The batch size (if already provided).
    batch_size: Option<usize>,
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
    /// - If `evaluation_split` is less then 0 or not a normal float.
    /// - If there are no samples.
    /// - If there are no training samples with the given `evaluation_split`.
    /// - If there are no evaluation samples with the given `evaluation_split`,
    ///   but the split is not `0`.
    /// - If calling `storage.data_ids()` failed.
    #[allow(dead_code)] //FIXME is used by training (added in part 3 of this PR)
    pub(crate) fn new(
        storage: S,
        evaluation_split: f32,
    ) -> Result<Self, DataSourceError<S::Error>> {
        if evaluation_split < 0. || !evaluation_split.is_normal() {
            return Err(DataSourceError::BadEvaluationSplit(evaluation_split));
        }
        let nr_all_samples = storage.data_ids().map_err(DataSourceError::Storage)?.end;
        if nr_all_samples == 0 {
            return Err(DataSourceError::EmptyDatabase);
        }
        let nr_evaluation_samples = (nr_all_samples as f32 * evaluation_split).round() as usize;
        if nr_evaluation_samples >= nr_all_samples {
            return Err(DataSourceError::TooLargeEvaluationSplit(evaluation_split));
        }
        if nr_evaluation_samples == 0 && evaluation_split > 0. {
            return Err(DataSourceError::NoEvaluationSamples(evaluation_split));
        }
        let nr_training_samples = nr_all_samples - nr_evaluation_samples;
        let evaluation_ids = (nr_training_samples..nr_all_samples).collect();
        let training_ids = (0..nr_training_samples).collect();

        Ok(Self {
            storage,
            batch_size: None,
            training_data_order: DataLookupOrder::new(training_ids),
            evaluation_data_order: DataLookupOrder::new(evaluation_ids),
        })
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
    /// A batch size of 0 is not usable for training.
    BatchSize0,
    /// The batch size is larger then the number of samples.
    TooLargeBatchSize(usize),
    /// Not reset/initialized.
    ResetWasNotCalledBeforeTraining,
    /// Empty database cannot be used for training.
    EmptyDatabase,
    /// Fetching sample from storage failed: {0}.
    Storage(SE),
}

impl<S> list_net::DataSource for DataSource<S>
where
    S: Storage,
{
    type Error = DataSourceError<S::Error>;

    fn reset(&mut self, batch_size: usize) -> Result<usize, Self::Error> {
        if batch_size == 0 {
            return Err(DataSourceError::BatchSize0);
        }
        let mut rng = rand::thread_rng();
        self.batch_size = Some(batch_size);
        self.training_data_order.reset(&mut rng);
        self.evaluation_data_order.reset(&mut rng);

        let nr_batches = self.training_data_order.number_of_batches(batch_size);
        if nr_batches == 0 {
            return Err(DataSourceError::TooLargeBatchSize(nr_batches));
        }
        Ok(nr_batches)
    }

    fn next_training_batch(&mut self) -> Result<Vec<Sample>, Self::Error> {
        let batch_size = self
            .batch_size
            .ok_or(DataSourceError::ResetWasNotCalledBeforeTraining)?;

        let ids = self.training_data_order.next_batch(batch_size);
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        self.storage
            .load_batch(&ids)
            .map_err(DataSourceError::Storage)
    }

    fn next_evaluation_sample(&mut self) -> Result<Option<Sample>, Self::Error> {
        if let Some(id) = self.evaluation_data_order.next() {
            match self.storage.load_sample(id) {
                Ok(sample) => Ok(Some(sample)),
                Err(error) => Err(DataSourceError::Storage(error)),
            }
        } else {
            Ok(None)
        }
    }
}

pub(crate) trait Storage {
    type Error: StdError + 'static;

    /// Return all sample ids.
    ///
    /// For now this is a range. I.e. `0..nr_of_samples`. We might need
    /// to change this in the future, but for now this is simpler.
    fn data_ids(&self) -> Result<RangeTo<usize>, Self::Error>;

    /// Loads a sample and returns a reference to it.
    ///
    /// This method is `&mut self` as it might change the internal state
    /// (due to loading/caching) and we also want to make sure that there
    /// are no more lend out samples, when `load_sample` or `load_batch` are
    /// called again.
    fn load_sample(&mut self, id: DataId) -> Result<Sample, Self::Error>;

    /// Loads a batch of samples and returns a vector of references to them.
    ///
    /// See [`Storage.load_batch()`] about why this is `&mut self`.
    fn load_batch<'a>(&'a mut self, ids: &'_ [DataId]) -> Result<Vec<Sample<'a>>, Self::Error>;
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

    /// Returns the number of batches which will be produced with the given batch size.
    fn number_of_batches(&self, batch_size: usize) -> usize {
        self.data_ids.len() / batch_size
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
pub(crate) struct InMemorySamples {
    /// Vector of concatenated `inputs` and `target_prob_dist`, this slightly improves memory size and
    /// cache locality, we can derive the number of document from the vectors length as the vectors length
    /// is `nr_document * INPUT_NR_FEATURES + nr_document*1`, i.e. `nr_documents * 51`.
    data: Vec<Vec<f32>>,
}

#[derive(Error, Debug, Display)]
pub enum StorageError {
    /// The database was parsed successfully but contains broken invariants at index {at_index}.
    BrokenInvariants { at_index: usize },
}

impl InMemorySamples {
    /// Serializes this instance into a file, preferably using the `.samples` file ending.
    //FIXME[follow-up PR] version the file format, it's used to persist data and it's not
    //                    unlikely to slightly change in the future.
    pub(crate) fn serialize_into_file(&self, file: impl AsRef<Path>) -> Result<(), Error> {
        self.serialize_into(BufWriter::new(File::create(file)?))
    }

    /// Serializes this instance into the given writer.
    fn serialize_into(&self, writer: impl Write) -> Result<(), Error> {
        bincode::DefaultOptions::new()
            .serialize_into(writer, self)
            .map_err(Into::into)
    }

    /// Deserialize an instance from the given file.
    #[allow(dead_code)] //FIXME is used by training (added in part 3 of this PR)
    pub(crate) fn deserialize_from_file(file: impl AsRef<Path>) -> Result<Self, Error> {
        Self::deserialize_from(BufReader::new(File::open(file)?))
    }

    /// Deserialize an instance from the given (preferably buffered) reader.
    #[allow(dead_code)] //FIXME is used by training (added in part 3 of this PR)
    fn deserialize_from(reader: impl Read) -> Result<Self, Error> {
        let self_: Self = bincode::DefaultOptions::new().deserialize_from(reader)?;
        debug!("Loaded {} samples.", self_.data.len());
        Ok(self_)
    }

    fn load_sample_helper(&self, id: DataId) -> Result<Sample, StorageError> {
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

        Ok(Sample {
            inputs,
            target_prob_dist,
        })
    }

    /// Add a new sample.
    ///
    /// # Panics
    ///
    /// - If the number of documents in `inputs` doesn't match the number of probabilities in
    /// `target_prob_dist`.
    /// - If the number of features per document is not equal to `[ListNet::INPUT_NR_FEATURES]`.
    pub(crate) fn add_sample(
        &mut self,
        inputs: ArrayView2<f32>,
        target_prob_dist: ArrayView1<f32>,
    ) -> Result<(), Error> {
        if inputs.shape() != [target_prob_dist.len(), ListNet::INPUT_NR_FEATURES] {
            bail!("Sample with bad array shapes. Expected shapes [{nr_docs}, {nr_feats}] & [{nr_docs}] but got shapes {inputs_shape:?} & {prob_dist_shape:?}.",
                nr_docs=inputs.shape()[0],
                nr_feats=ListNet::INPUT_NR_FEATURES,
                inputs_shape=inputs.shape(),
                prob_dist_shape=target_prob_dist.shape(),
            );
        }
        let mut datum = Vec::with_capacity(inputs.len() + target_prob_dist.len());
        extend_vec_with_ndarray(&mut datum, inputs);
        extend_vec_with_ndarray(&mut datum, target_prob_dist);
        self.data.push(datum);
        Ok(())
    }
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

    fn data_ids(&self) -> Result<RangeTo<usize>, Self::Error> {
        Ok(..self.data.len())
    }

    fn load_sample(&mut self, id: DataId) -> Result<Sample, Self::Error> {
        self.load_sample_helper(id)
    }

    fn load_batch<'a>(&'a mut self, ids: &'_ [DataId]) -> Result<Vec<Sample<'a>>, Self::Error> {
        let mut samples = Vec::new();
        for id in ids {
            samples.push(self.load_sample_helper(*id)?)
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use rand::thread_rng;
    use xayn_ai::assert_approx_eq;

    use super::*;

    fn dummy_storage() -> InMemorySamples {
        let mut storage = InMemorySamples::default();

        let inputs = Array::zeros((10, 50));
        let target_prob_dist = Array::ones((10,));
        storage
            .add_sample(inputs.view(), target_prob_dist.view())
            .unwrap();

        let inputs = Array::ones((10, 50));
        let target_prob_dist = Array::zeros((10,));
        storage
            .add_sample(inputs.view(), target_prob_dist.view())
            .unwrap();

        let inputs = Array::from_elem((10, 50), 4.);
        let target_prob_dist = Array::ones((10,));
        storage
            .add_sample(inputs.view(), target_prob_dist.view())
            .unwrap();

        storage
    }

    #[test]
    fn test_adding_bad_matrices_fails() {
        let mut storage = InMemorySamples::default();
        let inputs = Array::zeros((10, 48));
        let target_prob_dist = Array::ones((10,));
        let err = storage
            .add_sample(inputs.view(), target_prob_dist.view())
            .unwrap_err();

        assert_eq!(&format!("{}", err), "Sample with bad array shapes. Expected shapes [10, 50] & [10] but got shapes [10, 48] & [10]." );

        let inputs = Array::zeros((8, 50));
        let target_prob_dist = Array::ones((11,));
        let err = storage
            .add_sample(inputs.view(), target_prob_dist.view())
            .unwrap_err();

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
        let mut rng = thread_rng();
        let mut dlo = DataLookupOrder::new(vec![0, 1, 2, 3, 4]);

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
}