use std::{
    collections::HashSet,
    convert::TryInto,
    fmt::{self, Display},
    iter,
    ops::RangeInclusive,
    path::PathBuf,
    str::FromStr,
};

use anyhow::{bail, Error};
use displaydoc::Display;
use itertools::Itertools;
use log::debug;
use ndarray::{ArrayBase, ArrayD, Data, Dimension};
use structopt::StructOpt;

use xayn_ai::list_net::ndutils::io::BinParams;

use super::data_source::{InMemorySamples, Storage};
use crate::exit_code::{NON_FATAL_ERROR, NO_ERROR};

/// Inspect data dumps (binparams, samples).
#[derive(StructOpt, Debug)]
pub struct InspectCmd {
    //FIXME[followup pr] add a tag to the begin of bincode serialized files to
    //   automatically select the right deserializer and prevent confusion when
    //   accidentally mixing up this files.
    /// Type of the file (either "binparams" or "samples")
    #[structopt(short, long, parse(try_from_str))]
    r#type: FileType,

    /// Prints the elements of all inspected matrices.
    #[structopt(short, long)]
    print_data: bool,

    /// If set only skip arrays which names are not in the filter.
    ///
    /// This accepts comma separated lists, as well as including the option multiple times.
    #[structopt(short, long)]
    filter: Option<Vec<String>>,

    /// Prints stats including `min`, `max`, `mean`, `std`, `has_nans`, `has_infs`, `has_subnormals`.
    #[structopt(short = "s", long)]
    stats: bool,

    /// Checks if all elements in all matrices are "normal".
    #[structopt(short = "c", long)]
    check_normal: bool,

    /// Checks if all values in all matrices are in given range.
    ///
    /// Format: <from>..=<to>
    ///
    /// E.g.: --check-range="-10..=20"
    ///
    /// Currently only full inclusive ranges are supported.
    #[structopt(short = "r", long)]
    check_range: Option<String>,

    /// Path to a input file.
    file: PathBuf,
}

#[derive(Debug, Display, Clone, Copy)]
/// Supported file types.
pub enum FileType {
    /// BinParams file
    BinParams,
    /// Samples file
    Samples,
}

impl FromStr for FileType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use FileType::*;
        match &*s.trim().to_lowercase() {
            "binparams" | "bin-params" => Ok(BinParams),
            "samples" => Ok(Samples),
            _ => bail!("Unexpected file type. Supported types are: \"binparams\", \"samples\""),
        }
    }
}

type MatricesIter = Box<dyn Iterator<Item = Result<(String, ArrayD<f32>), Error>>>;

impl InspectCmd {
    pub fn run(self) -> Result<i32, Error> {
        debug!("Loading Matrices from {}.", self.r#type);
        let matrices = self.load_matrices()?;

        debug!("Inspecting Matrices");
        self.run_inspect_matrices(matrices)
    }

    fn load_matrices(&self) -> Result<MatricesIter, Error> {
        match self.r#type {
            FileType::BinParams => {
                let bin_params = BinParams::deserialize_from_file(&self.file)?;
                Ok(bin_params_to_matrices_iter(bin_params))
            }
            FileType::Samples => {
                let samples = InMemorySamples::deserialize_from_file(&self.file)?;
                samples_to_matrices_iter(samples)
            }
        }
    }

    fn run_inspect_matrices(self, matrices: MatricesIter) -> Result<i32, Error> {
        let Self {
            r#type: _,
            print_data,
            file: _,
            stats,
            check_normal,
            check_range,
            filter,
        } = self;

        let check_range = check_range.map(parse_range).transpose()?;
        let filter = filter.map(parse_filter);

        let mut failed_normal_checks = Vec::new();
        let mut failed_range_checks = Vec::new();
        for result in matrices {
            let (name, array) = result?;
            if let Some(filter) = &filter {
                if !filter.contains(&name) {
                    debug!("Skipping Array: {}", name);
                    continue;
                }
            }

            println!("----------------------------------------");
            println!("Name: {}", name);
            println!("Shape: {:?}", array.shape());
            if stats || check_normal || check_range.is_some() {
                let array_stats =
                    Stats::calculate(array.view(), check_normal, check_range.as_ref());
                if stats {
                    println!("Stats: {}", array_stats);
                }
                if check_normal && array_stats.normal_checks_failed {
                    failed_normal_checks.push(name.clone());
                }
                if check_range.is_some() && array_stats.range_checks_failed {
                    failed_range_checks.push(name);
                }
            }

            if print_data {
                println!("Array: {:?}", array);
            }
        }

        failed_normal_checks.sort();
        failed_range_checks.sort();

        if failed_range_checks.is_empty() && failed_normal_checks.is_empty() {
            Ok(NO_ERROR)
        } else {
            let mut msg = "Checks Failed:\n".to_owned();
            if check_normal {
                write_failure_message(&mut msg, "Failed normal checks", &failed_normal_checks)?;
            }
            if let Some(check_range) = check_range {
                write_failure_message(
                    &mut msg,
                    &format!(
                        "Failed range checks ({}..={})",
                        check_range.start(),
                        check_range.end()
                    ),
                    &failed_range_checks,
                )?;
            }
            eprintln!("{}", msg);
            Ok(NON_FATAL_ERROR)
        }
    }
}

/// Turns a [`BinParams`] instance into a iterator over it's matrices and their names.
///
/// - The matrices are converted to the `ArrayD` type.
/// - The matrices are returned sorted by their names.
fn bin_params_to_matrices_iter(bin_params: BinParams) -> MatricesIter {
    let iter = bin_params
        .into_iter()
        .map(|(name, array)| (name, array.try_into()))
        .sorted_by_key(|(name, _)| name.clone())
        .into_iter()
        .map(|(name, array_res)| Ok((name, array_res?)));

    Box::new(iter)
}

/// Turns an [`InMemorySamples`] instance into a iterator over all it's matrices and their names.
///
/// - Matrices are returned in order of the samples, and for each the sample it's matrices are
///   returned sorted by their name.
/// - The name is created in the form of `{index}.{matrix_name}` the `index` will be
///   formatted with leading `0` chars appropriate for the number of samples.
fn samples_to_matrices_iter(mut samples: InMemorySamples) -> Result<MatricesIter, Error> {
    let mut count = 0;
    let end = samples.data_ids()?.end;
    let nr_digits = format!("{}", end.saturating_sub(1)).len();
    let iter = iter::from_fn(move || {
        let idx = count / 2;
        let for_inputs = count % 2 == 0;
        count += 1;

        if idx >= end {
            return None;
        }
        let sample = match samples.load_sample(idx) {
            Ok(sample) => sample,
            Err(err) => return Some(Err(err.into())),
        };
        let (name, array) = if for_inputs {
            ("inputs", sample.inputs.to_owned().into_dyn())
        } else {
            (
                "target_prob_dist",
                sample.target_prob_dist.to_owned().into_dyn(),
            )
        };
        Some(Ok((
            format!("{:0>width$}.{}", idx, name, width = nr_digits),
            array,
        )))
    });
    Ok(Box::new(iter))
}

fn write_failure_message(
    out: &mut String,
    title: &str,
    array: &[String],
) -> Result<(), fmt::Error> {
    use fmt::Write;
    writeln!(out, " {}:", title)?;
    for array_name in array {
        writeln!(out, " - {}", array_name)?;
    }
    Ok(())
}

/// Parses the filter by splitting each filter sequence and trimming each filter.
fn parse_filter(filter: Vec<String>) -> HashSet<String> {
    filter
        .iter()
        .flat_map(|names| {
            names
                .split(',')
                .map(|name| name.trim().to_owned())
                .filter(|name| !name.is_empty())
        })
        .collect::<HashSet<_>>()
}

/// Parses a `<from>..=<to>` (f32) range.
fn parse_range(range: impl AsRef<str>) -> Result<RangeInclusive<f32>, Error> {
    let mut split = range.as_ref().split("..=");
    let first = split.next().unwrap();
    if let Some(second) = split.next() {
        if split.next().is_some() {
            bail!("Only <from>..=<to> syntax is currently allowed.");
        }
        let first: f32 = first.trim().parse()?;
        let second: f32 = second.trim().parse()?;
        Ok(first..=second)
    } else {
        bail!("Only <from>..=<to> syntax is currently allowed.");
    }
}

/// Stats for a matrix.
struct Stats {
    /// The max element in the matrix, or 0 if the matrix is empty.
    min: f32,
    /// The min element in the matrix, or 0 if the matrix is empty.
    max: f32,
    /// The arithmetic mean of the elements in the matrix, or 0 if the matrix is empty.
    mean: f32,
    /// The standard derivation (ddof=0) of the elements in the matrix.
    std: f32,
    /// Is `true`, if the matrix contains `NaN` elements.
    has_nans: bool,
    /// Is `true`, if the matrix contains `Inf` elements (independent of sign).
    has_infs: bool,
    /// Is `true`, if the matrix contains subnormal elements.
    has_subnormals: bool,
    /// True if any element is not normal.
    normal_checks_failed: bool,
    /// True if any range checks failed.
    range_checks_failed: bool,
}

impl Stats {
    fn calculate<S, D>(
        array: ArrayBase<S, D>,
        do_normal_checks: bool,
        range_checks: Option<&RangeInclusive<f32>>,
    ) -> Self
    where
        S: Data<Elem = f32>,
        D: Dimension,
    {
        array.iter().copied().fold(
            Self {
                min: 0.0,
                max: 0.0,
                mean: array.mean().unwrap_or_default(),
                std: array.std(0.),
                has_nans: false,
                has_infs: false,
                has_subnormals: false,
                normal_checks_failed: false,
                range_checks_failed: false,
            },
            |mut self_, val| {
                self_.min = self_.min.min(val);
                self_.max = self_.max.max(val);
                self_.has_nans |= val.is_nan();
                self_.has_infs |= val.is_infinite();
                // FIXME use once we switch to rust 1.53 on CI
                // self_.has_subnormals |= val.is_subnormal();
                self_.has_subnormals |= !val.is_normal() && !val.is_nan() && !val.is_infinite();
                self_.normal_checks_failed |= do_normal_checks && !val.is_normal();
                self_.range_checks_failed |= range_checks
                    .map(|range| !range.contains(&val))
                    .unwrap_or_default();
                self_
            },
        )
    }
}

impl Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            min,
            max,
            mean,
            std,
            has_nans,
            has_infs,
            has_subnormals,
            normal_checks_failed: normal_check_failed,
            range_checks_failed,
        } = self;
        write!(
            f,
            concat!(
                "min = {min}\n",
                "  mean = {mean}\n",
                "  max  = {max}\n",
                "  std  = {std}\n",
                "  NANs? {has_nans}\n",
                "  Infs? {has_infs}\n",
                "  Subnormals? {has_subnormals}\n",
                "  Failed range check? {range_check}\n",
                "  Failed normal check? {normal_check}\n",
            ),
            min = min,
            max = max,
            mean = mean,
            std = std,
            has_nans = highlight_if_true(*has_nans),
            has_infs = highlight_if_true(*has_infs),
            has_subnormals = highlight_if_true(*has_subnormals),
            range_check = highlight_if_true(*range_checks_failed),
            normal_check = highlight_if_true(*normal_check_failed),
        )
    }
}

/// Formats the boolean, "highlighting" true.
fn highlight_if_true(val: bool) -> &'static str {
    if val {
        "<<TRUE>>"
    } else {
        "false"
    }
}

#[cfg(test)]
mod tests {
    use std::iter::{Enumerate, FromIterator};

    use ndarray::{arr1, arr2, Array, Array1, Array2};
    use xayn_ai::assert_approx_eq;

    use super::*;

    #[test]
    fn test_parse_range() {
        let range = parse_range("-10.3..=-100.4").unwrap();
        assert_approx_eq!(f32, range.start(), -10.3);
        assert_approx_eq!(f32, range.end(), -100.4);

        let range = parse_range("0..=10").unwrap();
        assert_approx_eq!(f32, range.start(), 0.);
        assert_approx_eq!(f32, range.end(), 10.);

        let range = parse_range("-10..=10").unwrap();
        assert_approx_eq!(f32, range.start(), -10.);
        assert_approx_eq!(f32, range.end(), 10.);

        let range = parse_range("0. ..= 10.").unwrap();
        assert_approx_eq!(f32, range.start(), 0.);
        assert_approx_eq!(f32, range.end(), 10.);
    }

    #[test]
    fn test_parse_filter() {
        assert_eq!(
            parse_filter(vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]),
            HashSet::from_iter(vec!["a".to_owned(), "b".to_owned(), "c".to_owned()])
        );

        assert_eq!(
            parse_filter(vec!["a,b".to_owned(),]),
            HashSet::from_iter(vec!["a".to_owned(), "b".to_owned()])
        );

        assert_eq!(
            parse_filter(vec![
                ",,".to_owned(),
                " a ,,b , c,d,".to_owned(),
                ",c,d d,".to_owned()
            ]),
            HashSet::from_iter(vec![
                "a".to_owned(),
                "b".to_owned(),
                "c".to_owned(),
                "d".to_owned(),
                "d d".to_owned()
            ])
        );
    }

    #[test]
    fn test_calculate_stats() {
        let Stats {
            min,
            max,
            mean,
            std,
            has_nans,
            has_infs,
            has_subnormals,
            normal_checks_failed,
            range_checks_failed,
        } = Stats::calculate(arr2(&[[0.25, 0.125, -10., 1.]]), false, None);

        assert_approx_eq!(f32, min, -10.0, ulps = 0);
        assert_approx_eq!(f32, max, 1.0, ulps = 0);
        assert_approx_eq!(f32, mean, -2.15625, ulps = 0);
        assert_approx_eq!(f32, std, 4.540938, ulps = 0);
        assert!(!has_nans);
        assert!(!has_infs);
        assert!(!has_subnormals);
        assert!(!normal_checks_failed);
        assert!(!range_checks_failed);

        let Stats {
            has_nans,
            has_infs,
            has_subnormals,
            ..
        } = Stats::calculate(arr2(&[[f32::NAN, 0.125, -10., 1.]]), false, None);
        assert!(has_nans);
        assert!(!has_infs);
        assert!(!has_subnormals);

        let Stats {
            has_nans,
            has_infs,
            has_subnormals,
            ..
        } = Stats::calculate(arr2(&[[4., 0.125], [f32::INFINITY, 1.]]), false, None);
        assert!(!has_nans);
        assert!(has_infs);
        assert!(!has_subnormals);

        let subnormal = 0.00000000000000000000000000000000000000001;
        let Stats {
            has_nans,
            has_infs,
            has_subnormals,
            ..
        } = Stats::calculate(arr2(&[[0.1, subnormal, -10., 1.]]), false, None);
        assert!(!has_nans);
        assert!(!has_infs);
        assert!(has_subnormals);

        let Stats {
            normal_checks_failed,
            range_checks_failed,
            ..
        } = Stats::calculate(arr2(&[[f32::NAN, 0.125, -10., 1.]]), true, None);
        assert!(normal_checks_failed);
        assert!(!range_checks_failed);

        let Stats {
            normal_checks_failed,
            range_checks_failed,
            ..
        } = Stats::calculate(arr2(&[[10., 0.125, -10., 1.]]), true, Some(&(-0.5..=0.5)));
        assert!(!normal_checks_failed);
        assert!(range_checks_failed);

        let Stats {
            normal_checks_failed,
            range_checks_failed,
            ..
        } = Stats::calculate(
            arr2(&[[10., f32::NAN, -10., 1.]]),
            true,
            Some(&(-0.5..=0.5)),
        );
        assert!(normal_checks_failed);
        assert!(range_checks_failed);
    }

    #[test]
    fn test_failed_checks_affect_exit_status() {
        let cmd = InspectCmd {
            print_data: false,
            filter: None,
            stats: false,
            check_normal: false,
            check_range: Some("0..=0.5".to_owned()),
            file: PathBuf::from("/my/path"),
            r#type: FileType::BinParams,
        };

        let mut bin_params = BinParams::default();
        bin_params.insert("foobar", arr2(&[[1., 3.], [-4., 0.24]]));
        bin_params.insert("barfoot", arr1(&[0., 0.12]));

        let matrices = bin_params_to_matrices_iter(bin_params);
        let exit_code = cmd.run_inspect_matrices(matrices).unwrap();
        assert_eq!(exit_code, NON_FATAL_ERROR);
    }

    #[test]
    fn test_filter_works() {
        let cmd = InspectCmd {
            print_data: false,
            filter: Some(vec!["foobar".to_owned()]),
            stats: false,
            check_normal: true,
            check_range: None,
            file: PathBuf::from("/my/path"),
            r#type: FileType::BinParams,
        };

        let mut bin_params = BinParams::default();
        bin_params.insert("foobar", arr2(&[[1., 3.], [-4., 0.24]]));
        bin_params.insert("barfoot", arr1(&[0., f32::NAN, 0.12]));

        let matrices = bin_params_to_matrices_iter(bin_params);
        let exit_code = cmd.run_inspect_matrices(matrices).unwrap();

        assert_eq!(exit_code, NO_ERROR);
    }

    #[test]
    fn test_bin_params_to_matrices_iter() {
        let mut bin_params = BinParams::default();
        bin_params.insert("foobar", arr2(&[[1., 3.], [-4., 0.24]]));
        bin_params.insert("x", arr1(&[]));
        bin_params.insert("barfoot", arr1(&[0., f32::NAN, 0.12]));

        let mut iter = bin_params_to_matrices_iter(bin_params);

        test(iter.next(), ("barfoot", arr1(&[0., f32::NAN, 0.12])));
        test(iter.next(), ("foobar", arr2(&[[1., 3.], [-4., 0.24]])));
        test(iter.next(), ("x", arr1(&[])));
        assert!(iter.next().is_none());

        fn test(
            got: Option<Result<(String, ArrayD<f32>), Error>>,
            expected: (&str, Array<f32, impl Dimension>),
        ) {
            let (name, array) = got.expect("iter ended prematurely").unwrap();

            assert_eq!(name, expected.0);
            assert_approx_eq!(f32, array, expected.1, ulps = 0);
        }
    }

    #[test]
    fn test_samples_to_matrices_iter() {
        let mut storage = InMemorySamples::default();

        storage
            .add_sample(
                Array::from_elem((2, 50), 3.25).view(),
                arr1(&[0.25, 0.50]).view(),
            )
            .unwrap();
        storage
            .add_sample(Array::from_elem((0, 50), 3.25).view(), arr1(&[]).view())
            .unwrap();
        storage
            .add_sample(
                Array::from_elem((5, 50), 3.25).view(),
                arr1(&[0.25, 0.50, 1.25, 0.25, 0.44]).view(),
            )
            .unwrap();
        storage
            .add_sample(
                Array::from_elem((2, 50), 3.25).view(),
                arr1(&[0.52, 0.05]).view(),
            )
            .unwrap();

        let mut iter = samples_to_matrices_iter(storage).unwrap().enumerate();
        test(
            &mut iter,
            Array::from_elem((2, 50), 3.25),
            arr1(&[0.25, 0.50]),
        );
        test(&mut iter, Array::from_elem((0, 50), 3.25), arr1(&[]));
        test(
            &mut iter,
            Array::from_elem((5, 50), 3.25),
            arr1(&[0.25, 0.50, 1.25, 0.25, 0.44]),
        );
        test(
            &mut iter,
            Array::from_elem((2, 50), 3.25),
            arr1(&[0.52, 0.05]),
        );
        assert!(iter.next().is_none());

        fn test(
            iter: &mut Enumerate<MatricesIter>,
            inputs: Array2<f32>,
            target_prob_dist: Array1<f32>,
        ) {
            let (twice_idx, item) = iter.next().unwrap();
            let (name, array) = item.unwrap();
            let idx = twice_idx / 2;
            assert_eq!(name, format!("{}.inputs", twice_idx / 2));
            assert_approx_eq!(f32, array, inputs);
            let (twice_idx_plus_1, item) = iter.next().unwrap();
            let (name, array) = item.unwrap();
            assert_eq!(name, format!("{}.target_prob_dist", idx));
            assert_approx_eq!(f32, array, target_prob_dist);
            assert_eq!(twice_idx_plus_1, idx * 2 + 1);
        }
    }
}
