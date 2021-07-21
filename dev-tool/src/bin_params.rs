use std::{
    collections::HashSet,
    convert::TryInto,
    fmt::{self, Display},
    ops::RangeInclusive,
    path::PathBuf,
};

use anyhow::{bail, Error};
use log::debug;
use ndarray::{ArrayBase, ArrayD, Data, Dimension};
use structopt::StructOpt;
use xayn_ai::list_net::ndutils::io::{BinParams, LoadingBinParamsFailed};

use crate::exit_code::{NON_FATAL_ERROR, NO_ERROR};

#[derive(StructOpt, Debug)]
pub enum BinParamsCmd {
    /// Inspect a ".binparams" file.
    Inspect(InspectBinParamsCmd),
}

impl BinParamsCmd {
    pub fn run(self) -> Result<i32, Error> {
        match self {
            BinParamsCmd::Inspect(cmd) => cmd.run(),
        }
    }
}

#[derive(StructOpt, Debug)]
pub struct InspectBinParamsCmd {
    /// Prints all the values in all the matrices.
    #[structopt(short, long)]
    print_data: bool,

    /// If set only skip arrays which names are not in the filter.
    ///
    /// This accepts accepts comma separated lists. Additionally
    /// to including the option multiple times.
    #[structopt(short, long)]
    filter: Option<Vec<String>>,

    /// Prints stats including min,max,mean,std, has_nans, has_infs, has_subnormals
    #[structopt(short = "s", long)]
    stats: bool,

    /// Checks if all values in all arrays are "normal"
    #[structopt(short = "c", long)]
    check_normal: bool,

    /// Checks if all values in all arrays are in given range
    ///
    /// Format: <from>..=<to>
    ///
    /// E.g.: --check-range="-10..=20"
    #[structopt(short = "r", long)]
    check_range: Option<String>,

    /// Path to a `.binparams` file.
    file: PathBuf,
}

impl InspectBinParamsCmd {
    pub fn run(self) -> Result<i32, Error> {
        self.run_(BinParams::deserialize_from_file)
    }
    fn run_(
        self,
        load_bin_params: impl FnOnce(PathBuf) -> Result<BinParams, LoadingBinParamsFailed>,
    ) -> Result<i32, Error> {
        let Self {
            print_data,
            file,
            stats,
            check_normal,
            check_range,
            filter,
        } = self;

        let check_range = check_range.map(parse_range).transpose()?;

        let filter = filter.map(parse_filter);

        debug!("Loading BinParams.");
        let params = load_bin_params(file)?;

        debug!("Inspecting BinParams");
        let mut failed_normal_checks = Vec::new();
        let mut failed_range_checks = Vec::new();
        for (name, flat_array) in params.into_iter() {
            if let Some(filter) = &filter {
                if !filter.contains(&name) {
                    debug!("Skipping Array: {}", name);
                    continue;
                }
            }

            println!("----------------------------------------");
            let array: ArrayD<f32> = match flat_array.try_into() {
                Ok(array) => array,
                Err(err) => {
                    eprint!("Array {} has invalid data: {}\n{:?}", name, err, err);
                    break;
                }
            };
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

struct Stats {
    /// The max value in the array, or 0 if the array is empty.
    min: f32,
    /// The min value in the array, or 0 if the array is empty.
    max: f32,
    /// The arithmetic mean of the values in the array, or 0 if the array is empty.
    mean: f32,
    /// The standard derivation (ddof=0) of the values in the array.
    std: f32,
    /// Is `true` if the array contains `NaN` values.
    has_nans: bool,
    /// Is `true` if the array contains `Inf` values (independent of sign).
    has_infs: bool,
    /// Is `true` if the array contains subnormal values.
    has_subnormals: bool,

    /// True if any value is not normal.
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
                self_.has_subnormals |= val.is_subnormal();
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

fn highlight_if_true(val: bool) -> &'static str {
    if val {
        "<<TRUE>>"
    } else {
        "false"
    }
}

#[cfg(test)]
mod tests {
    use std::{iter::FromIterator, path::Path};

    use ndarray::{arr1, arr2};
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
        let cmd = InspectBinParamsCmd {
            print_data: false,
            filter: None,
            stats: false,
            check_normal: false,
            check_range: Some("0..=0.5".to_owned()),
            file: PathBuf::from("/my/path"),
        };

        let mut bin_params = BinParams::default();
        bin_params.insert("foobar", arr2(&[[1., 3.], [-4., 0.24]]));
        bin_params.insert("barfoot", arr1(&[0., 0.12]));

        let exit_code = cmd
            .run_(move |path| {
                assert_eq!(path, Path::new("/my/path"));
                Ok(bin_params)
            })
            .unwrap();

        assert_eq!(exit_code, NON_FATAL_ERROR);
    }

    #[test]
    fn test_filter_works() {
        let cmd = InspectBinParamsCmd {
            print_data: false,
            filter: Some(vec!["foobar".to_owned()]),
            stats: false,
            check_normal: true,
            check_range: None,
            file: PathBuf::from("/my/path"),
        };

        let mut bin_params = BinParams::default();
        bin_params.insert("foobar", arr2(&[[1., 3.], [-4., 0.24]]));
        bin_params.insert("barfoot", arr1(&[0., f32::NAN, 0.12]));

        let exit_code = cmd
            .run_(move |path| {
                assert_eq!(path, Path::new("/my/path"));
                Ok(bin_params)
            })
            .unwrap();

        assert_eq!(exit_code, NO_ERROR);
    }
}
