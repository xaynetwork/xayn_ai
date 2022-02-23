#[cfg(target_arch = "wasm32")]
use std::time::Duration;
use std::{cmp::Ordering, time::SystemTime};

#[cfg(target_arch = "wasm32")]
use js_sys::Date;
use serde::Serialize;

use crate::Error;

macro_rules! to_vec_of_ref_of {
    ($data: expr, $type:ty) => {
        $data
            .iter()
            .map(|data| -> $type { data })
            .collect::<Vec<_>>()
    };
}
#[allow(clippy::useless_attribute, clippy::single_component_path_imports)]
pub(crate) use to_vec_of_ref_of;

/// Allows comparing and sorting f32 even if `NaN` is involved.
///
/// Pretend that f32 has a total ordering.
///
/// `NaN` is treated as the lowest possible value if `nan_min`, similar to what [`f32::max`] does.
/// Otherwise it is treated as the highest possible value, similar to what [`f32::min`] does.
pub(crate) fn nan_safe_f32_cmp_base(a: &f32, b: &f32, nan_min: bool) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        // if `partial_cmp` returns None we have at least one `NaN`,
        let cmp = match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, _) => Ordering::Less,
            (_, true) => Ordering::Greater,
            _ => unreachable!("partial_cmp returned None but both numbers are not NaN"),
        };
        if nan_min {
            cmp
        } else {
            cmp.reverse()
        }
    })
}

/// Allows comparing and sorting f32 even if `NaN` is involved.
///
/// Pretend that f32 has a total ordering.
///
/// `NaN` is treated as the lowest possible value, similar to what [`f32::max`] does.
///
/// If this is used for sorting this will lead to an ascending order, like
/// for example `[NaN, 0.5, 1.5, 2.0]`.
///
/// By switching the input parameters around this can be used to create a
/// descending sorted order, like e.g.: `[2.0, 1.5, 0.5, NaN]`.
pub(crate) fn nan_safe_f32_cmp(a: &f32, b: &f32) -> Ordering {
    nan_safe_f32_cmp_base(a, b, true)
}

/// Allows comparing and sorting f32 even if `NaN` is involved.
///
/// Pretend that f32 has a total ordering.
///
/// `NaN` is treated as the highest possible value, similar to what [`f32::min`] does.
///
/// If this is used for sorting this will lead to an ascending order, like
/// for example `[0.5, 1.5, 2.0, NaN]`.
///
/// By switching the input parameters around this can be used to create a
/// descending sorted order, like e.g.: `[NaN, 2.0, 1.5, 0.5]`.
pub(crate) fn nan_safe_f32_cmp_high(a: &f32, b: &f32) -> Ordering {
    nan_safe_f32_cmp_base(a, b, false)
}

/// `nan_safe_f32_cmp_desc(a,b)` is syntax suggar for `nan_safe_f32_cmp(b, a)`
#[inline]
pub(crate) fn nan_safe_f32_cmp_desc(a: &f32, b: &f32) -> Ordering {
    nan_safe_f32_cmp(b, a)
}

/// Serializes the given data, tagged with the given version number.
pub(crate) fn serialize_with_version(data: &impl Serialize, version: u8) -> Result<Vec<u8>, Error> {
    let size = bincode::serialized_size(data)? + 1;
    let mut serialized = Vec::with_capacity(size as usize);
    // version is encoded in the first byte
    serialized.push(version);
    bincode::serialize_into(&mut serialized, data)?;

    Ok(serialized)
}

/// The number of seconds per day (without leap seconds).
pub(crate) const SECONDS_PER_DAY: f32 = 86400.;

/// Gets the current system time depending on the target architecture.
#[inline]
pub(crate) fn system_time_now() -> SystemTime {
    #[cfg(target_arch = "wasm32")]
    return SystemTime::UNIX_EPOCH + Duration::from_secs_f64(Date::now() / 1000.);

    #[cfg(not(target_arch = "wasm32"))]
    return SystemTime::now();
}

pub(crate) mod serde_duration_as_days {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::utils::SECONDS_PER_DAY;

    pub(crate) fn serialize<S>(horizon: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (horizon.as_secs() / SECONDS_PER_DAY as u64).serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        u64::deserialize(deserializer)
            .map(|days| Duration::from_secs(SECONDS_PER_DAY as u64 * days))
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, time::Duration};

    use serde::Deserialize;
    use serde_json::{from_str, to_string};

    use super::*;
    use test_utils::assert_approx_eq;

    #[test]
    fn test_nan_safe_f32_cmp_sorts_in_the_right_order() {
        let data = &mut [f32::NAN, 1., 5., f32::NAN, 4.];
        data.sort_by(nan_safe_f32_cmp);

        assert_approx_eq!(f32, &data[2..], [1., 4., 5.], ulps = 0);
        assert!(data[0].is_nan());
        assert!(data[1].is_nan());

        data.sort_by(nan_safe_f32_cmp_desc);

        assert_approx_eq!(f32, &data[..3], [5., 4., 1.], ulps = 0);
        assert!(data[3].is_nan());
        assert!(data[4].is_nan());

        let data = &mut [1., 5., 3., 4.];

        data.sort_by(nan_safe_f32_cmp);
        assert_approx_eq!(f32, &data[..], [1., 3., 4., 5.], ulps = 0);

        data.sort_by(nan_safe_f32_cmp_desc);
        assert_approx_eq!(f32, &data[..], [5., 4., 3., 1.], ulps = 0);
    }

    #[test]
    fn test_nan_safe_f32_cmp_nans_compare_as_expected() {
        assert_eq!(nan_safe_f32_cmp(&f32::NAN, &f32::NAN), Ordering::Equal);
        assert_eq!(nan_safe_f32_cmp(&-12., &f32::NAN), Ordering::Greater);
        assert_eq!(nan_safe_f32_cmp_desc(&-12., &f32::NAN), Ordering::Less);
        assert_eq!(nan_safe_f32_cmp(&f32::NAN, &-12.), Ordering::Less);
        assert_eq!(nan_safe_f32_cmp_desc(&f32::NAN, &-12.), Ordering::Greater);
        assert_eq!(nan_safe_f32_cmp(&12., &f32::NAN), Ordering::Greater);
        assert_eq!(nan_safe_f32_cmp_desc(&12., &f32::NAN), Ordering::Less);
        assert_eq!(nan_safe_f32_cmp(&f32::NAN, &12.), Ordering::Less);
        assert_eq!(nan_safe_f32_cmp_desc(&f32::NAN, &12.), Ordering::Greater);
    }

    #[derive(Deserialize, Serialize)]
    struct Days(#[serde(with = "serde_duration_as_days")] Duration);

    #[test]
    fn test_less() -> Result<(), Box<dyn Error>> {
        let duration = Duration::from_secs(SECONDS_PER_DAY as u64 - 1);
        let serialized = to_string(&Days(duration))?;
        assert_eq!(serialized, "0");
        let deserialized = from_str::<Days>(&serialized)?.0;
        assert_eq!(deserialized, Duration::ZERO);
        Ok(())
    }

    #[test]
    fn test_equal() -> Result<(), Box<dyn Error>> {
        let duration = Duration::from_secs(SECONDS_PER_DAY as u64);
        let serialized = to_string(&Days(duration))?;
        assert_eq!(serialized, "1");
        let deserialized = from_str::<Days>(&serialized)?.0;
        assert_eq!(deserialized, duration);
        Ok(())
    }

    #[test]
    fn test_greater() -> Result<(), Box<dyn Error>> {
        let duration = Duration::from_secs(SECONDS_PER_DAY as u64 + 1);
        let serialized = to_string(&Days(duration))?;
        assert_eq!(serialized, "1");
        let deserialized = from_str::<Days>(&serialized)?.0;
        assert_eq!(deserialized, Duration::from_secs(SECONDS_PER_DAY as u64));
        Ok(())
    }
}
