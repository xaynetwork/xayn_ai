use indicatif::{ProgressBar, ProgressStyle};

pub(crate) fn progress_spin_until_done<R>(msg: &'static str, func: impl FnOnce() -> R) -> R {
    let progress_bar = ProgressBar::new_spinner()
        .with_style(ProgressStyle::default_bar().template("{msg}: {elapsed:>10} {spinner:.green}"));
    progress_bar.set_message(msg);
    progress_bar.enable_steady_tick(100);
    let res = func();
    progress_bar.finish();
    res
}

/// Provides functionality to (de-)serialize optional bytes as base64 string.
///
/// Use it with the `#[serde(with=serde_opt_bytes_as_base64)]` annotation.
pub mod serde_opt_bytes_as_base64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bytes: &Option<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded = bytes.as_ref().map(base64::encode);
        encoded.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<u8>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        if let Some(encoded) = Option::<String>::deserialize(deserializer)? {
            base64::decode(encoded)
                .map(Some)
                .map_err(<D::Error as serde::de::Error>::custom)
        } else {
            Ok(None)
        }
    }
}
