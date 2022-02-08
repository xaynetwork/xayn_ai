use std::{
    collections::HashMap,
    env::var_os,
    fs::File,
    io::{BufReader, Error, ErrorKind, Result},
    path::{Path, PathBuf},
};

use serde::Deserialize;
use serde_json::from_reader;

pub const DATA_DIR: &str = "data";

/// Resolves the path to the requested data relative to the workspace directory.
pub fn resolve_path(path: &[impl AsRef<Path>]) -> Result<PathBuf> {
    let manifest = var_os("CARGO_MANIFEST_DIR")
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "missing CARGO_MANIFEST_DIR"))?;
    let workspace = PathBuf::from(manifest)
        .parent()
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "missing cargo workspace dir"))?
        .to_path_buf();

    path.iter()
        .fold(workspace, |path, component| path.join(component))
        .canonicalize()
}

#[derive(Deserialize)]
struct Asset {
    #[serde(rename(deserialize = "id"))]
    name: String,
    url_suffix: String,
}

#[derive(Deserialize)]
struct Assets {
    data_assets: Vec<Asset>,
}

/// Reads the asset paths from the static assets file.
fn read_assets() -> Result<HashMap<String, PathBuf>> {
    from_reader::<_, Assets>(BufReader::new(File::open(resolve_path(&[
        "assets_manifest.json",
    ])?)?))
    .map(|assets| {
        assets
            .data_assets
            .into_iter()
            .map(|asset| (asset.name, [DATA_DIR, &asset.url_suffix].iter().collect()))
            .collect()
    })
    .map_err(|error| Error::new(ErrorKind::InvalidData, error.to_string()))
}

/// Resolves the path to the requested asset relative to the workspace directory.
pub fn resolve_asset(asset: &str) -> Result<PathBuf> {
    resolve_path(&[read_assets()?
        .get(asset)
        .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("missing asset '{}'", asset)))?])
}
