//! The single source of truth for all data paths.
//!
//! Run `cargo make gen-assets` before using this!

pub mod bench;
pub mod example;
pub mod ltr;
pub mod qambert;
pub mod smbert;
pub mod test;

use std::{
    collections::HashMap,
    env::{current_dir, var_os},
    fs::File,
    io::{BufReader, Error, ErrorKind, Result},
    path::{Path, PathBuf},
};

use serde::Deserialize;
use serde_json::{from_reader, from_value, Value};

/// Resolves the path to the requested data.
fn resolve_path(path: &[impl AsRef<Path>]) -> Result<PathBuf> {
    let manifest = var_os("CARGO_MANIFEST_DIR")
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "missing CARGO_MANIFEST_DIR"))?;
    let manifest = PathBuf::from(manifest).canonicalize()?;
    let current = current_dir()?.canonicalize()?;

    let workspace = if current == manifest {
        current
            .parent()
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "missing workspace"))?
            .to_path_buf()
    } else if Some(current.as_path()) == manifest.parent() {
        current
    } else {
        return Err(Error::new(ErrorKind::NotFound, "missing workspace"));
    };

    path.iter()
        .fold(workspace, |path, component| path.join(component))
        .canonicalize()
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    path: String,
}

#[derive(Deserialize)]
struct Assets {
    assets: Vec<Value>,
}

/// Reads the asset paths.
fn read_assets() -> Result<HashMap<String, PathBuf>> {
    from_reader::<_, Assets>(BufReader::new(File::open(resolve_path(&[
        "out",
        "assets.json",
    ])?)?))
    .map(|assets| {
        assets
            .assets
            .into_iter()
            .filter_map(|asset| {
                from_value::<Asset>(asset)
                    .map(|asset| (asset.name, asset.path.into()))
                    .ok()
            })
            .collect()
    })
    .map_err(|error| Error::new(ErrorKind::InvalidData, error.to_string()))
}

/// Resolves the path to the requested asset.
fn resolve_asset(asset: &str) -> Result<PathBuf> {
    resolve_path(&[read_assets()?
        .get(asset)
        .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("missing asset '{}'", asset)))?])
}
