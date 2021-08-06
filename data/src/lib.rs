//! The single source of truth for all data paths.

pub mod bench;
pub mod example;
pub mod ltr;
pub mod qambert;
pub mod smbert;
pub mod test;

use std::{
    env::{current_dir, var_os},
    io::{Error, ErrorKind, Result},
    path::{Path, PathBuf},
};

/// Tries to resolve the path to the requested data.
fn resolve_path(dir: impl AsRef<Path>, file: impl AsRef<Path>) -> Result<PathBuf> {
    let manifest = var_os("CARGO_MANIFEST_DIR")
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "missing CARGO_MANIFEST_DIR"))?;
    let manifest = PathBuf::from(manifest).canonicalize()?;
    let current = current_dir()?.canonicalize()?;

    let workspace = if current == manifest {
        current
            .parent()
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "missing workspace"))?
    } else if Some(current.as_path()) == manifest.parent() {
        current.as_path()
    } else {
        return Err(Error::new(ErrorKind::NotFound, "missing workspace"));
    };

    workspace.join("data").join(dir).join(file).canonicalize()
}
