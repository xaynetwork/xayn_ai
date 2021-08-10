use std::{io::Result, path::PathBuf};

use crate::resolve_path;

const ASSET: &str = "bench_matmul_v0000";

/// Resolves the path to the matrix multiplication benchmark data.
pub fn data_dir() -> Result<PathBuf> {
    resolve_path(&["data", ASSET])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_dir() {
        assert!(data_dir().is_ok());
    }
}
