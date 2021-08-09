use std::{io::Result, path::PathBuf};

use crate::resolve_path;

const ASSET: &str = "bench_matmul_v0000";

/// Resolves the path to the matrix multiplication benchmark data.
pub fn data() -> Result<PathBuf> {
    resolve_path(&["data", ASSET])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data() {
        assert!(data().is_ok());
    }
}
