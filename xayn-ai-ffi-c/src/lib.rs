//! C FFI for the Xayn AI.
#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]
#![allow(unused_unsafe)]

pub mod data;
pub mod reranker;
pub mod result;
mod slice;
pub mod utils;

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use std::{
        collections::hash_map::DefaultHasher,
        fs::File,
        hash::{Hash, Hasher},
        io::{BufReader, Read},
    };

    /// Path to the current vocabulary file.
    pub const VOCAB: &str = "../data/rubert_v0001/vocab.txt";

    /// Path to the current onnx model file.
    pub const SMBERT_MODEL: &str = "../data/rubert_v0001/smbert.onnx";

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn hash_file(file: &str) -> u64 {
        let mut reader = BufReader::new(File::open(file).unwrap());
        let mut buffer = vec![0; 1024];
        let mut hasher = DefaultHasher::new();
        loop {
            if reader.read(&mut buffer).unwrap() == 0 {
                break;
            }
            buffer.hash(&mut hasher);
        }
        hasher.finish()
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_vocab_unchanged() {
        assert_eq!(hash_file(VOCAB), 5045043227147541355);
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_model_unchanged() {
        assert_eq!(hash_file(SMBERT_MODEL), 13727150546539837987);
    }
}
