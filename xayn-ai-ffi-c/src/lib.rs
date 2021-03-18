//! C FFI for the Xayn AI.
#![cfg_attr(doc, forbid(broken_intra_doc_links, private_intra_doc_links))]
#![allow(unused_unsafe)]

mod ai;
mod systems;
mod utils;

#[cfg(doc)]
pub use crate::{
    ai::{
        error_message_drop,
        xaynai_drop,
        xaynai_new,
        xaynai_rerank,
        CDocument,
        CXaynAi,
        CXaynAiError,
    },
    utils::dummy_function,
};

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        collections::hash_map::DefaultHasher,
        fs::File,
        hash::{Hash, Hasher},
        io::{BufReader, Read},
    };

    /// Path to the current vocabulary file.
    pub const VOCAB: &str = "../data/rubert_v0000/vocab.txt";

    /// Path to the current onnx model file.
    pub const MODEL: &str = "../data/rubert_v0000/model.onnx";

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
    fn test_vocab() {
        // assure that the vocab hasn't been changed accidentally
        assert_eq!(hash_file(VOCAB), 5045043227147541355);
    }

    #[test]
    fn test_model() {
        // assure that the model hasn't been changed accidentally
        assert_eq!(hash_file(MODEL), 13727150546539837987);
    }
}
