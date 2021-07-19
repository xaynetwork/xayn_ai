//! Run as `cargo run --example mbert <kind>` with `<kind>`:
//! - `s` for SMBert
//! - `qa` for QAMBert

use rubert::{FirstPooler, QAMBertBuilder, SMBertBuilder};

macro_rules! build_and_run {
    ($builder:expr) => {{
        let mbert = $builder
            .unwrap()
            .with_accents(false)
            .with_lowercase(true)
            .with_token_size(64)
            .unwrap()
            .with_pooling(FirstPooler)
            .build()
            .unwrap();
        (
            mbert.run("This is a sequence.").unwrap(),
            mbert.embedding_size(),
        )
    }};
}

fn main() {
    let (embedding, size) = match std::env::args().nth(1).unwrap().as_str() {
        "s" => {
            build_and_run!(SMBertBuilder::from_files(
                "../data/smbert_v0000/vocab.txt",
                "../data/smbert_v0000/smbert.onnx",
            ))
        }
        "qa" => {
            build_and_run!(QAMBertBuilder::from_files(
                "../data/qambert_v0001/vocab.txt",
                "../data/qambert_v0001/qambert.onnx",
            ))
        }
        _ => panic!("unknown MBert kind"),
    };
    println!("{}", *embedding);
    assert_eq!(embedding.shape(), [size]);
}
