//! Run as `cargo run --example mbert <kind>` with `<kind>`:
//! - `s` for SMBert
//! - `qa` for QAMBert

use rubert::{Config, FirstPooler, Pipeline, QAMBertConfig, SMBertConfig};
use test_utils::{qambert, smbert};

fn main() {
    let (embedding, size) = match std::env::args().nth(1).unwrap().as_str() {
        "s" => {
            let config: SMBertConfig<_> =
                Config::from_files(smbert::vocab().unwrap(), smbert::model().unwrap())
                    .unwrap()
                    .with_pooling(FirstPooler)
                    .with_token_size(64)
                    .unwrap();

            let mbert = Pipeline::from(config).unwrap();
            (
                mbert.run("This is a sequence.").unwrap(),
                mbert.embedding_size(),
            )
        }
        "qa" => {
            let config: QAMBertConfig<_> =
                Config::from_files(qambert::vocab().unwrap(), qambert::model().unwrap())
                    .unwrap()
                    .with_pooling(FirstPooler);

            let mbert = Pipeline::from(config).unwrap();
            (
                mbert.run("This is a sequence.").unwrap(),
                mbert.embedding_size(),
            )
        }
        _ => panic!("unknown MBert kind"),
    };
    println!("{}", *embedding);
    assert_eq!(embedding.shape(), [size]);
}
