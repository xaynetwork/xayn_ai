use kpe::{Config, Pipeline};

use test_utils::kpe::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config =
        Config::from_files(vocab()?, bert()?, cnn()?, classifier()?)?.with_token_size(128)?;

    let kpe = Pipeline::from(config)?;

    let key_phrases = kpe.run("This sequence will be split into key phrases.")?;
    println!("{:?}", key_phrases);
    assert_eq!(key_phrases.len(), 30);

    Ok(())
}
