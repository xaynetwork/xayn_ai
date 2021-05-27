use rubert::{FirstPooler, SMBertBuilder};

fn main() {
    let smbert = SMBertBuilder::from_files(
        "../data/rubert_v0001/vocab.txt",
        "../data/rubert_v0001/smbert.onnx",
    )
    .unwrap()
    .with_accents(false)
    .with_lowercase(true)
    .with_token_size(64)
    .unwrap()
    .with_pooling(FirstPooler)
    .build()
    .unwrap();

    let embedding = smbert.run("This is a sequence.").unwrap();
    assert_eq!(embedding.shape(), [smbert.embedding_size()]);
}
