use rubert::{FirstPooler, QAMBertBuilder};

fn main() {
    let qambert = QAMBertBuilder::from_files(
        "../data/qambert_v0001/vocab.txt",
        "../data/qambert_v0001/qambert.onnx",
    )
    .unwrap()
    .with_accents(false)
    .with_lowercase(true)
    .with_token_size(64)
    .unwrap()
    .with_pooling(FirstPooler)
    .build()
    .unwrap();

    let embedding = qambert.run("This is a sequence.").unwrap();
    assert_eq!(embedding.shape(), [qambert.embedding_size()]);
}
