This crate is a fork of: https://github.com/huggingface/tokenizers.

We removed all the cli parts, the multithreaded features and every non-BERT-related part to make it more compact and
to allow it to work on wasm.
