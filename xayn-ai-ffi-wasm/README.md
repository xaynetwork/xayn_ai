# Xayn AI WASM

## Prerequisites

- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- `rustup target add wasm32-unknown-unknown`

## Build

`wasm-pack build`

## Test

`wasm-pack test --node`

## Run in the browser

```
ln -s $(pwd)/../data/ $(pwd)/data
wasm-pack build --target web --release
python3 -m http.server
```
