# Xayn AI WASM

## Prerequisites

- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- `rustup target add wasm32-unknown-unknown`

## Building the WASM module

```
wasm-pack build
```

## Running the WASM test

```
wasm-pack test --node
```

## Running the example

```shell
ln -s $(pwd)/../data/ $(pwd)/example/data
wasm-pack build --target web --release --no-typescript --out-dir example/pkg
cd example
python3 -m http.server
```
