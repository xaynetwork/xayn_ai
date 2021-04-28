# Xayn AI WASM

## Prerequisites

- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- `rustup target add wasm32-unknown-unknown`
- [nodejs](https://nodejs.org/en/) (only if you want to run the tests on nodejs)

## Building the WASM module

```
wasm-pack build
```

## Running the WASM test

**Browser**

```
wasm-pack test --firefox --chrome --safari --headless
```

**Note:**

In `Safari` you first need to [enable the WebDriver support](https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari)
via `safaridriver --enable`, before you can run tests.

**nodejs**

```
wasm-pack test node -- --no-default-features --features=node
```

## Running the example

```shell
ln -s $(pwd)/../data/ $(pwd)/example/data
wasm-pack build --target web --release --no-typescript --out-dir example/pkg
cd example
python3 -m http.server
```
