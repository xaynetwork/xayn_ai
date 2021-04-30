# Xayn-AI

## Models
To download the models use the `download_data.sh` script.

## Build
To build the library you just need to run `cargo build` in the root of the project.

To generate the dart ffi you need to run `flutter pub get` and `flutter pub run ffigen` in
the directory `bindings/dart`.

The project provides a `Makefile.toml` that can be run with `cargo make`.
You can install cargo make with:

```
cargo install --version 0.32.17 cargo-make
```

All of the above can be automatically done by `cargo make build`.

To build libraries for mobile targets you can use:

```
cargo make build-mobile
```

On Linux this will only build Android libraries, while on Mac it will build
for both Android and iOS.

### Android

To build for Android the following targets are needed:

```
rustup target add \
  aarch64-linux-android \
  armv7-linux-androideabi \
  x86_64-linux-android \
  i686-linux-android
```

Also `cargo-ndk` is needed:

```
cargo install --version 2.3.0 cargo-ndk
```

### iOS

To build for iOS the following targets are needed:

```
rustup target add \
  aarch64-apple-ios \
  x86_64-apple-ios
```

Also `cargo-lipo` is needed:

```
cargo install --version 3.1.1 cargo-lipo
```

### WASM

#### Prerequisites

- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- `rustup target add wasm32-unknown-unknown`
- [nodejs](https://nodejs.org/en/) (only if you want to run the tests on nodejs)

#### Building the WASM module

```
wasm-pack build
```

#### Running the WASM test

**Browser**

```
wasm-pack test --firefox --chrome --safari --headless
```

Note:

In `Safari` you first need to [enable the WebDriver support](https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari)
via `safaridriver --enable`, before you can run tests.

**nodejs**

```
wasm-pack test node -- --no-default-features --features=node
```

#### Running the example

```shell
cd xayn-ai-ffi-wasm
wasm-pack build --target web --release --no-typescript --out-dir example/pkg
cd example
python3 -m http.server
```

## License

This repository contains code from other software in the following
directories, licensed under their own particular licenses:

 * `rubert-tokenizer/*`: Apache2 ([LICENSE](rubert-tokenizer/LICENSE))

Xayn-ai and its components, unless otherwise stated, are licensed under
 * Xayn AG Transparency License ([LICENSE](LICENSE)).
