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

All the above can be automatically done by `cargo make build`.

To build libraries for mobile targets you can use:
```
cargo make build-mobile
```
On Linux this will only build Android libraries, while on Mac it will build
for both Android and iOS.

### Android

To build for Android the following targets are needed:

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

## License

This repository contains code from other software in the following
directories, licensed under their own particular licenses:

 * `rubert-tokenizer/*`: Apache2 ([LICENSE](rubert-tokenizer/LICENSE))

Xayn-ai and its components, unless otherwise stated, are licensed under
 * Xayn AG Transparency License ([LICENSE](LICENSE)).
