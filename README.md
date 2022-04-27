`xain_ai` is no longer actively maintained. Parts of the code have been copied to
[xayn_discovery_engine](https://github.com/xaynetwork/xayn_discovery_engine) and will be further developed there.

---

# Xayn-AI

## Models

To download the models use the `download_data.sh` script.

## Tools

In order to generate the code that contains the metadata of the assets,
you will need to install [`gomplate`](https://github.com/hairyhenderson/gomplate).

## Build

To build the library you just need to run `cargo build` in the root of the project.

To generate the dart ffi you need to run `flutter pub get` and `flutter pub run ffigen` in
the directory `bindings/dart`.

To update the non-ffi auto generated dart code after changes to dart code you need
to run `flutter pub run build_runner build`.

The project provides a `Makefile.toml` that can be run with `cargo make`.
You can install cargo make with:

```
cargo install --version 0.35.0 cargo-make
```

All of the above can be automatically done by `cargo make build`.

To build libraries for mobile targets you can use:

```
cargo make build-mobile
```

On Linux this will only build Android libraries, while on Mac it will build
for both Android and iOS.

To build the library for web you can use:

```
cargo make build-web
```

or

```
DISABLE_WASM_THREADS=1 cargo make build-web
```

to build a version which doesn't require the browser to
support `SharedArrayBuffer` and `Atomics`.

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
cargo install --version 2.4.1 cargo-ndk
```

You also need to install android ndk and might need to set the `ANDROID_NDK_HOME` variable.
If you install the android ndk through AndroidStudio you would need to set `ANDROID_NDK_HOME`
to something like `~/Android/Sdk/ndk/<ndk-version>` which for example could be
`/home/user/Android/Sdk/ndk/22.1.7171670/`.

### iOS

To build for iOS the following targets are needed:

```
rustup target add \
  aarch64-apple-ios \
  x86_64-apple-ios
```

### WASM

#### Prerequisites

- wasm-pack

```
cargo install --version 0.10.1 wasm-pack
```

- `rustup target add wasm32-unknown-unknown`
- [nodejs](https://nodejs.org/en/) (only if you want to run the tests on nodejs)
- [yarn](https://yarnpkg.com/)

All `wasm-pack` commands below are to be run in the directory `xayn-ai-ffi-wasm/`.

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
wasm-pack test --node -- --no-default-features --features=node
```

### Flutter example

#### Running on a mobile device/emulator

```shell
cargo make build-mobile
cd bindings/dart/example
flutter run
```

#### Running in Chrome

```shell
cargo make build-web
cargo make serve-web
# Then open http://localhost:8000/ in a browser.
```

`flutter run` can not be used as it doesn't set the
right headers, and even with the right headers it
doesn't provide the right files in the right way.

(At least for now `flutter run` can still be used
with `DISABLE_WASM_THREADS=1`, this is not guaranteed
in the future but can be useful as `flutter run` provides
hot reloading and better debug-ability.)

**Hint:** There had been some cases where for some reason
`genesis.js` wasn't updated when switching from `DISABLE_WASM_THREADS=0`
to `DISABLE_WASM_THREADS=1` it's not clear what causes it as
it's not reproducible but it was observed by different dev.
If that happens run `cargo make clean-non-rust`.

#### Running with a branch of the release repository

**bindings/dart/example/pubspec.yaml**

```diff
  xayn_ai_ffi_dart:
-   path: '../'
+   git:
+       url: git@github.com:xaynetwork/xayn_ai_release.git
+       ref: <branch>
```

**bindings/dart/example/lib/data_provider/web.dart**

```diff
- const _baseAssetUrl = 'assets';
+ const _baseAssetUrl = 'https://ai-assets.xaynet.dev';
```

Then use `cargo make build-web` and `cargo make serve-web` like above.

If flutter analyze fails there is a good chance that the local version
and the used branch in the release repo are not compatible.

## License

This repository contains code from other software in the following
directories, licensed under their own particular licenses:

 * `rubert-tokenizer/*`: Apache2 ([LICENSE](rubert-tokenizer/LICENSE))

Xayn-ai and its components, unless otherwise stated, are licensed under
 * AGPL-3.0 ([LICENSE](LICENSE)).
