# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to the [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [unreleased]

### Changed

- `wasmScript` is no longer a javascript module.
- Functions of `wasmScript` do not need to be exported to `window.xayn_ai_ffi_wasm` anymore.

## 3.0 - 2021/11/02

### Added

- AI data is synchronizable between devices via the new functions `syncdata_bytes` and `synchronize`.
- Add multi-threading support: Parallelism is always enabled for mobile targets and can optionally be enabled for web targets.
  An optional set of `Feature`s is introduced to pick the assets for the chosen target during the setup (see `getInputData()` in the example).
  `Feature` are currently only supported on the web: the caller needs to pass the set of available features on the browser to the `getAssets`
  function and the library automatically returns the most suitable WASM assets for the given features.
  For the AI to run in parallel on the web it is required that it runs on a WebWorker.
- Add methods to instantiate the AI from a state: `restore`.
- All methods of `XaynAi` have been made asynchronous.
- The `RerankMode`s of `XaynAi.rerank()` have been extended with unpersonalized variants.
- MAB system is not executed anymore. Only the context value is used instead to rank the documents.
- For standard search QA-mBert is used to rerank.
- The AI does not fail at the first error but try to rerank the documents with the remaining systems.

### Changed

- The method `create` does not accept anymore a serialized state and it can only be used to create a clean AI.

### Removed

- Removed the optional `features` parameter in the `SetupData` constructor.
- Removed the `wasmParallelX` `AssetType`s. Both parallel and sequential WASM assets share now the same `AssetType` (`wasmModule`/`wasmScript`).

## 2.0.1 - 2021/09/07

### Fixed

- Allow `query_count`s starting with 0, if possible they now also should start with 0.

## 2.0.0 - 2021/06/28

### Changed

- Outsourcing of ai assets. The assets are no longer part of the library. See [#94](https://github.com/xaynetwork/xayn_ai/pull/94) for more information about the API changes.
- `XaynAi.rerank` takes as parameter `RerankMode` that allows to specify if we are reranking news or results from a search.
- Assets for releases on the `staging` branch are uploaded to a S3 bucket (`https://xayn_ai_staging_assets.s3-de-central.profitbricks.com`).
- Renamed `Feedback` to `UserFeedback` and `History.feedback` to `History.userFeedback`.
- Use keyword arguments for `Document`/`History` constructors.
- `XainAi` is initialized with an additional parameter to support the domain reranker.
- Assets for releases on the `release` branch are uploaded to KeyCDN.
- The token size of smbert has been decreased to `52`.

## 1.0.0 - 2021/05/18

The first release of oxidized **XAYN AI**.
