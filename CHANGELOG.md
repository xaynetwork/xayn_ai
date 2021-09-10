# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to the [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [unreleased]

### Changed

- AI data is synchronizable between devices via the new functions `syncdata_bytes` and `synchronize`.
- Add multi-threading support: Parallelism is always enabled for mobile targets and can optionally be enabled for web targets. An optional list of features is introduced to pick the assets for the chosen target during the setup (see `getInputData()`), currently the only available feature is `webParallel` (sequential by default). The asset generation and publishing has been updated accordingly.

## 2.0.1 - 2021/09/07

### Changed

- Allow `query_count`s starting with 0, if possible they now also should start with 0.

## 2.0.0 - 2021/06/28

### Changed

- AI data is synchronizable between devices via the new functions `syncdata_bytes` and `synchronize`.
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
