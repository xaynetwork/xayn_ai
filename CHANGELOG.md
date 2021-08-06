# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to the [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [unreleased]

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

## 1.0.0 - 2021/05/18

The first release of oxidized **XAYN AI**.
