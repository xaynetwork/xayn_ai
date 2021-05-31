import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show SetupData, Asset, AssetType;

//https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity
const baseUrl = 'http://127.0.0.1:8000/assets/';
final assets = {
  common.AssetType.vocab: common.Asset(
      url: baseUrl + 'vocab.txt',
      checksum: 'sha256-/7I5iBD6l37jnooBA7B/nCYU8fmCXfU6LJnnFWnjRRw='),
  common.AssetType.smbert: common.Asset(
      url: baseUrl + 'smbert.onnx',
      checksum: 'sha256-wHZTo8xWiHYHHeFX8PuMzBowIZfdrZ6tBpqGFA4Z2RM='),
  common.AssetType.qabert: common.Asset(
      url: baseUrl + 'qambert.onnx',
      checksum: 'sha256-9nzYcZqT4fMkAvxgUdgp1vv1PnDoXHckNKoeAbCxwbM='),
  common.AssetType.ltr: common.Asset(
      url: baseUrl + 'ltr.binparams',
      checksum: 'sha256-82qHS6JuoVSgIHAYDPyrrxkfZU7P9buUqEctKIHUwGI='),
  common.AssetType.wasm: common.Asset(
      url: baseUrl + 'xayn.wasm',
      checksum: 'sha256-YxhDwNXamA0LeWLTXpbMpqr0J6FTRUZF5yTLIrQ/DHo='),
};

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  final Uint8List vocab;
  final Uint8List model;
  final Uint8List wasm;

  SetupData(this.vocab, this.model, this.wasm);
}
