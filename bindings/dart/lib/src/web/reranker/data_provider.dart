import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show SetupData, Asset;

//https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity
const baseUrl = 'http://127.0.0.1:8000/assets/';
final assets = [
  common.Asset(
      name: 'vocab.txt',
      url: baseUrl + 'vocab.txt',
      checksum: 'sha256-/7I5iBD6l37jnooBA7B/nCYU8fmCXfU6LJnnFWnjRRw='),
  common.Asset(
      name: 'smbert.onnx',
      url: baseUrl + 'smbert.onnx',
      checksum: 'sha256-wHZTo8xWiHYHHeFX8PuMzBowIZfdrZ6tBpqGFA4Z2RM='),
  common.Asset(
      name: 'qabert.onnx',
      url: baseUrl + 'qabert.onnx',
      checksum: 'sha256-9nzYcZqT4fMkAvxgUdgp1vv1PnDoXHckNKoeAbCxwbM='),
  common.Asset(
      name: 'ltr.binparams',
      url: baseUrl + 'ltr.binparams',
      checksum: 'sha256-82qHS6JuoVSgIHAYDPyrrxkfZU7P9buUqEctKIHUwGI='),
  common.Asset(
      name: 'xayn.wasm',
      url: baseUrl + 'xayn.wasm',
      checksum: 'sha256-YxhDwNXamA0LeWLTXpbMpqr0J6FTRUZF5yTLIrQ/DHo='),
];

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  final Uint8List vocab;
  final Uint8List model;
  final Uint8List wasm;

  SetupData(this.vocab, this.model, this.wasm);
}
