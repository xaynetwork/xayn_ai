import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show SetupData, Asset, AssetType;

const baseUrl = 'http://192.168.1.9:8000/assets/';
final assets = {
  common.AssetType.vocab: common.Asset(
      url: baseUrl + 'vocab.txt',
      checksum:
          'ffb2398810fa977ee39e8a0103b07f9c2614f1f9825df53a2c99e71569e3451c'),
  common.AssetType.smbert: common.Asset(
      url: baseUrl + 'smbert.onnx',
      checksum:
          'c07653a3cc568876071de157f0fb8ccc1a302197ddad9ead069a86140e19d913'),
  common.AssetType.qabert: common.Asset(
      url: baseUrl + 'qabert.onnx',
      checksum:
          'f67cd8719a93e1f32402fc6051d829d6fbf53e70e85c772434aa1e01b0b1c1b3'),
  common.AssetType.ltr: common.Asset(
      url: baseUrl + 'ltr.binparams',
      checksum:
          'f36a874ba26ea154a02070180cfcabaf191f654ecff5bb94a8472d2881d4c062'),
};

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  final String vocab;
  final String model;

  SetupData(this.vocab, this.model);
}
