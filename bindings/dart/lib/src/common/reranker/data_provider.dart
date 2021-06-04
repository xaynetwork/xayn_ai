import 'dart:convert' show base64;
import 'package:hex/hex.dart' show HEX;

/// Base assets that are required for both mobile and web.
///
/// The checksum is the SRI hash of the asset.
/// https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity#tools_for_generating_sri_hashes
final baseAssets = <AssetType, Asset>{
  AssetType.smbertVocab: Asset('smbert_v0000/vocab.txt',
      'ffb2398810fa977ee39e8a0103b07f9c2614f1f9825df53a2c99e71569e3451c'),
  AssetType.smbertModel: Asset('smbert_v0000/smbert.onnx',
      'c07653a3cc568876071de157f0fb8ccc1a302197ddad9ead069a86140e19d913'),
  AssetType.qambertVocab: Asset('qambert_v0000/vocab.txt',
      'ffb2398810fa977ee39e8a0103b07f9c2614f1f9825df53a2c99e71569e3451c'),
  AssetType.qambertModel: Asset('qambert_v0000/qambert.onnx',
      'f67cd8719a93e1f32402fc6051d829d6fbf53e70e85c772434aa1e01b0b1c1b3'),
};

enum AssetType {
  smbertVocab,
  smbertModel,
  qambertVocab,
  qambertModel,
  wasmModule
}

class Asset {
  late String suffix;
  late String _checksum;

  Asset(String suffix, String checksum) {
    this.suffix = suffix;
    _checksum = checksum;
  }

  /// Returns the Sha256 hash (hex-encoded) of the asset.
  String getChecksumAsHex() {
    return HEX.encode(base64.decode(_checksum.split('-').last));
  }

  /// Returns the SRI hash of the asset.
  String getChecksumSri() {
    return _checksum;
  }
}

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<AssetType, Asset> getAssets() {
  throw UnsupportedError('Unsupported platform.');
}

/// Data that can be used to initialize [`XaynAi`].
class SetupData {
  SetupData(dynamic smbertVocab, dynamic smbertModel, dynamic qambertVocab,
      dynamic qambertModel,
      [dynamic wasm]);
}
