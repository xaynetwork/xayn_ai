import 'dart:convert' show base64;
import 'package:hex/hex.dart' show HEX;

/// Base assets that are required for both mobile and web.
///
/// The checksum is the sha256 hash of the asset.
/// To calculate the checksum run 'shasum -a 256 vocab.txt'.
final baseAssets = <AssetType, Asset>{
  AssetType.smbertVocab: Asset('smbert_v0000/vocab.txt',
      'ffb2398810fa977ee39e8a0103b07f9c2614f1f9825df53a2c99e71569e3451c'),
  AssetType.smbertModel: Asset('smbert_v0000/smbert.onnx',
      'c07653a3cc568876071de157f0fb8ccc1a302197ddad9ead069a86140e19d913'),
  AssetType.qambertVocab: Asset('qambert_v0001/vocab.txt',
      'ffb2398810fa977ee39e8a0103b07f9c2614f1f9825df53a2c99e71569e3451c'),
  AssetType.qambertModel: Asset('qambert_v0001/qambert.onnx',
      '030e7d68cc82f59640c6b989f145d6eaa7609c8615bd6ae9e7890e1a68be5b71'),
};

enum AssetType {
  smbertVocab,
  smbertModel,
  qambertVocab,
  qambertModel,
  wasmModule
}

class Asset {
  final String suffix;
  final String _checksum;

  Asset(this.suffix, this._checksum);

  /// Returns the sha256 hash (hex-encoded) of the asset.
  String get checksumAsHex => _checksum;

  /// Returns the SRI hash of the asset.
  /// https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity#tools_for_generating_sri_hashes
  String get checksumSri => 'sha256-' + base64.encode(HEX.decode(_checksum));
}

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<AssetType, Asset> getAssets() {
  throw UnsupportedError('Unsupported platform.');
}

/// Data that can be used to initialize [`XaynAi`].
class SetupData {
  SetupData(Map<AssetType, dynamic> assets);
}
