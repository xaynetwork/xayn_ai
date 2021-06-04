import 'dart:convert' show base64;
import 'package:hex/hex.dart' show HEX;

/// Base assets that are required for both mobile and web.
///
/// The checksum is the SRI hash of the asset.
/// https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity#tools_for_generating_sri_hashes
final baseAssets = <AssetType, Asset>{
  AssetType.vocab: Asset('rubert_v0001/vocab.txt',
      'sha256-/7I5iBD6l37jnooBA7B/nCYU8fmCXfU6LJnnFWnjRRw='),
  AssetType.smbert: Asset('rubert_v0001/smbert.onnx',
      'sha256-wHZTo8xWiHYHHeFX8PuMzBowIZfdrZ6tBpqGFA4Z2RM='),
  AssetType.qambert: Asset('rubert_v0001/qambert.onnx',
      'sha256-9nzYcZqT4fMkAvxgUdgp1vv1PnDoXHckNKoeAbCxwbM='),
  AssetType.ltr: Asset('ltr_v0000/ltr.binparams',
      'sha256-82qHS6JuoVSgIHAYDPyrrxkfZU7P9buUqEctKIHUwGI='),
};

enum AssetType { vocab, smbert, qambert, ltr, wasm }

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
  SetupData(dynamic vocab, dynamic smbertModel, dynamic qambertModel,
      [dynamic wasm]);
}
