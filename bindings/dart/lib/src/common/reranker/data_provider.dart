import 'dart:convert' show base64;
import 'package:hex/hex.dart' show HEX;

part 'assets.dart';

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
