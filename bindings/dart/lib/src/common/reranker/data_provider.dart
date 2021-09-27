import 'dart:convert' show base64;
import 'package:hex/hex.dart' show HEX;

part 'assets.dart';

/// An asset consists of an URL suffix, a [`Checksum`] and optionally
/// a list of [`Fragment`]s.
///
/// The base URL (defined by the caller) concatenated with the URL suffix
/// creates the URL to fetch an asset.
///
/// The checksum is the hash of an asset and can be used to verify its
/// integrity after it has been fetched.
///
/// In order to keep larger assets in the http cache of a browser,
/// an asset might be split into multiple fragments.
///
/// Implementation details for fetching assets:
///
/// If the list of fragments is empty, the caller must use the URL suffix of the
/// asset to fetch it.
///
/// If the list of fragments is not empty, the caller must fetch each
/// [`Fragment`] in the fragments list and concatenate them in the same order
/// as they are defined in the fragments list in order to reassemble the asset.
/// Using the URL suffix of the [`Asset`] is not allowed. The checksum of the
/// [`Asset`] can be used to to verify its integrity after it has been
/// reassembled.
class Asset {
  final String urlSuffix;
  final Checksum checksum;
  final List<Fragment> fragments;

  Asset(this.urlSuffix, this.checksum, this.fragments);
}

/// A fragment of an asset.
class Fragment {
  final String urlSuffix;
  final Checksum checksum;

  Fragment(this.urlSuffix, this.checksum);
}

/// The checksum an asset/fragment.
class Checksum {
  final String _checksum;

  Checksum(this._checksum);

  /// Returns the sha256 hash (hex-encoded) of the asset/fragment.
  String get checksumAsHex => _checksum;

  /// Returns the SRI hash of the asset/fragment.
  /// https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity#tools_for_generating_sri_hashes
  String get checksumSri => 'sha256-' + base64.encode(HEX.decode(_checksum));
}

/// The optional features to be enabled by picking platform dependent assets.
class Feature {
  // The constructor is private and is only callable within this file.
  // We use the constructor only for creating the unique `WebFeature`/`WasmFeature`s.
  Feature._();
}

class MobileFeature extends Feature {
  MobileFeature._() : super._() {
    // make sure no instance of this class can exists
    throw UnimplementedError('This class cannot be instantiated');
  }
}

class WebFeature extends Feature {
  WebFeature._() : super._();
}

// When we compare `WasmFeature`s with each other, we rely on their object identity.
// To avoid unexpected results when comparing them, we have to make sure that a
// `WebFeature` cannot be created by a user but that the user only uses
// `WasmFeature`.
class WasmFeature {
  static final bulkMemory = WebFeature._();
  static final mutableGlobals = WebFeature._();
  static final threads = WebFeature._();
}

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<AssetType, Asset> getAssets({Set<Feature> features = const {}}) {
  throw UnsupportedError('Unsupported platform.');
}

/// Data that is required to initialize [`XaynAi`].
class SetupData {
  SetupData(Map<AssetType, dynamic> assets) {
    throw UnsupportedError('Unsupported platform.');
  }
}
