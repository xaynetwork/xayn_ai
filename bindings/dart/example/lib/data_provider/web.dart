import 'dart:html' show window;
import 'dart:typed_data' show Uint8List, ByteBuffer;

import 'package:xayn_ai_ffi_dart/package.dart'
    show AssetType, getAssets, SetupData, WebFeature;

import 'package:xayn_ai_ffi_dart_example/data_provider/data_provider.dart'
    show joinPaths;

const _baseAssetUrl = 'assets/assets';

/// Prepares and returns the data that is needed to init [`XaynAi`].
Future<SetupData> getInputData() async {
  final fetched = <AssetType, Uint8List>{};
  final features = <WebFeature>{};

  // uncomment the following section to load the parallel version
  // final features = <WebFeature>{
  //   WasmFeature.bulkMemory,
  //   WasmFeature.mutableGlobals,
  //   WasmFeature.threads
  // };

  for (var asset in getAssets(features: features).entries) {
    final path = joinPaths([_baseAssetUrl, asset.value.urlSuffix]);
    final data = await _fetchAsset(path, asset.value.checksum.checksumSri);
    fetched.putIfAbsent(asset.key, () => data);
  }

  return SetupData(fetched);
}

Future<Uint8List> _fetchAsset(String url, String checksum) async {
  try {
    final dynamic response =
        await window.fetch(url, <String, String>{'integrity': checksum});
    final arrayBuffer = await response.arrayBuffer() as ByteBuffer;
    return Uint8List.view(arrayBuffer);
  } catch (e) {
    return Future.error('error loading asset: $url, error: $e');
  }
}
