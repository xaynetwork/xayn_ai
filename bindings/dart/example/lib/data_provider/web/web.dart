import 'dart:html';
import 'dart:typed_data';
import 'package:xayn_ai_ffi_dart/package.dart'
    show assets, AssetType, SetupData;

Future<SetupData> getInputData() async {
  final fetched = <AssetType, Uint8List>{};
  for (var asset in assets.entries) {
    final data = await _fetchAsset(asset.value.url!, asset.value.checksum!);
    fetched.putIfAbsent(asset.key, () => data);
  }

  return SetupData(fetched[AssetType.vocab]!, fetched[AssetType.smbert]!,
      fetched[AssetType.wasm]!);
}

Future<Uint8List> _fetchAsset(String url, String checksum) async {
  try {
    final dynamic responseModel =
        await window.fetch(url, <String, String>{'integrity': checksum});
    final arrayBuffer = await responseModel.arrayBuffer() as ByteBuffer;
    return Uint8List.view(arrayBuffer);
  } catch (e) {
    return Future.error('error loading asset: $url, error: $e');
  }
}
