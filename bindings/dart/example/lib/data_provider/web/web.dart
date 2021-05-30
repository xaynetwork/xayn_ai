import 'dart:html';
import 'dart:typed_data';
import 'package:xayn_ai_ffi_dart/package.dart' show SetupData, assets;

Future<SetupData> getInputData() async {
  final fetched = <String, Uint8List>{};
  for (var asset in assets) {
    final data = await _fetchAsset(asset.url!, asset.checksum!);
    fetched.putIfAbsent(asset.name!, () => data);
  }

  return SetupData(
      fetched['vocab.txt']!, fetched['smbert.onnx']!, fetched['xayn.wasm']!);
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
