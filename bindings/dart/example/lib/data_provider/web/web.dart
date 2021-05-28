import 'dart:html';
import 'dart:typed_data';
import 'package:xayn_ai_ffi_dart/package.dart' as xayn_ai
    show SetupData, assets;

class DataProvider {
  static Future<xayn_ai.SetupData> getInputData() async {
    final toFetch = <String, Uint8List>{};
    for (var asset in xayn_ai.assets) {
      final data = await _fetchAsset(asset['url']!, asset['checksum']!);
      toFetch.putIfAbsent(asset['name']!, () => data);
    }

    return xayn_ai.SetupData(
        toFetch['vocab.txt']!, toFetch['smbert.onnx']!, toFetch['xayn.wasm']!);
  }

  static Future<Uint8List> _fetchAsset(String url, String checksum) async {
    try {
      final dynamic responseModel =
          await window.fetch(url, <String, String>{'integrity': checksum});
      final arrayBuffer = await responseModel.arrayBuffer() as ByteBuffer;
      return Uint8List.view(arrayBuffer);
    } catch (e) {
      return Future.error('error loading asset: $url, error: $e');
    }
  }
}
