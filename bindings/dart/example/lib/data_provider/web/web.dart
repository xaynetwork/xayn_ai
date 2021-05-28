import 'dart:html';
import 'dart:typed_data';
import 'package:xayn_ai_ffi_dart/package.dart' as xayn_ai
    show SetupData, assets;
import 'package:xayn_ai_ffi_dart/package.dart';

class SetupData implements xayn_ai.SetupData {
  final Uint8List vocab;
  final Uint8List model;

  SetupData(this.vocab, this.model);

  static Future<SetupData> getInputData() async {
    final vocab =
        xayn_ai.assets.where((element) => element['name'] == 'vocab.txt').first;
    final model = xayn_ai.assets
        .where((element) => element['name'] == 'smbert.onnx')
        .first;
    final wasm =
        xayn_ai.assets.where((element) => element['name'] == 'xayn.wasm').first;

    final vocabData = await _fetchAsset(vocab['url']!, vocab['checksum']!);
    final modelData = await _fetchAsset(model['url']!, model['checksum']!);

    // wasmbindgen's `init` function takes care of loading the WASM module.
    // interally it calls `WebAssembly.instantiateStreaming`
    // it would be nice if we could move the `init` call to XaynAI, however
    // init is async so we would need somthing on XaynAI to call that.
    await init(wasm['url']);
    return SetupData(vocabData, modelData);
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
