import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show SetupData;

const baseUrl = 'http://127.0.0.1:8000/assets/';
final assets = [
  {
    'name': 'vocab.txt',
    'url': baseUrl + 'vocab.txt',
    'checksum':
        'sha384-qPJ/Z9jJLT2Qssyb5EPSnqFl/0MJasriyNbjC/IrsK1vT65x9vdetYCPlMF8Y8Dm'
  },
  {
    'name': 'smbert.onnx',
    'url': baseUrl + 'smbert.onnx',
    'checksum':
        'sha384-oS4xtjqZbqUnU1gdCroL5R8txtkfVjInr9JQCDYShOUU/2Rp7ip2SN5PZEJpGmNm'
  },
  {
    'name': 'xayn.wasm',
    'url': baseUrl + 'xayn.wasm',
    'checksum':
        'sha384-uYLreBsMHzsp+ZhLIBSzadjSBmAtBNkqovcEXULnIzlqidqYSXFJfvyFgazOevU/'
  },
];

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  final Uint8List vocab;
  final Uint8List model;

  SetupData(this.vocab, this.model);
}
