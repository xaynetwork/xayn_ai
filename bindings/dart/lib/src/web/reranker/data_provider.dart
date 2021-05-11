import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show SetupData;

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  late Uint8List vocab;
  late Uint8List model;

  SetupData(this.vocab, this.model);
}
