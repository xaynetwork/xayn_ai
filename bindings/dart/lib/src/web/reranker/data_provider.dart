import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show SetupData;

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  final Uint8List smbertVocab;
  final Uint8List smbertModel;
  final Uint8List qambertVocab;
  final Uint8List qambertModel;

  SetupData(
      this.smbertVocab, this.smbertModel, this.qambertVocab, this.qambertModel);
}
