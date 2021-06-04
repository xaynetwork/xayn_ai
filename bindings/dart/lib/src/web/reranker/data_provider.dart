import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show Asset, AssetType, baseAssets, SetupData;

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets() {
  return common.baseAssets;
}

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  final Uint8List smbertVocab;
  final Uint8List smbertModel;
  final Uint8List qambertVocab;
  final Uint8List qambertModel;
  final Uint8List wasm;

  SetupData(this.smbertVocab, this.smbertModel, this.qambertVocab,
      this.qambertModel, this.wasm);
}
