import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show Asset, AssetType, baseAssets, SetupData;

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets() {
  return common.baseAssets;
}

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  late Uint8List smbertVocab;
  late Uint8List smbertModel;
  late Uint8List qambertVocab;
  late Uint8List qambertModel;
  late Uint8List wasmModule;

  SetupData(Map<common.AssetType, Uint8List> assets) {
    smbertVocab = assets[common.AssetType.smbertVocab]!;
    smbertModel = assets[common.AssetType.smbertModel]!;
    qambertVocab = assets[common.AssetType.qambertVocab]!;
    qambertModel = assets[common.AssetType.qambertModel]!;
    wasmModule = assets[common.AssetType.wasmModule]!;
  }
}
