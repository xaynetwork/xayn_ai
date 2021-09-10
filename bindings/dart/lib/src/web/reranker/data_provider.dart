import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show Asset, AssetType, baseAssets, SetupData;

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets(
    {List<String> features = const []}) {
  final redundantAssets = features.contains('webParallel')
      ? [
          common.AssetType.wasmSequentialModule,
          common.AssetType.wasmSequentialScript,
        ]
      : [
          common.AssetType.wasmParallelModule,
          common.AssetType.wasmParallelScript,
          common.AssetType.wasmParallelSnippet,
        ];
  return Map.fromEntries(common.baseAssets.entries
      .where((asset) => redundantAssets.contains(asset.key) == false));
}

/// Data that is required to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  late Uint8List smbertVocab;
  late Uint8List smbertModel;
  late Uint8List qambertVocab;
  late Uint8List qambertModel;
  late Uint8List ltrModel;
  late Uint8List wasmModule;

  SetupData(Map<common.AssetType, Uint8List> assets,
      {List<String> features = const []}) {
    smbertVocab = assets[common.AssetType.smbertVocab]!;
    smbertModel = assets[common.AssetType.smbertModel]!;
    qambertVocab = assets[common.AssetType.qambertVocab]!;
    qambertModel = assets[common.AssetType.qambertModel]!;
    ltrModel = assets[common.AssetType.ltrModel]!;
    wasmModule = features.contains('webParallel')
        ? assets[common.AssetType.wasmParallelModule]!
        : assets[common.AssetType.wasmSequentialModule]!;
  }
}
