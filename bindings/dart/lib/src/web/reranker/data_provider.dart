import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show Asset, AssetType, baseAssets, FeatureHint, SetupData;

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets([common.FeatureHint? hint]) {
  if (hint == null) {
    throw ArgumentError('requires a feature hint for web assets');
  }

  switch (hint) {
    case common.FeatureHint.wasmSequential:
      final wasmNonSequentialAssets = [
        common.AssetType.wasmParallelModule,
        common.AssetType.wasmParallelScript,
      ];
      return Map.fromEntries(common.baseAssets.entries.where(
          (asset) => wasmNonSequentialAssets.contains(asset.key) == false));
    case common.FeatureHint.wasmParallel:
      final wasmNonParallelAssets = [
        common.AssetType.wasmSequentialModule,
        common.AssetType.wasmSequentialScript,
      ];
      return Map.fromEntries(common.baseAssets.entries.where(
          (asset) => wasmNonParallelAssets.contains(asset.key) == false));
  }
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
      [common.FeatureHint? hint]) {
    if (hint == null) {
      throw ArgumentError('requires a feature hint for web assets');
    }

    smbertVocab = assets[common.AssetType.smbertVocab]!;
    smbertModel = assets[common.AssetType.smbertModel]!;
    qambertVocab = assets[common.AssetType.qambertVocab]!;
    qambertModel = assets[common.AssetType.qambertModel]!;
    ltrModel = assets[common.AssetType.ltrModel]!;
    switch (hint) {
      case common.FeatureHint.wasmSequential:
        wasmModule = assets[common.AssetType.wasmSequentialModule]!;
        break;
      case common.FeatureHint.wasmParallel:
        wasmModule = assets[common.AssetType.wasmParallelModule]!;
        break;
    }
  }
}
