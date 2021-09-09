import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show Asset, AssetType, baseAssets, FeatureHint, SetupData;

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets(common.FeatureHint hint) {
  switch (hint) {
    case common.FeatureHint.webSequential:
      final wasmNonSequentialAssets = [
        common.AssetType.wasmParallelModule,
        common.AssetType.wasmParallelScript,
        common.AssetType.wasmParallelSnippet,
      ];
      return Map.fromEntries(common.baseAssets.entries.where(
          (asset) => wasmNonSequentialAssets.contains(asset.key) == false));
    case common.FeatureHint.webParallel:
      final wasmNonParallelAssets = [
        common.AssetType.wasmSequentialModule,
        common.AssetType.wasmSequentialScript,
      ];
      return Map.fromEntries(common.baseAssets.entries.where(
          (asset) => wasmNonParallelAssets.contains(asset.key) == false));
    default:
      throw ArgumentError('unsupported feature hint for web: $hint');
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

  SetupData(Map<common.AssetType, Uint8List> assets, common.FeatureHint hint) {
    smbertVocab = assets[common.AssetType.smbertVocab]!;
    smbertModel = assets[common.AssetType.smbertModel]!;
    qambertVocab = assets[common.AssetType.qambertVocab]!;
    qambertModel = assets[common.AssetType.qambertModel]!;
    ltrModel = assets[common.AssetType.ltrModel]!;
    switch (hint) {
      case common.FeatureHint.webSequential:
        wasmModule = assets[common.AssetType.wasmSequentialModule]!;
        break;
      case common.FeatureHint.webParallel:
        wasmModule = assets[common.AssetType.wasmParallelModule]!;
        break;
      default:
        throw ArgumentError('unsupported feature hint for web: $hint');
    }
  }
}
