import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show Asset, AssetType, baseAssets, SetupData;

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets(
    {List<String> features = const []}) {
  final wasmAssets = [
    common.AssetType.wasmSequentialModule,
    common.AssetType.wasmSequentialScript,
    common.AssetType.wasmParallelModule,
    common.AssetType.wasmParallelScript,
    common.AssetType.wasmParallelSnippet,
  ];
  return Map.fromEntries(common.baseAssets.entries
      .where((asset) => wasmAssets.contains(asset.key) == false));
}

/// Data that is required to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  late String smbertVocab;
  late String smbertModel;
  late String qambertVocab;
  late String qambertModel;
  late String ltrModel;

  SetupData(Map<common.AssetType, String> assets,
      {List<String> features = const []}) {
    smbertVocab = assets[common.AssetType.smbertVocab]!;
    smbertModel = assets[common.AssetType.smbertModel]!;
    qambertVocab = assets[common.AssetType.qambertVocab]!;
    qambertModel = assets[common.AssetType.qambertModel]!;
    ltrModel = assets[common.AssetType.ltrModel]!;
  }
}
