import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show Asset, AssetType, baseAssets, SetupData;

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets() {
  final wasmAssets = [common.AssetType.wasmModule, common.AssetType.wasmScript];
  return Map.fromEntries(common.baseAssets.entries
      .where((asset) => wasmAssets.contains(asset.key) == false));
}

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  late String smbertVocab;
  late String smbertModel;
  late String qambertVocab;
  late String qambertModel;
  late String ltrModel;

  SetupData(Map<common.AssetType, String> assets) {
    smbertVocab = assets[common.AssetType.smbertVocab]!;
    smbertModel = assets[common.AssetType.smbertModel]!;
    qambertVocab = assets[common.AssetType.qambertVocab]!;
    qambertModel = assets[common.AssetType.qambertModel]!;
    ltrModel = assets[common.AssetType.ltrModel]!;
  }
}
