import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show Asset, AssetType, baseAssets, Feature, SetupData;

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets(
    {Set<common.Feature> features = const {}}) {
  return common.baseAssets;
}

/// Data that is required to initialize [`XaynAi`].
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
