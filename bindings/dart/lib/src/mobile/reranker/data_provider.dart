import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show Asset, AssetType, baseAssets, SetupData;

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets() {
  return common.baseAssets;
}

/// Data that can be used to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  final String smbertVocab;
  final String smbertModel;
  final String qambertVocab;
  final String qambertModel;

  SetupData(this.smbertVocab, this.smbertModel, this.qambertVocab, this.qambertModel);
}
