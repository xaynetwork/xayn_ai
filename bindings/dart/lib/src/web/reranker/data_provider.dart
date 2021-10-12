import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common
    show
        Asset,
        AssetType,
        baseAssets,
        // ignore: unused_shown_name
        Checksum,
        SetupData,
        WebFeature;

part 'assets.dart';

// ignore: unused_element
final _multithreaded = {
  common.WebFeature.bulkMemory,
  common.WebFeature.mutableGlobals,
  common.WebFeature.threads,
};

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets(
        {Set<common.WebFeature> features = const {}}) =>
    {...common.baseAssets, ...getWasmAssets(features)};

/// Data that is required to initialize [`XaynAi`].
class SetupData implements common.SetupData {
  late Uint8List smbertVocab;
  late Uint8List smbertModel;
  late Uint8List qambertVocab;
  late Uint8List qambertModel;
  late Uint8List ltrModel;
  late Uint8List wasmModule;

  SetupData(Map<common.AssetType, Uint8List> assets) {
    smbertVocab = assets[common.AssetType.smbertVocab]!;
    smbertModel = assets[common.AssetType.smbertModel]!;
    qambertVocab = assets[common.AssetType.qambertVocab]!;
    qambertModel = assets[common.AssetType.qambertModel]!;
    ltrModel = assets[common.AssetType.ltrModel]!;
    wasmModule = assets[common.AssetType.wasmModule]!;
  }
}
