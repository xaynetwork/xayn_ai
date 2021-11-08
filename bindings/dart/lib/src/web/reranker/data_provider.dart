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
  late String webWorkerScript;

  SetupData(Map<common.AssetType, dynamic> assets) {
    smbertVocab = assets[common.AssetType.smbertVocab]! as Uint8List;
    smbertModel = assets[common.AssetType.smbertModel]! as Uint8List;
    qambertVocab = assets[common.AssetType.qambertVocab]! as Uint8List;
    qambertModel = assets[common.AssetType.qambertModel]! as Uint8List;
    ltrModel = assets[common.AssetType.ltrModel]! as Uint8List;
    wasmModule = assets[common.AssetType.wasmModule]! as Uint8List;
    webWorkerScript = assets[common.AssetType.webWorkerScript]! as String;
  }
}
