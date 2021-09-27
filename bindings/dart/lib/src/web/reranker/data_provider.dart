import 'dart:typed_data' show Uint8List;

import 'package:xayn_ai_ffi_dart/src/common/reranker/data_provider.dart'
    as common show Asset, AssetType, baseAssets, Checksum, SetupData;

part 'assets.dart';

/// The optional features to be enabled by picking platform dependent assets.
enum Feature { bulkMemory, mutableGlobals, threads }

const _parallel = {Feature.bulkMemory, Feature.mutableGlobals, Feature.threads};

/// Returns the most suitable wasm assets for the given features.
Map<common.AssetType, common.Asset> getWasmAssets(Set<Feature> features) {
  if (features.containsAll(_parallel)) {
    return wasmParallel;
  } else {
    return wasmSequential;
  }
}

/// Returns a map of all assets required for initializing [`XaynAi`].
Map<common.AssetType, common.Asset> getAssets(
        {Set<Feature> features = const {}}) =>
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
