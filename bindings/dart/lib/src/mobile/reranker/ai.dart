import 'dart:ffi' show nullptr, Pointer, Uint8;
import 'dart:io' show Platform;
import 'dart:typed_data' show Uint8List;

import 'package:ffi/ffi.dart' show malloc, StringUtf8Pointer;

import 'package:xayn_ai_ffi_dart/src/common/data/document.dart' show Document;
import 'package:xayn_ai_ffi_dart/src/common/data/history.dart' show History;
import 'package:xayn_ai_ffi_dart/src/common/reranker/ai.dart' as common
    show XaynAi;
import 'package:xayn_ai_ffi_dart/src/common/reranker/analytics.dart'
    show Analytics;
import 'package:xayn_ai_ffi_dart/src/common/reranker/mode.dart'
    show RerankMode, RerankModeToInt;
import 'package:xayn_ai_ffi_dart/src/common/reranker/utils.dart'
    show selectThreadPoolSize;
import 'package:xayn_ai_ffi_dart/src/common/result/outcomes.dart'
    show RerankingOutcomes;
import 'package:xayn_ai_ffi_dart/src/common/utils.dart' show assertNeq;
import 'package:xayn_ai_ffi_dart/src/mobile/data/document.dart' show Documents;
import 'package:xayn_ai_ffi_dart/src/mobile/data/history.dart' show Histories;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/genesis.dart' show CXaynAi;
import 'package:xayn_ai_ffi_dart/src/mobile/ffi/library.dart' show ffi;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/analytics.dart'
    show AnalyticsBuilder;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/bytes.dart' show Bytes;
import 'package:xayn_ai_ffi_dart/src/mobile/reranker/data_provider.dart'
    show SetupData;
import 'package:xayn_ai_ffi_dart/src/mobile/result/error.dart' show XaynAiError;
import 'package:xayn_ai_ffi_dart/src/mobile/result/fault.dart' show Faults;
import 'package:xayn_ai_ffi_dart/src/mobile/result/outcomes.dart'
    show RerankingOutcomesBuilder;

/// The Xayn AI.
class XaynAi implements common.XaynAi {
  late Pointer<CXaynAi> _ai;

  /// Creates and initializes the Xayn AI from a given state.
  ///
  /// Requires the necessary [SetupData] and the state.
  /// It will throw an error if the provided state is empty.
  static Future<XaynAi> restore(SetupData data, Uint8List serialized) async {
    if (serialized.isEmpty) {
      throw ArgumentError('Serialized state cannot be empty');
    }

    return XaynAi._(data.smbertVocab, data.smbertModel, data.qambertVocab,
        data.qambertModel, data.ltrModel, serialized);
  }

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the necessary [SetupData] for the AI.
  static Future<XaynAi> create(SetupData data) async {
    return XaynAi._(data.smbertVocab, data.smbertModel, data.qambertVocab,
        data.qambertModel, data.ltrModel, null);
  }

  /// Creates and initializes the Xayn AI.
  ///
  /// Requires the path to the vocabulary and model of the tokenizer/embedder
  /// and the path of the LTR model. Optionally accepts the serialized reranker
  /// database, otherwise creates a new one.
  XaynAi._(
    String smbertVocab,
    String smbertModel,
    String qambertVocab,
    String qambertModel,
    String ltrModel,
    Uint8List? serialized,
  ) {
    final smbertVocabPtr = smbertVocab.toNativeUtf8().cast<Uint8>();
    final smbertModelPtr = smbertModel.toNativeUtf8().cast<Uint8>();
    final qambertVocabPtr = qambertVocab.toNativeUtf8().cast<Uint8>();
    final qambertModelPtr = qambertModel.toNativeUtf8().cast<Uint8>();
    final ltrModelPtr = ltrModel.toNativeUtf8().cast<Uint8>();
    Bytes? bytes;
    final error = XaynAiError();

    try {
      ffi.xaynai_init_thread_pool(
          selectThreadPoolSize(Platform.numberOfProcessors), error.ptr);
      if (error.isError()) {
        throw error.toException();
      }

      bytes = Bytes.fromList(serialized ?? Uint8List(0));
      _ai = ffi.xaynai_new(
        smbertVocabPtr,
        smbertModelPtr,
        qambertVocabPtr,
        qambertModelPtr,
        ltrModelPtr,
        bytes.ptr,
        error.ptr,
      );
      if (error.isError()) {
        throw error.toException();
      }
      assertNeq(_ai, nullptr);
    } finally {
      malloc.free(smbertVocabPtr);
      malloc.free(smbertModelPtr);
      malloc.free(qambertVocabPtr);
      malloc.free(qambertModelPtr);
      malloc.free(ltrModelPtr);
      bytes?.free();
      error.free();
    }
  }

  /// Reranks the documents.
  ///
  /// The list of ranks is in the same order as the documents.
  ///
  /// In case of a `Code.panic`, the ai is dropped and its pointer invalidated. The last known
  /// valid state can be restored with a previously serialized reranker database obtained from
  /// [XaynAi.serialize].
  @override
  Future<RerankingOutcomes> rerank(
    RerankMode mode,
    List<History> histories,
    List<Document> documents,
  ) async {
    final hists = Histories(histories);
    final docs = Documents(documents);
    final error = XaynAiError();

    final outcomeBuilder = RerankingOutcomesBuilder(
        ffi.xaynai_rerank(_ai, mode.toInt(), hists.ptr, docs.ptr, error.ptr));

    try {
      if (error.isError()) {
        if (error.isPanic()) {
          await free();
        }
        throw error.toException();
      }
      return outcomeBuilder.build();
    } finally {
      hists.free();
      docs.free();
      error.free();
      outcomeBuilder.free();
    }
  }

  /// Serializes the current state of the reranker.
  @override
  Future<Uint8List> serialize() async {
    final error = XaynAiError();

    final bytes = Bytes(ffi.xaynai_serialize(_ai, error.ptr));
    try {
      if (error.isError()) {
        throw error.toException();
      }
      return bytes.toList();
    } finally {
      error.free();
      bytes.free();
    }
  }

  /// Retrieves faults which might occur during reranking.
  ///
  /// Faults can range from warnings to errors which are handled in some default way internally.
  @override
  Future<List<String>> faults() async {
    final error = XaynAiError();

    final faults = Faults(ffi.xaynai_faults(_ai, error.ptr));
    try {
      if (error.isError()) {
        throw error.toException();
      }
      return faults.toList();
    } finally {
      error.free();
      faults.free();
    }
  }

  /// Retrieves the analytics which were collected in the penultimate reranking.
  @override
  Future<Analytics?> analytics() async {
    final error = XaynAiError();

    final builder = AnalyticsBuilder(ffi.xaynai_analytics(_ai, error.ptr));
    try {
      if (error.isError()) {
        throw error.toException();
      }
      return builder.build();
    } finally {
      builder.free();
      error.free();
    }
  }

  /// Serializes the synchronizable data of the reranker.
  @override
  Future<Uint8List> syncdataBytes() async {
    final error = XaynAiError();

    final bytes = Bytes(ffi.xaynai_syncdata_bytes(_ai, error.ptr));
    try {
      if (error.isError()) {
        throw error.toException();
      }
      return bytes.toList();
    } finally {
      error.free();
      bytes.free();
    }
  }

  /// Synchronizes the internal data of the reranker with another.
  @override
  Future<void> synchronize(Uint8List serialized) async {
    Bytes? bytes;
    final error = XaynAiError();

    try {
      bytes = Bytes.fromList(serialized);
      ffi.xaynai_synchronize(_ai, bytes.ptr, error.ptr);
      if (error.isError()) {
        throw error.toException();
      }
    } finally {
      bytes?.free();
      error.free();
    }
  }

  /// Frees the memory.
  @override
  Future<void> free() async {
    if (_ai != nullptr) {
      ffi.xaynai_drop(_ai);
      _ai = nullptr;
    }
  }
}
