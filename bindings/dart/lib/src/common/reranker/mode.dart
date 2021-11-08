import 'package:json_annotation/json_annotation.dart' show JsonValue;

import 'package:xayn_ai_ffi_dart/src/common/ffi/genesis.dart' as ffi
    show RerankMode;

/// Rerank mode
enum RerankMode {
  @JsonValue(ffi.RerankMode.StandardNews)
  standardNews,
  @JsonValue(ffi.RerankMode.PersonalizedNews)
  personalizedNews,
  @JsonValue(ffi.RerankMode.StandardSearch)
  standardSearch,
  @JsonValue(ffi.RerankMode.PersonalizedSearch)
  personalizedSearch,
}

extension RerankModeToInt on RerankMode {
  /// Gets the discriminant.
  int toInt() {
    // We can't use `_$RerankModeEnumMap` as it only gets generated for
    // files which have a `@JsonSerializable` type containing the enum.
    // You can't make enums `@JsonSerializable`. Given that `RerankMode`
    // has only few variants and rarely changes we just write this switch
    // statement by hand.
    switch (this) {
      case RerankMode.standardNews:
        return ffi.RerankMode.StandardNews;
      case RerankMode.personalizedNews:
        return ffi.RerankMode.PersonalizedNews;
      case RerankMode.standardSearch:
        return ffi.RerankMode.StandardSearch;
      case RerankMode.personalizedSearch:
        return ffi.RerankMode.PersonalizedSearch;
    }
  }
}
