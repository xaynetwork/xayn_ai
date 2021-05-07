import 'dart:typed_data' show Uint8List;

abstract class XaynAi {
  factory XaynAi(dynamic vocab, dynamic model, [dynamic? serialized]) =>
      throw UnsupportedError('Plattform not supported');
  List<int> rerank(
    covariant List<dynamic> histories,
    covariant List<dynamic> documents,
  ) =>
      throw UnsupportedError('Plattform not supported');
  Uint8List serialize() => throw UnsupportedError('Plattform not supported');
  List<String> faults() => throw UnsupportedError('Plattform not supported');
  void analytics() => throw UnsupportedError('Plattform not supported');
  void free() => throw UnsupportedError('Plattform not supported');
}
