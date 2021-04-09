import 'dart:typed_data';

/// MemoryBox is a glorified Map that acts as an in-memory store
/// for StoreManager implementations with an interface similar
/// to Hive database Box implementation.
class MemoryBox {
  final Map<Uint8List, Uint8List> _store = {};

  Future<Uint8List?> get(Uint8List key) async => _store[key];

  Future<void> put(Uint8List key, Uint8List value) async {
    _store[key] = value;
  }

  Future<void> delete(Uint8List key) async {
    _store.remove(key);
  }
}
