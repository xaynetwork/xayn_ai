import 'dart:typed_data';

enum GetResponse {
  fetched,
  missing,
  error,
}

enum PutResponse {
  created,
  updated,
  error,
}

enum DeleteResponse {
  deleted,
  error,
}

/// Singleton access
class StoreManagerRef {
  static late StoreManager _instance;

  static StoreManager get instance => _instance;

  static void registerCreated(StoreManager manager) {
    _instance = manager;
  }
}

abstract class StoreManager {
  const StoreManager();

  /// fetches the value for a key
  Future<StoredObject> get(Uint8List key);

  /// stores K, V, returns true if successful, false if not
  Future<PutResponse> put({
    required Uint8List key,
    required Uint8List value,
  });

  /// deletes K, returns true if successful, false if not
  Future<DeleteResponse> delete(Uint8List key);

  /// terminates connection
  Future<void> close();
}

class StoredObject {
  final Uint8List? value;
  final GetResponse response;

  const StoredObject({
    required this.response,
    required this.value,
  });
}
