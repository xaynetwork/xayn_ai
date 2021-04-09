import 'dart:async';
import 'dart:typed_data';

import 'memory_box.dart';
import 'store_manager.dart';

/// Creates a new storage manager,
/// This implementation creates a local in-memory setup,
/// team blue will have a different implementation where all storage methods
/// will instead use an Isolate to request/receive.
class ContainedStoreManager extends StoreManager {
  final MemoryBox genericDataBox = MemoryBox();

  ContainedStoreManager() {
    StoreManagerRef.registerCreated(this);
  }

  @override
  Future<DeleteResponse> delete(Uint8List key) async {
    await genericDataBox.delete(key);

    return DeleteResponse.deleted;
  }

  @override
  Future<StoredObject> get(Uint8List key) async {
    final value = await genericDataBox.get(key);

    return StoredObject(
        response: value != null ? GetResponse.fetched : GetResponse.missing,
        value: value);
  }

  @override
  Future<PutResponse> put(
      {required Uint8List key, required Uint8List value}) async {
    final existingEntry = await get(key);

    genericDataBox.put(key, value);

    return existingEntry.response == GetResponse.fetched
        ? PutResponse.updated
        : PutResponse.created;
  }

  @override
  Future<void> close() async {}
}
