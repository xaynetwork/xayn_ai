import 'package:xayn_ai_ffi_dart/src/web/worker/request_handler.dart'
    show handleRequests;

void main() async {
  try {
    await handleRequests();
  } catch (e) {
    print(e);
  }
}
