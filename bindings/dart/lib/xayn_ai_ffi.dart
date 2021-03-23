// AUTO GENERATED FILE, DO NOT EDIT.
//
// Generated by `package:ffigen`.
import 'dart:ffi' as ffi;

/// Bindings to the xayn-ai-ffi-c library.
class XaynAi {
  /// Holds the symbol lookup function.
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
      _lookup;

  /// The symbols are looked up in [dynamicLibrary].
  XaynAi(ffi.DynamicLibrary dynamicLibrary) : _lookup = dynamicLibrary.lookup;

  /// The symbols are looked up with [lookup].
  XaynAi.fromLookup(
      ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
          lookup)
      : _lookup = lookup;

  /// Creates and initializes the Xayn AI.
  ///
  /// # Errors
  /// Returns a null pointer if:
  /// - The vocab or model paths are invalid.
  ///
  /// # Safety
  /// The behavior is undefined if:
  /// - A non-null vocab or model path doesn't point to an aligned, contiguous area of memory with a
  /// terminating null byte.
  /// - A non-null error doesn't point to an aligned, contiguous area of memory with an
  /// [`ExternError`].
  ffi.Pointer<CXaynAi> xaynai_new(
    ffi.Pointer<ffi.Int8> vocab,
    ffi.Pointer<ffi.Int8> model,
    ffi.Pointer<CXaynAiError> error,
  ) {
    return _xaynai_new(
      vocab,
      model,
      error,
    );
  }

  late final _xaynai_new_ptr =
      _lookup<ffi.NativeFunction<_c_xaynai_new>>('xaynai_new');
  late final _dart_xaynai_new _xaynai_new =
      _xaynai_new_ptr.asFunction<_dart_xaynai_new>();

  /// Reranks the documents with the Xayn AI.
  ///
  /// The reranked order is written to the ranks of the documents array.
  ///
  /// # Errors
  /// Returns without changing the ranks if:
  /// - The xaynai is null.
  /// - The documents are invalid.
  ///
  /// # Safety
  /// The behavior is undefined if:
  /// - A non-null xaynai doesn't point to memory allocated by [`xaynai_new()`].
  /// - A non-null documents array doesn't point to an aligned, contiguous area of memory with
  /// at least size [`CDocument`]s.
  /// - A documents size is too large to address the memory of a non-null documents array.
  /// - A non-null id or snippet doesn't point to an aligned, contiguous area of memory with a
  /// terminating null byte.
  /// - A non-null error doesn't point to an aligned, contiguous area of memory with an
  /// [`ExternError`].
  void xaynai_rerank(
    ffi.Pointer<CXaynAi> xaynai,
    ffi.Pointer<CDocument> docs,
    int size,
    ffi.Pointer<CXaynAiError> error,
  ) {
    return _xaynai_rerank(
      xaynai,
      docs,
      size,
      error,
    );
  }

  late final _xaynai_rerank_ptr =
      _lookup<ffi.NativeFunction<_c_xaynai_rerank>>('xaynai_rerank');
  late final _dart_xaynai_rerank _xaynai_rerank =
      _xaynai_rerank_ptr.asFunction<_dart_xaynai_rerank>();

  /// Frees the memory of the Xayn AI.
  ///
  /// # Safety
  /// The behavior is undefined if:
  /// - A non-null xaynai doesn't point to memory allocated by [`xaynai_new()`].
  /// - A non-null xaynai is freed more than once.
  /// - A non-null xaynai is accessed after being freed.
  void xaynai_drop(
    ffi.Pointer<CXaynAi> xaynai,
  ) {
    return _xaynai_drop(
      xaynai,
    );
  }

  late final _xaynai_drop_ptr =
      _lookup<ffi.NativeFunction<_c_xaynai_drop>>('xaynai_drop');
  late final _dart_xaynai_drop _xaynai_drop =
      _xaynai_drop_ptr.asFunction<_dart_xaynai_drop>();

  /// Frees the memory of the error message.
  ///
  /// This *does not* free the error memory itself, which is allocated somewhere else. But this *does*
  /// free the message field memory of the error. Not freeing the error message on consecutive errors
  /// (ie. where the error code is not success) will potentially leak the error message memory of the
  /// overwritten error.
  ///
  /// # Safety
  /// The behavior is undefined if:
  /// - A non-null error doesn't point to an aligned, contiguous area of memory with an
  /// [`ExternError`].
  /// - A non-null error message doesn't point to memory allocated by [`xaynai_new()`] or
  /// [`xaynai_rerank()`].
  /// - A non-null error message is freed more than once.
  /// - A non-null error message is accessed after being freed.
  ///
  /// [`xaynai_new()`]: crate::ai::xaynai_new
  /// [`xaynai_rerank()`]: crate::ai::xaynai_rerank
  void error_message_drop(
    ffi.Pointer<CXaynAiError> error,
  ) {
    return _error_message_drop(
      error,
    );
  }

  late final _error_message_drop_ptr =
      _lookup<ffi.NativeFunction<_c_error_message_drop>>('error_message_drop');
  late final _dart_error_message_drop _error_message_drop =
      _error_message_drop_ptr.asFunction<_dart_error_message_drop>();

  /// This function does nothing.
  ///
  /// Calling this prevents Swift to drop the library.
  void dummy_function() {
    return _dummy_function();
  }

  late final _dummy_function_ptr =
      _lookup<ffi.NativeFunction<_c_dummy_function>>('dummy_function');
  late final _dart_dummy_function _dummy_function =
      _dummy_function_ptr.asFunction<_dart_dummy_function>();
}

/// The Xayn AI error codes.
abstract class CXaynAiErrorCode {
  /// An irrecoverable error.
  static const int Panic = -1;

  /// No error.
  static const int Success = 0;

  /// A vocab null pointer error.
  static const int VocabPointer = 1;

  /// A model null pointer error.
  static const int ModelPointer = 2;

  /// A vocab or model file IO error.
  static const int ReadFile = 3;

  /// A Bert builder error.
  static const int BuildBert = 4;

  /// A Reranker builder error.
  static const int BuildReranker = 5;

  /// A Xayn AI null pointer error.
  static const int XaynAiPointer = 6;

  /// A documents null pointer error.
  static const int DocumentsPointer = 7;

  /// A document id null pointer error.
  static const int IdPointer = 8;

  /// A document snippet null pointer error.
  static const int SnippetPointer = 9;
}

class CXaynAi extends ffi.Opaque {}

/// Represents an error that occured within rust, storing both an error code, and additional data
/// that may be used by the caller.
///
/// Misuse of this type can cause numerous issues, so please read the entire documentation before
/// usage.
///
/// ## Rationale
///
/// This library encourages a pattern of taking a `&mut ExternError` as the final parameter for
/// functions exposed over the FFI. This is an "out parameter" which we use to write error/success
/// information that occurred during the function's execution.
///
/// To be clear, this means instances of `ExternError` will be created on the other side of the FFI,
/// and passed (by mutable reference) into Rust.
///
/// While this pattern is not particularly ergonomic in Rust (although hopefully this library
/// helps!), it offers two main benefits over something more ergonomic (which might be `Result`
/// shaped).
///
/// 1. It avoids defining a large number of `Result`-shaped types in the FFI consumer, as would
/// be required with something like an `struct ExternResult<T> { ok: *mut T, err:... }`
///
/// 2. It offers additional type safety over `struct ExternResult { ok: *mut c_void, err:... }`,
/// which helps avoid memory safety errors. It also can offer better performance for returning
/// primitives and repr(C) structs (no boxing required).
///
/// It also is less tricky to use properly than giving consumers a `get_last_error()` function, or
/// similar.
///
/// ## Caveats
///
/// Note that the order of the fields is `code` (an i32) then `message` (a `*mut c_char`), getting
/// this wrong on the other side of the FFI will cause memory corruption and crashes.
///
/// The fields are public largely for documentation purposes, but you should use
/// [`ExternError::new_error`] or [`ExternError::success`] to create these.
///
/// ## Layout/fields
///
/// This struct's field are not `pub` (mostly so that we can soundly implement `Send`, but also so
/// that we can verify rust users are constructing them appropriately), the fields, their types, and
/// their order are *very much* a part of the public API of this type. Consumers on the other side
/// of the FFI will need to know its layout.
///
/// If this were a C struct, it would look like
///
/// ```c,no_run
/// struct ExternError {
/// int32_t code;
/// char *message; // note: nullable
/// };
/// ```
///
/// In rust, there are two fields, in this order: `code: ErrorCode`, and `message: *mut c_char`.
/// Note that ErrorCode is a `#[repr(transparent)]` wrapper around an `i32`, so the first property
/// is equivalent to an `i32`.
///
/// #### The `code` field.
///
/// This is the error code, 0 represents success, all other values represent failure. If the `code`
/// field is nonzero, there should always be a message, and if it's zero, the message will always be
/// null.
///
/// #### The `message` field.
///
/// This is a null-terminated C string containing some amount of additional information about the
/// error. If the `code` property is nonzero, there should always be an error message. Otherwise,
/// this should will be null.
///
/// This string (when not null) is allocated on the rust heap (using this crate's
/// [`rust_string_to_c`]), and must be freed on it as well. Critically, if there are multiple rust
/// packages using being used in the same application, it *must be freed on the same heap that
/// allocated it*, or you will corrupt both heaps.
///
/// Typically, this object is managed on the other side of the FFI (on the "FFI consumer"), which
/// means you must expose a function to release the resources of `message` which can be done easily
/// using the [`define_string_destructor!`] macro provided by this crate.
///
/// If, for some reason, you need to release the resources directly, you may call
/// `ExternError::release()`. Note that you probably do not need to do this, and it's
/// intentional that this is not called automatically by implementing `drop`.
///
/// ## Example
///
/// ```rust,no_run
/// use ffi_support::{ExternError, ErrorCode};
///
/// #[derive(Debug)]
/// pub enum MyError {
/// IllegalFoo(String),
/// InvalidBar(i64),
/// // ...
/// }
///
/// // Putting these in a module is obviously optional, but it allows documentation, and helps
/// // avoid accidental reuse.
/// pub mod error_codes {
/// // note: -1 and 0 are reserved by ffi_support
/// pub const ILLEGAL_FOO: i32 = 1;
/// pub const INVALID_BAR: i32 = 2;
/// // ...
/// }
///
/// fn get_code(e: &MyError) -> ErrorCode {
/// match e {
/// MyError::IllegalFoo(_) => ErrorCode::new(error_codes::ILLEGAL_FOO),
/// MyError::InvalidBar(_) => ErrorCode::new(error_codes::INVALID_BAR),
/// // ...
/// }
/// }
///
/// impl From<MyError> for ExternError {
/// fn from(e: MyError) -> ExternError {
/// ExternError::new_error(get_code(&e), format!("{:?}", e))
/// }
/// }
/// ```
class CXaynAiError extends ffi.Struct {
  @ffi.Int32()
  external int code;

  external ffi.Pointer<ffi.Int8> message;
}

/// A raw document.
class CDocument extends ffi.Struct {
  /// The raw pointer to the document id.
  external ffi.Pointer<ffi.Int8> id;

  /// The raw pointer to the document snippet.
  external ffi.Pointer<ffi.Int8> snippet;

  /// The rank of the document.
  @ffi.Uint32()
  external int rank;
}

typedef _c_xaynai_new = ffi.Pointer<CXaynAi> Function(
  ffi.Pointer<ffi.Int8> vocab,
  ffi.Pointer<ffi.Int8> model,
  ffi.Pointer<CXaynAiError> error,
);

typedef _dart_xaynai_new = ffi.Pointer<CXaynAi> Function(
  ffi.Pointer<ffi.Int8> vocab,
  ffi.Pointer<ffi.Int8> model,
  ffi.Pointer<CXaynAiError> error,
);

typedef _c_xaynai_rerank = ffi.Void Function(
  ffi.Pointer<CXaynAi> xaynai,
  ffi.Pointer<CDocument> docs,
  ffi.Uint32 size,
  ffi.Pointer<CXaynAiError> error,
);

typedef _dart_xaynai_rerank = void Function(
  ffi.Pointer<CXaynAi> xaynai,
  ffi.Pointer<CDocument> docs,
  int size,
  ffi.Pointer<CXaynAiError> error,
);

typedef _c_xaynai_drop = ffi.Void Function(
  ffi.Pointer<CXaynAi> xaynai,
);

typedef _dart_xaynai_drop = void Function(
  ffi.Pointer<CXaynAi> xaynai,
);

typedef _c_error_message_drop = ffi.Void Function(
  ffi.Pointer<CXaynAiError> error,
);

typedef _dart_error_message_drop = void Function(
  ffi.Pointer<CXaynAiError> error,
);

typedef _c_dummy_function = ffi.Void Function();

typedef _dart_dummy_function = void Function();
