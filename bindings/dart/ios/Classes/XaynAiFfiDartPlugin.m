#import "XaynAiFfiDartPlugin.h"
#if __has_include(<xayn_ai_ffi_dart/xayn_ai_ffi_dart-Swift.h>)
#import <xayn_ai_ffi_dart/xayn_ai_ffi_dart-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "xayn_ai_ffi_dart-Swift.h"
#endif

@implementation XaynAiFfiDartPlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftXaynAiFfiDartPlugin registerWithRegistrar:registrar];
}
@end
