import Flutter
import UIKit

public class SwiftXaynAiFfiDartPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "xayn_ai_ffi_dart", binaryMessenger: registrar.messenger())
    let instance = SwiftXaynAiFfiDartPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    result("iOS " + UIDevice.current.systemVersion)
  }

  // The Xcode toolchain won't include the shared library in the build
  // process unless a method from the library is invoked. So, this
  // call to a component directly related to the ai is just done to ensure
  // that the library is included, independent of the actual return value
  // or failure.
  public func enforceBinding(){
    xaynai_new(nil, nil, nil, nil, nil, nil)
  }
}
