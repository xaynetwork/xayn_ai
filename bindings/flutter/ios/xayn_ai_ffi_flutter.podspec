#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint xayn_ai_ffi_flutter.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'xayn_ai_ffi_flutter'
  s.version          = '0.0.1'
  s.summary          = 'XaynAI flutter plugin project.'
  s.description      = <<-DESC
XaynAI plugin project.
                       DESC
  s.homepage         = 'http://xayn.com'
  s.license          = { :file => '../../../LICENSE' }
  s.author           = { 'Xayn' => 'engineering@xaynet.dev' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.vendored_libraries = "**/*.a"
  s.dependency 'Flutter'
  s.platform = :ios, '9.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'
  s.xcconfig = { 'OTHER_LDFLAGS' => '-force_load "${PODS_ROOT}/../.symlinks/plugins/xayn_ai_ffi_flutter/ios/libxayn_ai_ffi_c_x86_64-apple-ios.a" -force_load "${PODS_ROOT}/../.symlinks/plugins/xayn_ai_ffi_flutter/ios/libxayn_ai_ffi_c_aarch64-apple-ios.a"'}
end
