module.exports = {
  target: "web",
  output: {
    library: "xayn_ai_ffi_wasm",
    libraryTarget: "var",
  },
  optimization: {
    minimize: true,
  },
};
