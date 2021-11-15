module.exports = {
  mode: "production",
  target: "webworker",
  output: {
    filename: "genesis.js",
    library: {
      name: "xayn_ai_ffi_wasm",
      type: "self",
    },
    clean: true,
  },
  module: {
    rules: [
      {
        test: /\.wasm/,
        type: "asset/resource",
        generator: {
          filename: "[name][ext]",
        },
      },
    ],
  },
};
