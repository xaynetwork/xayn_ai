module.exports = {
  mode: "production",
  target: "web",
  output: {
    filename: "genesis.js",
    library: {
      name: "xayn_ai_ffi_wasm",
      type: "window",
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
