use std::{env, fs::read_dir, path::PathBuf};

use cbindgen::{Builder, Config};

fn main() {
    let crate_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("Failed to read CARGO_MANIFEST_DIR env."),
    );
    let bind_config = crate_dir.join("cbindgen.toml");
    let bind_file = crate_dir.join("ffi.h");

    // cargo doesn't check directories recursively so we have to do it by hand, also emitting a
    // rerun-if line cancels the default rerun for changes in the crate directory
    let src_dir = crate_dir.join("src");
    for file in read_dir(src_dir.as_path()).expect("Failed to read src dir.") {
        println!(
            "cargo:rerun-if-changed={}",
            src_dir
                .join(file.expect("Failed to read src file.").file_name())
                .display()
        );
    }
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed={}", bind_config.display());

    // generate bindings
    let config = Config::from_file(bind_config).expect("Failed to read config.");
    let builder = Builder::new().with_crate(crate_dir).with_config(config);
    builder
        .generate()
        .expect("Failed to generate bindings.")
        .write_to_file(bind_file);
}
