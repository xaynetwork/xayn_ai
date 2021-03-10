use std::{
    env,
    fs::{read_dir, read_to_string},
    path::PathBuf,
};

use cbindgen::{Builder, Config};

fn main() {
    let crate_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("Failed to read CARGO_MANIFEST_DIR env."),
    );
    let bind_config = crate_dir.join("cbindgen.toml");
    let bind_file = crate_dir.join("xayn.h");
    // let make_file = crate_dir.join("Makefile.toml");
    let ios_dir = crate_dir.join("ios").join("Classes");
    let bind_file_tpl = ios_dir.join("XaynAiPlugin.h.tpl");
    let bind_file_ios = ios_dir.join("XaynAiPlugin.h");

    // cargo doesn't check directories recursively so we have to do it by hand, also emitting a
    // rerun-if line cancels the default rerun for changes in the crate directory
    for file in read_dir("src").expect("Failed to read src dir.") {
        println!(
            "cargo:rerun-if-changed={}",
            PathBuf::from("src")
                .join(file.expect("Failed to read src file.").file_name())
                .display()
        );
    }
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed={}", bind_config.display());
    // println!("cargo:rerun-if-changed={}", make_file.display());
    println!("cargo:rerun-if-changed={}", bind_file_tpl.display());

    // generate bindings
    let config = Config::from_file(bind_config).expect("Failed to read config.");
    let builder = Builder::new().with_crate(crate_dir).with_config(config);
    builder
        .clone()
        .generate()
        .expect("Failed to generate bindings.")
        .write_to_file(bind_file);

    // generate bindings with dart ios header
    let ios_tpl = read_to_string(bind_file_tpl).expect("Failed to read ios template.");
    builder
        .with_after_include(ios_tpl)
        .generate()
        .expect("Failed to generate bindings.")
        .write_to_file(bind_file_ios);
}
