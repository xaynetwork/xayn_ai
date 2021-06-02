use std::{
    env,
    fs::read_dir,
    path::{Path, PathBuf},
};

use cbindgen::{generate_with_config, Config};

// cargo doesn't check directories recursively so we have to do it by hand, also emitting a
// rerun-if line cancels the default rerun for changes in the crate directory
fn cargo_rerun_if_changed(entry: impl AsRef<Path>) {
    let entry = entry.as_ref();
    if entry.is_dir() {
        for entry in read_dir(entry).expect("Failed to read dir.") {
            cargo_rerun_if_changed(entry.expect("Failed to read entry.").path());
        }
    } else {
        println!("cargo:rerun-if-changed={}", entry.display());
    }
}

fn main() {
    let crate_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("Failed to read CARGO_MANIFEST_DIR env."),
    );
    let dart_dir = crate_dir.parent().unwrap().join("bindings").join("dart");

    let config_file = crate_dir.join("cbindgen.toml");
    let android_file = dart_dir
        .join("android")
        .join("src")
        .join("main")
        .join("XaynAiFfiCommon.h");
    let ios_file = dart_dir
        .join("ios")
        .join("Classes")
        .join("XaynAiFfiCommon.h");

    cargo_rerun_if_changed(crate_dir.join("src"));
    cargo_rerun_if_changed(crate_dir.join("Cargo.toml"));
    cargo_rerun_if_changed(config_file.as_path());

    let config = Config::from_file(config_file).expect("Failed to read config.");
    let bindings = generate_with_config(crate_dir, config).expect("Failed to generate bindings.");
    bindings.write_to_file(android_file);
    bindings.write_to_file(ios_file);
}
