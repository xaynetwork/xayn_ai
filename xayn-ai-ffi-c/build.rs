use std::{
    env,
    fs::{read_dir, read_to_string},
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

fn generate_android(crate_dir: impl AsRef<Path>, dart_dir: impl AsRef<Path>) {
    let crate_dir = crate_dir.as_ref();
    let dart_dir = dart_dir.as_ref();

    let config_file = crate_dir.join("cbindgen_android.toml");
    let bind_file = dart_dir
        .join("android")
        .join("src")
        .join("main")
        .join("XaynAiFfiDartPlugin.h");

    cargo_rerun_if_changed(config_file.as_path());

    let config = Config::from_file(config_file).expect("Failed to read android config.");
    generate_with_config(crate_dir, config)
        .expect("Failed to generate android bindings.")
        .write_to_file(bind_file);
}

fn generate_ios(crate_dir: impl AsRef<Path>, dart_dir: impl AsRef<Path>) {
    let crate_dir = crate_dir.as_ref();
    let dart_dir = dart_dir.as_ref();
    let ios_dir = dart_dir.join("ios").join("Classes");

    let config_file = crate_dir.join("cbindgen_ios.toml");
    let tpl_file = ios_dir.join("XaynAiFfiDartPlugin.h.tpl");
    let bind_file = ios_dir.join("XaynAiFfiDartPlugin.h");

    cargo_rerun_if_changed(config_file.as_path());
    cargo_rerun_if_changed(tpl_file.as_path());

    let mut config = Config::from_file(config_file).expect("Failed to read ios config.");
    let tpl = read_to_string(tpl_file).expect("Failed to read ios template.");
    config.header = Some(tpl);
    generate_with_config(crate_dir, config)
        .expect("Failed to generate ios bindings.")
        .write_to_file(bind_file);
}

fn main() {
    let crate_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("Failed to read CARGO_MANIFEST_DIR env."),
    );
    let dart_dir = crate_dir.parent().unwrap().join("bindings").join("dart");

    cargo_rerun_if_changed(crate_dir.join("src"));
    cargo_rerun_if_changed(crate_dir.join("Cargo.toml"));

    generate_android(crate_dir.as_path(), dart_dir.as_path());
    generate_ios(crate_dir, dart_dir);
}
