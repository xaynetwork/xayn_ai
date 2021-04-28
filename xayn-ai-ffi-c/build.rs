use std::{
    env,
    fs::{read_dir, read_to_string, File},
    io::{BufWriter, Write},
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
    let bind_config = crate_dir.join("cbindgen.toml");
    let bind_file = crate_dir.join("ffi.h");
    let ios_dir: PathBuf = ["..", "bindings", "dart", "ios", "Classes"]
        .iter()
        .collect();
    let ios_header_tpl = ios_dir.join("XaynAiFfiDartPlugin.h.tpl");
    let ios_header = ios_dir.join("XaynAiFfiDartPlugin.h");

    cargo_rerun_if_changed(crate_dir.join("src"));
    cargo_rerun_if_changed(crate_dir.join("Cargo.toml"));
    cargo_rerun_if_changed(bind_config.as_path());
    cargo_rerun_if_changed(ios_header_tpl.as_path());

    let config = Config::from_file(bind_config).expect("Failed to read config.");
    generate_with_config(crate_dir, config)
        .expect("Failed to generate bindings.")
        .write_to_file(&bind_file);

    // generate bindings for dart ios header
    let ios_tpl = read_to_string(ios_header_tpl).expect("Failed to read ios template.");
    let bindings = read_to_string(bind_file).expect("Failed to read bindings header.");
    let mut ios_header_file =
        BufWriter::new(File::create(ios_header).expect("Failed to open ios header"));
    ios_header_file
        .write_all(ios_tpl.as_bytes())
        .expect("Failed to write template to ios header");
    ios_header_file
        .write_all(bindings.as_bytes())
        .expect("failed to write bindings to ios header");
}
