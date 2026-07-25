use std::path::Path;

fn main() {
    // openSUSE may install LibreSSL development symlinks alongside OpenSSL 3.
    // Ladybug's prebuilt archive targets OpenSSL 3, so link its exact SONAMEs.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
        && Path::new("/usr/lib64/libssl.so.3").exists()
        && Path::new("/usr/lib64/libcrypto.so.3").exists()
    {
        println!("cargo:rustc-link-search=native=/usr/lib64");
        println!("cargo:rustc-link-arg=-Wl,-l:libssl.so.3");
        println!("cargo:rustc-link-arg=-Wl,-l:libcrypto.so.3");
    }
}
