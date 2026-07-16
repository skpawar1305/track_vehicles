fn main() {
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=gomp");
    // Provide stubs for glibc symbols that were removed in glibc 2.34+
    // but are still referenced by the downloaded ONNX Runtime library
    println!("cargo:rustc-link-search=native=/tmp");
    println!("cargo:rustc-link-lib=static=crt_stubs");
    // Explicitly link libc, libgcc, and other essentials (nodefaultlibs prevents auto-linking)
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
}
