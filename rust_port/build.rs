fn main() {
    println!("cargo:rustc-link-search=native=/home/skpawar1305/.local/lib");
    println!("cargo:rustc-link-search=native=/home/skpawar1305/robostack/.pixi/envs/humble/lib");
    println!("cargo:rustc-link-lib=static=ncnn");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=gomp");
}
