{
  description = "A Basic Rust DevShell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    naersk.url = "github:nix-community/naersk";
  };

  outputs =
    {
      self,
      nixpkgs,
      naersk,
    }:
    let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
      app_deps = [ pkgs.openssl ];
      naerskLib = pkgs.callPackage naersk { };
    in
    {

      packages."x86_64-linux".default = naerskLib.buildPackage {
        src = ./.;
        buildInputs = app_deps;
        nativeBuildInputs = [ pkgs.pkg-config ];
      };

      devShells."x86_64-linux".default = pkgs.mkShell {
        nativeBuildInputs = [ pkgs.pkg-config ];
        buildInputs =
          with pkgs;
          [
            cargo
            rustc
            rustfmt
            clippy
            rust-analyzer
          ]
          ++ app_deps;
        env.RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
      };

      env = {
        RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        OPENSSL_DIR = "${pkgs.openssl.dev}";
      };

    };
}
