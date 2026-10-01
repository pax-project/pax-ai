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

      # Default local embedding model for `adapter::ollama::OllamaAdapter`
      # (src/adapter/ollama.rs) — vendored so a downstream package (e.g.
      # lazy-pax's flake) can bundle it as part of one download instead of
      # requiring a separate `ollama pull` at first run. A caller who wants
      # a different model can still just point `OllamaAdapter` elsewhere;
      # this is only the default.
      #
      # Fetched from Ollama's registry by content digest (not the "latest"
      # tag), so this is already pinned exactly — no manual hash bump if
      # the tag moves to a newer model later. Keep the digests below in
      # sync with the model name in `OllamaAdapter` if either changes.
      nomicEmbedTextWeights = pkgs.fetchurl {
        url = "https://registry.ollama.ai/v2/library/nomic-embed-text/blobs/sha256:970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6";
        sha256 = "970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6";
      };
      nomicEmbedTextLicense = pkgs.fetchurl {
        url = "https://registry.ollama.ai/v2/library/nomic-embed-text/blobs/sha256:c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4";
        sha256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4";
      };
      # `num_ctx` matches this model's own published params blob upstream.
      nomicEmbedTextModelfile = pkgs.writeText "Modelfile" ''
        FROM ${nomicEmbedTextWeights}
        PARAMETER num_ctx 8192
      '';
    in
    {

      packages."x86_64-linux".default = naerskLib.buildPackage {
        src = ./.;
        buildInputs = app_deps;
        nativeBuildInputs = [ pkgs.pkg-config ];
      };

      # A consumer registers this with a local Ollama install via:
      #   ollama create nomic-embed-text -f <this>/Modelfile
      # which reads the weights straight out of the Nix store — no network
      # fetch to Ollama's own registry needed at that point.
      packages."x86_64-linux".nomic-embed-text-model = pkgs.linkFarm "nomic-embed-text-model" [
        {
          name = "Modelfile";
          path = nomicEmbedTextModelfile;
        }
        {
          name = "nomic-embed-text.gguf";
          path = nomicEmbedTextWeights;
        }
        {
          name = "LICENSE";
          path = nomicEmbedTextLicense;
        }
      ];

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
            # Quality-gate tooling (see justfile): `just` unifies the calling
            # interface across local dev, git hooks, and CI; `cargo-audit`
            # backs the `audit` check; `jq` parses hook-event JSON for the
            # Claude Code hooks in .claude/settings.json.
            just
            cargo-audit
            jq
          ]
          ++ app_deps;
        env.RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
        # Wire the repo's git hooks (see .githooks/) in automatically on
        # shell activation, so the quality gate runs the same way whether
        # it's triggered locally, by a hook, or by CI.
        shellHook = ''
          git config core.hooksPath .githooks
        '';
      };

      env = {
        RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        OPENSSL_DIR = "${pkgs.openssl.dev}";
      };

    };
}
