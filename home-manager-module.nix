# home-manager module for pax-mcp, exposed by this flake as `homeModules.default`.
# Usage: programs.pax-mcp.enable = true;
#
# It only puts `pax-mcp` on PATH. An MCP client (e.g. a project's
# `.mcp.json`) starts the server itself and passes its environment, such as
# `SEMANTIC_SCHOLAR_API_KEY`. That key is a secret, so it deliberately isn't
# an option here: a home-manager option would copy it into the
# world-readable Nix store.
self:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.programs.pax-mcp;
in
{
  options.programs.pax-mcp = {
    enable = lib.mkEnableOption "pax-mcp, the MCP server exposing pax-core's library tools";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.pax-mcp;
      defaultText = lib.literalExpression "pax-ai.packages.<system>.pax-mcp";
      description = "The pax-mcp package to install.";
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ cfg.package ];
  };
}
