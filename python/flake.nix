{
  description = "Data science notebook environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.ffmpeg # needed by librosa/soundfile for audio decoding
            pkgs.pandoc
            pkgs.texliveFull
            (pkgs.python312.withPackages (python_packages: [
              # notebook
              python_packages.pipe
              # notebook
              python_packages.jupyterlab
              python_packages.ipykernel
              python_packages.ipywidgets
              # data science
              python_packages.pandas
              python_packages.numpy
              python_packages.scipy
              python_packages.scikit-learn
              # audio I/O and feature extraction
              python_packages.librosa
              python_packages.soundfile
              # visualization
              python_packages.matplotlib
              python_packages.seaborn
              python_packages.plotly
              python_packages.dash
              # clustering and dimensionality reduction
              # python_packages.hdbscan # broken
              python_packages.umap-learn
            ]))
          ];

          shellHook = ''
            echo "python: $(python --version)"
            echo "jupyter lab --notebook-dir=./notebooks"
          '';
        };
      }
    );
}
