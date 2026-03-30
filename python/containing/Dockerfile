FROM nixos/nix:latest

RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf

WORKDIR /workspace

COPY flake.nix flake.lock ./
RUN nix develop --command true

COPY notebooks/ ./notebooks/

ARG PORT
ENV PORT=${PORT}
EXPOSE ${PORT}

CMD ["nix", "develop", "--command", \
     "sh", "-c", \
     "jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT}"]
