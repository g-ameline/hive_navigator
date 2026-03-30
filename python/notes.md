# for local notebook
`jupyter lab --notebook-dir=./notebooks --no-browser --port=8888`

# for remote ssh notebook
## from the local machine ssh session
`ssh user@remote_host`
`nix develop`
`jupyter lab --notebook-dir=./notebooks --no-browser --port=8888`
## from the local machine's other terminal
`ssh -N -L 8888:localhost:8888 user@ip`

