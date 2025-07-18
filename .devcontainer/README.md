# POL-AI Dev Container

This dev container provides a clean Ubuntu environment for testing the POL-AI system.

## Setup

1. Install VS Code and the "Dev Containers" extension
2. Open this project in VS Code
3. Press `Ctrl+Shift+P` and select "Dev Containers: Reopen in Container"
4. Wait for the container to build and dependencies to install

## What's Included

- Ubuntu 22.04 base
- Python 3 with pip
- Go 1.21
- All build tools and dependencies
- Auto-installation of requirements-ai.txt and requirements-dht.txt

## Port Forwarding

The following ports are automatically forwarded:
- 8551: AI Engine Auth
- 8552: AI Engine API  
- 8080: POL API
- 30303: Ethereum P2P
- 13000: Prysm Beacon
- 12000: Prysm P2P
- 4001: DHT P2P

## Running Components

Once in the container, you can run the system components as normal:
- `python3 ai/trainer.py`
- `go run engine/main.go` 