# POL-AI Hybrid Chain

A proof-of-concept blockchain combining **Proof-of-Work (PoW) proposer** + **Prysm PoS finality** + **Sybil-proof ACK-quorum** for AI model training validation.

## Architecture

- **ai/trainer.py**: Hivemind DHT + LoRA training with Cloady-235B + automatic data acquisition
- **ai/data_acquisition.py**: HuggingFace datasets integration for training data (auto-cached)
- **ai/local_averager.py**: Loss gate validation + Krum + FedAvg aggregation  
- **engine/ai_engine.go**: External engine for Geth with ACK-quorum validation
- **prysm/ack_quorum.go**: Prysm patch for ACK quorum counting and slashing

**Key Changes in Latest Version:**
- ✅ **Direct DHT connectivity**: Services connect directly to Hivemind DHT (no central daemon)
- ✅ **HuggingFace auto-caching**: Models and datasets cached automatically in `/root/.cache/huggingface`
- ✅ **Simplified deployment**: Fewer containers, shared cache volumes for efficiency

## Key Features

✅ **Single-node genesis**: Runs from genesis with just one validator  
✅ **Auto-scaling**: Committee size and quorum adapt as validators join  
✅ **HuggingFace auto-caching**: Models and datasets cached automatically  
✅ **Deterministic genesis**: Same GENESIS_SEED = same validator keys  
✅ **30s ACK timeout**: Empty AI blocks if quorum not reached  
✅ **Loss gate validation**: LoRA updates must improve model performance  
✅ **Automatic slashing**: Invalid AI blocks trigger proposer penalties  
✅ **GPU auto-detection**: Automatically uses GPU when available, falls back to CPU  
✅ **Direct DHT mesh**: No central DHT daemon, services form direct peer connections  

## Ports

Default ports (configurable in `.env` file):

| Service | Port | Protocol | Notes |
|---------|------|----------|-------|
| Geth P2P | 30303 | devp2p | Ethereum P2P network |
| Geth HTTP | 8545 | JSON-RPC | Ethereum JSON-RPC API |
| Geth AuthRPC | 8551 | JWT | Geth ↔ Prysm communication |
| Prysm gRPC | 4000 | gRPC | Prysm beacon node API |
| Prysm REST | 3500 | HTTP | Prysm REST API |
| Prysm P2P | 13000 | libp2p | Ethereum consensus P2P |
| AI Engine | 8552 | HTTP | Geth ↔ AI Engine API |

**Note**: DHT communication now happens directly between AI services using dynamic libp2p ports.

## Quick Start

### Prerequisites
- Docker & Docker Compose with NVIDIA Container Toolkit (for GPU support)
- 16GB+ RAM (for Cloady-235B model)
- NVIDIA GPU (RTX 5080 or similar, recommended for optimal performance)

### 1. First Startup

**Create `.env` file with your configuration:**
```bash
# Create .env file
cat > .env << 'EOF'
# POL-AI Hybrid Chain Configuration

# Genesis seed for deterministic validator generation (64 hex characters)
# Generate with: openssl rand -hex 32
GENESIS_SEED=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

# Network configuration
GENESIS_HOST=localhost

# AI Configuration
AI_QUORUM=auto

# Optional: Custom ports (defaults shown)
GETH_HTTP_PORT=8545
GETH_P2P_PORT=30303
GETH_AUTH_PORT=8551
PRYSM_GRPC_PORT=4000
PRYSM_REST_PORT=3500
PRYSM_P2P_PORT=13000
AI_ENGINE_PORT=8552
EOF

# Generate a unique GENESIS_SEED (recommended)
GENESIS_SEED=$(openssl rand -hex 32)
sed -i "s/GENESIS_SEED=.*/GENESIS_SEED=$GENESIS_SEED/" .env
```

**For Windows users:**
```cmd
REM Create .env file manually with these contents:
echo # POL-AI Hybrid Chain Configuration > .env
echo. >> .env
echo # Genesis seed for deterministic validator generation >> .env
echo GENESIS_SEED=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef >> .env
echo. >> .env
echo # Network configuration >> .env
echo GENESIS_HOST=localhost >> .env
echo. >> .env
echo # AI Configuration >> .env
echo AI_QUORUM=auto >> .env

REM Then replace the default GENESIS_SEED with a unique one if desired
```

**Build and start the network:**
```bash
# Build Docker images
docker-compose build

# Start all services
docker-compose up -d

# Watch logs
docker-compose logs -f
```

### 2. First Block Finalization

The network will automatically:
1. Download and cache Cloady-235B model from HuggingFace (first run)
2. Generate deterministic genesis validator 
3. Begin PoW mining with AI validation
4. **First block finalizes with 1-node quorum**

**Note**: Initial startup may take several minutes to download the Cloady-235B model. Subsequent starts use the cached model.

### 3. Common Commands

**Monitor logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f geth
docker-compose logs -f prysm-beacon
docker-compose logs -f ai-trainer
```

**Manage services:**
```bash
# Stop all services
docker-compose down

# Restart a specific service
docker-compose restart geth

# View running containers
docker-compose ps
```

### 4. GPU Setup (Recommended for RTX 5080)

The main `docker-compose.yaml` already includes GPU support. To enable GPU access:

**Install NVIDIA Container Toolkit:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify GPU Access:**
```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Should show your RTX 5080 information
```

**Start with GPU Support:**
```bash
# Normal startup (GPU support already included)
docker-compose up -d

# Verify GPU usage
docker-compose logs ai-trainer | grep -i cuda
```

### 5. Individual Docker Commands with GPU

If you prefer running individual containers:

**AI Trainer with GPU:**
```bash
docker run --gpus all \
  -e GENESIS_HOST=localhost \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v huggingface_cache:/root/.cache/huggingface \
  -v ./data_cache:/data_cache \
  pol-ai python3 ai/trainer.py --steps 1000
```

**AI Averager with GPU:**
```bash
docker run --gpus all \
  -e GENESIS_HOST=localhost \
  -e PRYSM_GRPC_PORT=4000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v huggingface_cache:/root/.cache/huggingface \
  pol-ai python3 ai/local_averager.py
```

**AI Engine:**
```bash
docker run -p 8552:8552 \
  -e GENESIS_HOST=localhost \
  pol-engine
```

### 6. Add Validators

```bash
# Additional validators join via standard deposits
# Committee size scales automatically: quorum = max(1, ceil(validators/2))
```

## Environment Variables

All configuration is stored in `.env` file (created from `env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GENESIS_SEED` | generated | Deterministic seed for genesis validator (64 hex chars) |
| `GENESIS_HOST` | localhost | Bootstrap peer for DHT discovery |
| `AI_QUORUM` | auto | Manual ACK quorum override |
| `GETH_HTTP_PORT` | 8545 | Geth JSON-RPC port |
| `GETH_AUTH_PORT` | 8551 | Geth AuthRPC port |
| `PRYSM_GRPC_PORT` | 4000 | Prysm gRPC port |
| `PRYSM_REST_PORT` | 3500 | Prysm REST API port |
| `AI_ENGINE_PORT` | 8552 | AI Engine port |
| `SLASHER_PORT` | 4003 | Prysm Slasher gRPC port |

## Architecture Details

### LoRA Training
- **r=8, α=16, dropout=0.05**
- Target modules: Q/K/V/O + MLP layers
- AdamW optimizer (lr=1e-4, β1=0.9, β2=0.95)
- LoRA δW ≤150KB FP16, published to DHT

### Loss Gate Validation
- 512 DEV_TEXTS (readable sentences ≤256 chars)
- Accept if `loss_after < 0.999 × loss_before`
- Skip gate if `loss_before < 0.05`
- Slashing for failed validation

### ACK Quorum Protocol
1. **Committee formation**: Prysm computes committee for block's slot (currently all active validators)
2. **Peer ID derivation**: `peerID = sha256(pubkey)[:8]` for each committee member  
3. **ML validation**: `local_averager.py` runs loss gate on committee members
4. **ACK publishing**: Valid updates → `ack:{updateID[:16]}:{peerID}` = `0x01` (hex format, TTL 3600s)
5. **Block format**: Binary ExtraData = `[32 bytes parentHash][32 bytes updateID]` (64 bytes total)
6. **Consensus validation**: Prysm counts ACKs only from committee members for that slot
7. **Quorum requirement**: votes ≥ max(1, ceil(committee_size/2)) to accept, otherwise slash proposer
8. **Timeout handling**: 30s timeout → empty AI block (updateID = 32 zero bytes)

### Consensus Flow
```
Training Layer:    Trainer → LoRA δW → Hivemind DHT (embedded)
Validation Layer:  Averager → Loss Gate → ACK Publication → DHT  
Consensus Layer:   Engine → PoW Block → Prysm ACK Count → Finality
```

**Key Architecture Changes:**
- **No central DHT daemon**: Each AI service runs its own Hivemind DHT node
- **Peer-to-peer mesh**: Services discover each other automatically via bootstrap peers
- **Shared HuggingFace cache**: Models and datasets cached in persistent Docker volume

## Data Acquisition

The POL-AI trainer includes automatic data acquisition from high-quality HuggingFace datasets.

### Available Datasets

**Wikipedia Datasets:**
- `wikipedia_en` - Wikipedia English (high-quality encyclopedic content)
- `wikipedia_simple` - Wikipedia Simple English (easier-to-understand content)
- `wikipedia_de` - Wikipedia German
- `wikipedia_fr` - Wikipedia French
- `wikipedia_es` - Wikipedia Spanish
- `wikipedia_it` - Wikipedia Italian
- `wikipedia_pt` - Wikipedia Portuguese
- `wikipedia_ru` - Wikipedia Russian
- `wikipedia_zh` - Wikipedia Chinese
- `wikipedia_ja` - Wikipedia Japanese
- `wikipedia_ar` - Wikipedia Arabic

**Other High-Quality Datasets:**
- `c4_en` - C4 English (Colossal Clean Crawled Corpus)
- `c4_de` - C4 German
- `c4_fr` - C4 French
- `c4_es` - C4 Spanish
- `fineweb` - High-quality web content from Common Crawl
- `fineweb_sample` - Smaller sample of FineWeb for quick testing
- `wikisource_en` - Public domain texts and documents
- `common_corpus` - Large multilingual open training dataset (2T tokens)
- `long_data` - Long-form text data for training

### Data Acquisition Usage

**List available datasets:**
```bash
python ai/trainer.py --list-datasets
```

**Auto-acquire data (uses all datasets if none specified):**
```bash
# Use all available datasets
python ai/trainer.py --auto-data --num-samples 1000

# Use specific datasets
python ai/trainer.py --auto-data --datasets wikipedia_simple fineweb_sample --num-samples 2000

# Mix multiple languages
python ai/trainer.py --auto-data --datasets wikipedia_en wikipedia_de wikipedia_fr --num-samples 1500

# Control samples per dataset
python ai/trainer.py --auto-data --datasets wikipedia_en c4_en --samples-per-dataset 500

# Save training data for later use
python ai/trainer.py --auto-data --datasets wikipedia_simple --save-data ./training_data.txt
```

**Programmatic usage:**
```python
from ai.data_acquisition import DataAcquisition
from ai.trainer import LoRATrainer

# Initialize trainer with data acquisition
trainer = LoRATrainer(data_cache_dir="./data_cache")

# Acquire training data from multiple sources
training_data = trainer.acquire_training_data(
    dataset_keys=["wikipedia_en", "c4_en"],
    total_samples=5000
)

# Start training
await trainer.training_loop(training_data, steps=100)
```

### Data Quality Features

- **Automatic caching**: Downloaded datasets are cached locally for faster access
- **Quality filtering**: Text samples filtered for minimum length and quality
- **Multilingual support**: Access datasets in multiple languages individually or combined
- **Flexible sampling**: Control total samples or samples per dataset

## Volume Mounts

| Volume | Purpose |
|--------|---------|
| `chain_data/` | Blockchain state |
| `dht_db/` | DHT persistence |
| `huggingface_cache` | Cloady-235B model + datasets (Docker volume) |
| `secrets_data/` | JWT secrets |

## CLI Examples

### Geth
```bash
geth --externalcl ./engine/ai_engine --jwtsecret jwt.hex
```

### Prysm  
```bash
prysm.sh beacon-chain --execution-endpoint http://localhost:8551 \
  --jwt-secret jwt.hex --ai-quorum auto
```

### Trainer
```bash
python3 ai/trainer.py --peer-id <hex> --steps 1000
```

## Development

### Build Individual Components
```bash
# AI components
docker build -f Dockerfile.ai -t pol-ai .

# Engine
docker build -f Dockerfile.engine -t pol-engine .

# DHT
docker build -f Dockerfile.dht -t pol-dht .

# Prysm
docker build -f Dockerfile.prysm -t pol-prysm .
```

### Manual Testing
```bash
# Test DHT connectivity
curl http://localhost:13337/dht/peers

# Check Geth sync
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  http://localhost:8545

# Prysm beacon status  
curl http://localhost:3500/eth/v1/beacon/headers/head
```

## Network Behavior

### Genesis (1 Validator)
- Committee size: 1
- ACK quorum: 1  
- Solo DHT bootstrap
- Immediate finalization

### Multi-Validator (N Validators)
- Committee size: N
- ACK quorum: ceil(N/2)
- DHT mesh network
- Byzantine fault tolerance

### AI Validation
- **Committee members**: Run loss gate in `local_averager.py`
- ✅ **Valid**: Loss improvement → ACK publication → Block acceptance  
- ❌ **Invalid**: No improvement → No ACK → Insufficient quorum → Slashing
- ⏰ **Timeout**: No ACKs within 30s → Empty AI block → Continue

## Security

### Sybil Resistance
- **Committee membership**: Only active validators (32 POL stake) selected for specific slot
- **ACK authenticity**: libp2p ED25519 signatures from DHT 
- **Vote restriction**: Prysm only counts ACKs from exact committee members (no random peers)
- **Loss gate validation**: Each committee member independently verifies ML improvements

### Slashing Conditions
- **Invalid AI update**: Failed ACK quorum → ProposerSlashing via gRPC to Prysm slasher
- **Proposer equivocation**: Standard double-signing detection
- **Committee malfeasance**: Invalid ML validation by committee members
- **Slashing integration**: Connects to Prysm slasher on port 4003, broadcasts via gossipsub

## Performance

### Throughput
- Block time: 12 seconds (Prysm standard)
- AI validation: ~5-30 seconds per update
- LoRA δW: ≤150KB per update

### Resource Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+ (model loading)
- **Storage**: 100GB+ (chain + model data)
- **GPU**: NVIDIA recommended (training)

### Performance Comparison

**With GPU (RTX 5080):**
- Model loading: ~30 seconds
- Training step: ~2-5 seconds
- Memory usage: ~8-12GB VRAM
- Precision: FP16
- Power: ~220W GPU usage

**Without GPU (CPU only):**
- Model loading: ~2-5 minutes
- Training step: ~30-60 seconds
- Memory usage: ~16-24GB RAM
- Precision: FP32
- Performance: 10-15x slower

## Troubleshooting

### Common Issues

1. **Model download fails**
   ```bash
   # Check internet connectivity and HuggingFace access
   docker-compose logs ai-trainer
   ```

3. **Genesis not starting**
   ```bash
   # Check GENESIS_SEED format (64-char hex)
   echo $GENESIS_SEED
   # Should be exactly 64 hexadecimal characters (0-9, a-f, A-F)
   ```

4. **Prysm won't sync**
   ```bash
   # Verify JWT secret matches between Geth and Prysm
   docker-compose exec geth cat /secrets/jwt.hex
   docker-compose exec prysm-beacon cat /secrets/jwt.hex
   ```

5. **GPU not detected ("No GPU or XPU found" error)**
   ```bash
   # Check NVIDIA runtime is installed
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   
   # Check GPU visibility in container
   docker-compose exec ai-trainer nvidia-smi
   docker-compose exec ai-trainer python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   
   # Restart Docker after installing NVIDIA toolkit
   sudo systemctl restart docker
   ```

6. **CUDA out of memory**
   ```bash
   # The trainer automatically handles this by:
   # - Using FP16 when GPU available
   # - Falling back to FP32 on CPU if needed
   # - Using low_cpu_mem_usage=True
   
   # Monitor GPU memory usage
   watch -n 1 nvidia-smi
   ```

### Log Analysis
```bash
# AI ACK quorum validation logs
docker-compose logs prysm-beacon | grep "ACK quorum"

# ML validation logs (loss gate)  
docker-compose logs ai-averager | grep "Loss gate"

# ACK publication logs
docker-compose logs ai-averager | grep "ACK"

# Training progress
docker-compose logs ai-trainer | grep "Step"

# GPU usage and CUDA detection
docker-compose logs ai-trainer | grep -i "cuda available"
docker-compose logs ai-trainer | grep -i "gpu"

# Monitor GPU usage in real-time
nvidia-smi dmon -s pucvmet -d 5
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Test with single-node setup
4. Submit pull request

---

**Note**: This is a proof-of-concept implementation. Production use requires additional security audits, testing, and optimizations. 