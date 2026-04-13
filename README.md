# HybridAI-NextG

Hybrid AI system for autonomous 5G/6G network optimization. Combines a Deep Reinforcement Learning agent (SlateDQN), an LLM-based cognitive reasoner (Qwen3 via llama.cpp), and rule-based procedures to adapt network parameters (MCS, TX power) in response to KPI anomalies orchestrated through a finite-state machine.

## Data Flow

1. **ns-3** sends KPIs via TCP to the xApp adapter
2. **Telemetry** detects anomalies using MiniRocket time-series models
3. **Orchestrator** transitions to THINKING, requests OTM from the LLM
4. **Cognitive agent** generates a structured OTM (JSON schema-constrained)
5. **Network optimizer** translates the OTM for the RL engine
6. **Proposer → Predictor → Actor** generate candidate playbooks, rank via DQN, execute the winner
7. **Actuator** sends MCS/TX power commands back to ns-3
8. **Orchestrator** monitors recovery, adapts or rolls back if needed

## Modes

| Mode | Description |
|------|-------------|
| `deploy` | Full system — LLM + RL + anomaly detection + procedures |
| `train` | RL curriculum training with 4 synthetic OTMs rotating every 90s |
| `baseline` | KPI logging only (no RL actions), outputs `kpms_baseline.csv` |
| `eval` | Trained DQN + fixed OTM, no LLM, no anomaly detection |

## Usage

Deploy with the default mode (`deploy`):

```bash
./src/core/deploy.sh
```

Train the DQN agent:
(This requires the seperate ns3 simulator to run in parallell)

```bash
MODE=train ./src/core/deploy.sh
```

Run a baseline measurement (no AI actions):

```bash
MODE=baseline ./src/core/deploy.sh
```

Evaluate a trained model against a fixed OTM:

```bash
MODE=eval ./src/core/deploy.sh
```

Custom host/port and log level:

```bash
MODE=deploy PORT=7000 WEB_PORT=9090 LOG_LEVEL=DEBUG ./src/core/deploy.sh
```

The `deploy_amd.sh` was made specifically for Dino's hardware setup

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `deploy` | Run mode (see table above) |
| `HOST` | `0.0.0.0` | xApp TCP listen address |
| `PORT` | `6000` | xApp TCP port (ns-3 connects here) |
| `WEB_PORT` | `8080` | Dashboard WebSocket/HTTP port |
| `TARGET_METRIC` | `DRB_PdcpSduDelayDl` | Primary KPI to optimize |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Pre-trained Models

Place models in the `models/` directory. The deploy script auto-detects them:

- `models/qnet_online.pt` — DQN checkpoint (preferred)
- `models/qnet_offline.pt` — DQN fallback
- `models/minirocket_xapp_gnb.joblib` — gNB anomaly detector
- `models/minirocket_xapp_ue.joblib` — UE anomaly detector

### Training MiniRocket Models

Train anomaly detection models from collected KPI data (`kpms.csv`):

```bash
# Train both gNB and UE models (default)
poetry run train_minirocket_xapp

# Train on a single metric
poetry run train_minirocket_xapp --metric DRB_PdcpSduDelayDl

# Train with synthetic data (requires offline_demo module)
poetry run train_minirocket
```

## Stack

- **RL**: PyTorch (SlateDQN, Huber loss, soft target updates)
- **LLM**: llama-cpp-python (quantized GGUF)
- **Anomaly detection**: sktime MiniRocket
- **Knowledge base**: rdflib (RDF/SPARQL)
- **Async**: asyncio, websockets, aiohttp

## License

See [LICENSE](LICENSE).
