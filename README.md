# HybridAI-NextG

Hybrid AI system for autonomous 5G/6G network optimization. An LLM-driven Cognitive Agent (Qwen3 via llama.cpp) translates operator intents and detected anomalies into structured Intent Specifications, which a symbolic Orchestrator and Constraint Executor then turn into deterministic MCS / TX-power actions on a simulated RAN. A Reflexion-style episodic memory lets the Cognitive Agent learn from its own failed adaptations across surge events.

## Data Flow

1. **ns-3** sends KPI messages via TCP to the xApp adapter.
2. **Telemetry agents** (MiniRocket time-series classifiers) flag deviations on cell- and per-UE metrics.
3. **Orchestrator** transitions to THINKING, queries the **Knowledge Base** for candidate procedures, and requests an **Intent Specification** from the Cognitive Agent.
4. **Cognitive Agent** (LLM) emits a JSON-schema-constrained Intent Specification naming a procedure identifier and an actuator-envelope.
5. **Constraint Executor** maps the envelope onto concrete MCS / TX-power values (objective-aware, with an optional adaptive-MCS escape hatch).
6. **Actuator** sends `set-mcs` / `set-enb-txpower` E2 commands back to ns-3.
7. **Orchestrator** runs deterministic health checks each ASSURANCE window; on failure it routes back to the Cognitive Agent, which generates a verbal **reflection** that is appended to the prompt for the next adaptation attempt.
8. Adaptation is capped at four attempts before an anomaly is declared unrecoverable.

## Modes

| Mode | Description |
|------|-------------|
| `deploy` | Full system — LLM Cognitive Agent + Orchestrator FSM + Constraint Executor + MiniRocket anomaly detection + Reflexion episodic memory |
| `baseline` | KPI logging only, no actions taken — used as a reference floor for evaluation |
| `fixed-otm` | Static expert rules with deterministic constraint execution. No LLM, no anomaly detection. |

## Usage

Deploy with the default mode (`deploy`):

```bash
./src/core/deploy.sh
```

Run a baseline measurement (no AI actions, KPIs logged to CSV):

```bash
MODE=baseline ./src/core/deploy.sh
```

Run with static expert rules (no LLM):

```bash
MODE=fixed-otm ./src/core/deploy.sh
```

Custom host/port and log level:

```bash
MODE=deploy PORT=7000 WEB_PORT=9090 LOG_LEVEL=DEBUG ./src/core/deploy.sh
```

### Ablation flags (passed through to `main.py`)

| Flag | Effect |
|---|---|
| `--no-episodes` | Disable episodic memory (Reflexion ablation). Cognitive Agent still uses the Knowledge Base but receives no past reflections in its prompt. |
| `--no-adaptive-mcs` | Evaluation mode. Disables the `dl_mcs_max ge` → `mcs=-1` escape hatch, so the LLM must pick a concrete MCS integer. Prevents ns-3's in-scheduler AMC from trivially solving every episode and starving Reflexion of failures. |

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

- `models/minirocket_xapp_gnb.joblib` — gNB anomaly detector
- `models/minirocket_xapp_ue.joblib` — UE anomaly detector

LLM weights (Qwen3 GGUF) are downloaded on first run by the Cognitive Agent.

### Training MiniRocket Models

Train anomaly detection models from collected KPI data (`kpms.csv`).
Run from the project root so the in-script imports resolve:

```bash
# Train both gNB and UE models (default)
poetry run python scripts/train_minirocket_xapp.py

# Train on a single metric
poetry run python scripts/train_minirocket_xapp.py --metric DRB_PdcpSduDelayDl
```

### Running experiments to collect data

`scripts/run_experiments.py` automates the full evaluation matrix used in the thesis Results chapter: four phases (Baseline, AI without Reflexion, AI with Reflexion, Energy Intent) of five seeded ns-3 runs each, producing 20 KPI CSVs and 15 token CSVs. Note that results might vary from the Results chapter due to MiniRocket and LLM (and the Reflexion logic) is not deterministic.

```bash
poetry run python scripts/run_experiments.py
```

**What it does, per run:**

1. Boots the RIC + xApp Docker stack (one-time, at start).
2. Wipes prior KPI / token / Reflexion-memory state.
3. Launches `deploy.sh` in the appropriate mode (`--mode baseline`, `--mode deploy [--no-episodes]`, etc.).
4. Launches the ns-3 simulator (`eval_scenario-static-4UEs.cc`, 150 s, deterministic surge at t=45 s + stochastic alternating surge/blockage events with the same seed across all four phases for matched comparison).
5. Renames the resulting `kpms.csv` → `<phase><run>.csv` (e.g. `reflexion3.csv`) and the token log → `tokens_<phase><run>.csv`.
6. Tears down the relay and xApp processes between runs to guarantee a clean pipeline.

**Requirements before running:**

- The `simulator-bridge` repository must be available. By default the script looks for it as a sibling of this repo (`../simulator-bridge`); override with `SIM_ROOT=/path/to/simulator-bridge poetry run python scripts/run_experiments.py` if it lives elsewhere. (`AI_ROOT` is auto-detected from the script's own location.)
- Docker must be running with the RIC and xApp containers available.
- The script prompts for the sudo password once at the start (needed for Docker / network operations); a background thread keeps the sudo timestamp alive for the duration of the run.
- Each phase takes roughly 15-25 minutes of wall-clock time (5 runs × 150 s simulated time × ~6× real-to-sim ratio + LLM inference overhead). The full matrix takes 2-3 hours.

**Output:** 35 CSV files in the project root (`baseline1.csv` … `energy5.csv`, plus `tokens_*.csv`), ready to feed into `scripts/evaluation.py` and `scripts/plot_token_cost.py` below.

If the run is interrupted (Ctrl-C), the script kills any orphaned `main.py` process and exits cleanly. Partial CSVs from an incomplete phase can be replayed by editing the `runs=[...]` argument of the relevant `run_phase(...)` call to retry only the failed seeds.

### Evaluation and plotting scripts

Plotting utilities live in `scripts/`. They take CSV paths as CLI arguments and write PNGs to `--out-dir`:

```bash
# Aggregate evaluation plots + LaTeX summary table
poetry run python scripts/evaluation.py \
    --baseline baseline1.csv baseline2.csv ... \
    --managed no_reflexion1.csv no_reflexion2.csv ... \
    --reflexion reflexion1.csv reflexion2.csv ... \
    --energy energy1.csv energy2.csv ... \
    --out-dir final_results

# Token cost / inference time plot
poetry run python scripts/plot_token_cost.py \
    --managed tokens_no_reflexion1.csv ... \
    --reflexion tokens_reflexion1.csv ... \
    --energy tokens_energy1.csv ...

# Single-run trace plot (per-UE latency / throughput / MCS distribution)
poetry run python scripts/evaluate_baseline.py --csv energy1.csv --out-dir final_results
```

## Stack

- **LLM**: llama-cpp-python (quantized GGUF, Qwen3-9B 8-bit)
- **Anomaly detection**: sktime MiniRocket time-series classifier
- **Knowledge base**: rdflib (RDF/SPARQL) with a small catalog of canonical recovery procedures
- **Episodic memory**: file-backed JSONL store, exact-match retrieval over `(metric, cell_id)` task tuples
- **Async**: asyncio, websockets, aiohttp

## License

See [LICENSE](LICENSE).
