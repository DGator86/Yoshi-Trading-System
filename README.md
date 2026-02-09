# 🦖 Yoshi Trading System (Monorepo)

**A unified, physics-inspired prediction market trading system for Kalshi.**

## 🏗️ Full System Architecture

The system is split into two main domains: **Yoshi-Bot** (Core Intelligence) and **ClawdBot** (Interface & Orchestration).

```mermaid
graph TD
    %% External Interfaces
    User((Telegram User))
    Kalshi[Kalshi API]
    Data[(Crypto Data Sources)]
    
    subgraph "VPS / Runtime Environment"
        
        subgraph "ClawdBot (Control Plane)"
            TGBot[Telegram Bot Service]
            Orch[Orchestrator]
            Bridge[Signal Bridge]
        end
        
        subgraph "Yoshi-Bot (Intelligence Plane)"
            Scanner[Kalshi Scanner]
            
            subgraph "Gnosis Engine"
                Physics[Particle Physics Model]
                Quantum[Quantum Price Engine]
                Ralph[Ralph Meta-Learner]
            end
            
            CoreAPI[Trading Core API :8000]
        end
        
        %% Internal Flows
        TGBot <--> User
        TGBot --> Orch
        Orch --> Scanner : "Trigger Scan"
        
        Scanner <-- Data : "OHLCV / Orderbook"
        Scanner --> Physics : "Market State"
        Physics --> Quantum : "Kinematics & Forces"
        Quantum --> Scanner : "Probabilistic Cloud"
        
        Scanner -- "Value Plays" --> Bridge
        Bridge -- "POST /propose" --> CoreAPI
        
        Ralph -.-> |"Walk-Forward Tuning"| Quantum
        
        CoreAPI <--> Kalshi : "Execution"
    end
```

---

## 🧩 Component Breakdown

### 1. Yoshi-Bot (The Brain)

Located in `/yoshi-bot`, this contains the deep quantitative logic.

* **Particle Physics Engine (`gnosis.particle.physics`)**:
  * Treats price as a physical object with **Mass** (inverse volatility), **Velocity**, and **Momentum**.
  * Calculates forces like **Liquidation Gravity**, **Trend Momentum**, and **Mean Reversion Springs**.
* **Quantum Price Engine (`gnosis.particle.quantum`)**:
  * Runs Monte Carlo simulations (default 10,000 paths) to project future price clouds.
  * Uses **Regime Switching** (Trending vs. Ranging) to adjust physics constants dynamically.
* **Ralph Learner (`gnosis.harness`)**:
  * A meta-learning system that uses **Walk-Forward Validation**.
  * Optimizes hyperparameters (lookback windows, force multipliers) without looking at test data (prevents overfitting).
* **Trading Core API**:
  * A local FastAPI service that manages the portfolio, positions, and risk checks.

### 2. ClawdBot (The Interface)

Located in `/clawdbot`, this handles user interaction and deployment.

* **Telegram Bot**:
  * Long-polling automated assistant (`@Crypto_Gnosis_Bot`).
  * Commands: `/scan` (force prediction), `/status` (system health), `/ralph` (learning report).
* **Orchestrator**:
  * Manages the background loop, ensuring scans happen at correct intervals.
  * Routes alerts to Telegram.
* **Bridge**:
  * Parses the `scanner.log` and `value_plays` from Yoshi-Bot.
  * Formats them into structured trade proposals for the Trading Core.

---

## 🌊 Data Flow: From Chaos to Execution

1. **Ingestion**: `KalshiScanner` fetches real-time crypto data (Price, Volume, Funding Rates).
2. **Physics Modeling**: Raw data is converted into "Kinematic State" (e.g., "Price is accelerating downwards with high mass").
3. **Simulation**: `QuantumEngine` simulates 10,000 possible futures based on current forces.
4. **Signal Generation**: If the probability of a specific outcome (e.g., BTC > $90k) exceeds the implied probability on Kalshi (Edge), a **Value Play** is generated.
5. **Filtering**: **Ralph** validates the signal against historical performance regimes.
6. **Orchestration**: `ClawdBot` picks up the signal, notifies the User via Telegram, and sends it to the `Trading Core`.
7. **Execution**: `Trading Core` places the order on Kalshi if risk checks pass.

---

## 🛠️ Deployment & Status

**Current Status:**

* ✅ **Telegram Integration**: Fully Functional
* ✅ **VPS Deployment**: Active (165.245.140.115)
* ✅ **Physics Engine**: v2.1 (Implemented)
* ⚠️ **Live Data Feeds**: Refactoring for reliability (Known Gap)
* ❌ **Advanced Features**: Gamma Fields, Cross-Exchange Funding (See `FEATURE_GAPS.md`)

## 📂 Key Files & Scripts

### In `yoshi-bot/scripts/` (The Brains)

| Script | Description |
| :--- | :--- |
| **`start_trading_core.py`** | **The Bank.** Starts the local API server that manages your portfolio, positions, and risk. |
| **`kalshi_scanner.py`** | **The Eyes.** Fetches market data, runs the physics engine, and spots "Value Plays". |
| **`run_experiment.py`** | **The Lab.** Runs backtests to validate if the physics model is actually working. |
| **`run_improvement_loop.py`** | **The Coach.** Ralph uses this to tune hyperparameters automatically over time. |

### In `clawdbot/scripts/` (The Hands)

| Script | Description |
| :--- | :--- |
| **`telegram-bot.py`** | **The Voice.** Connects to Telegram to send alerts and receive your commands. |
| **`yoshi-bridge.py`** | **The Nervous System.** Reads the scanner's logs and pushes trade signals to the Trading Core. |
| **`deploy-ultimate-fix.sh`** | **The Mechanic.** A comprehensive script to fix deployment issues on the VPS. |

**Directory Structure:**

```
/
├── yoshi-bot/          # Core Logic
│   ├── src/gnosis/     # Physics & ML Modules
│   └── scripts/        # Scanners & API start scripts
├── clawdbot/           # Interface
│   ├── gnosis/telegram/# Bot implementation
│   └── scripts/        # Deployment helpers
└── .env                # Unified Secrets (Synced)
```

## 🚀 Quick Start

1. **Start Trading Core** (The Bank):

    ```bash
    cd yoshi-bot && python scripts/start_trading_core.py
    ```

2. **Start Telegram Bot** (The Controller):

    ```bash
    cd clawdbot && python scripts/telegram-bot.py
    ```

3. **Run a Manual Scan** (The Action):
    * Open Telegram.
    * Send `/scan` to `@Crypto_Gnosis_Bot`.
