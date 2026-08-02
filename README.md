# P2-TOPO-SCORE-DIFFUSION

**Topological-Guided Score-Based Generative Model — Generating Market Paths with Specified Topological Features**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine trains a **score-based diffusion model** to generate synthetic, realistic market paths guided by **topological constraints**. The guidance function minimizes the Wasserstein distance between the persistence diagram of the generated path and a target persistence diagram (e.g., exhibiting a specific crash-loop topology or a calm-sideways topology).

### Theory

**Score-Based Diffusion:**
- Learns the score function ∇_x log p_t(x) from market data
- Generates paths by reversing the diffusion process

**Topological Data Analysis (TDA):**
- Computes persistence diagrams (H0 components, H1 loops)
- Captures the topological signature of market regimes

**Topological Guidance:**
- Minimizes Wasserstein distance between generated and target diagrams
- Enables conditional generation of specific market regimes

**Target Regimes:**
- **CRASH_LOOP**: Crash + recovery cycles (long-lived H1 loops)
- **CALM_SIDEWAYS**: Low volatility, sideways movement
- **BULL_TREND**: Upward drift with few loops
- **BEAR_TREND**: Downward drift with loops
- **HIGH_VOLATILITY**: Many loops, long persistence
- **LOW_VOLATILITY**: Few, short-lived features

---

## Key Metrics

| Metric | What it tells you |
|--------|-------------------|
| **z-score** | Cross-sectional ranking of topological generation quality |
| **N Paths** | Number of generated paths |
| **N Regimes** | Number of regimes used for conditioning |
| **Best Window** | Optimal window for regime generation |

---

## Windows

| Window | Purpose |
|--------|---------|
| 63d | Short-term regime generation |
| 126d | Medium-term regime generation |
| 252d | Core signal (primary) |
| 504d | Long-term regime generation |

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-TOPO-SCORE-DIFFUSION
cd P2-TOPO-SCORE-DIFFUSION
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
