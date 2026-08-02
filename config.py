"""
config.py  —  Configuration for Topo-Score-Diffusion Engine
============================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - DIFFUSION: Score-based diffusion parameters
  - TDA: Topological Data Analysis parameters
  - GUIDANCE: Topological guidance parameters
  - WINDOWS: Time windows for regime generation
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-topo-score-diffusion-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Windows ──────────────────────────────────────────────────────────────────

WINDOWS = [63, 126, 252, 504]
WINDOW_LABELS = {
    63: "63d  (~3 months) — Short-term",
    126: "126d (~6 months) — Medium-term",
    252: "252d (~1 year) — Core Signal",
    504: "504d (~2 years) — Long-term",
}
PRIMARY_WINDOW = 252


# ── Diffusion Model Parameters ─────────────────────────────────────────────

DIFFUSION = {
    "n_steps": 100,           # Number of diffusion steps
    "n_samples": 50,          # Number of samples to generate
    "noise_scale": 0.1,       # Initial noise scale
    "score_net_hidden": 128,   # Score network hidden dimension
    "score_net_layers": 3,    # Score network layers
    "learning_rate": 0.001,   # Learning rate
    "n_epochs": 50,           # Training epochs
}


# ── Topological Data Analysis (TDA) Parameters ─────────────────────────────

TDA = {
    "max_dimension": 2,       # 0=components, 1=loops, 2=voids
    "persistence_threshold": 0.1,  # Minimum persistence to keep
    "distance_metric": "euclidean",  # Metric for TDA
}


# ── Topological Guidance Parameters ────────────────────────────────────────

GUIDANCE = {
    "target_regimes": [       # Available target regimes
        "CRASH_LOOP",
        "CALM_SIDEWAYS",
        "BULL_TREND",
        "BEAR_TREND",
        "HIGH_VOLATILITY",
        "LOW_VOLATILITY",
    ],
    "wasserstein_weight": 1.0,  # Weight for W-distance in guidance
    "guidance_strength": 0.1,   # Guidance strength
    "n_target_diagrams": 5,    # Number of target diagrams per regime
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
