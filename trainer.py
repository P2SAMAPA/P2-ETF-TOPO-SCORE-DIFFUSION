"""
trainer.py  —  Orchestrator for Topo-Score-Diffusion Engine
============================================================

Loads data → trains diffusion models → generates paths → builds JSON.
Uses parallel processing for speed.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from topo_diffusion import compute_universe_topo_diffusion
from push_results import upload_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def get_action(z_score: float) -> str:
    if z_score > 1.0:
        return "STRONG BUY"
    elif z_score > 0.5:
        return "BUY"
    elif z_score > -0.5:
        return "HOLD"
    elif z_score > -1.0:
        return "REDUCE"
    else:
        return "STRONG SELL"


def process_window(args: Tuple) -> Dict:
    """Process a single window for a universe in parallel."""
    window, universe_name, available, prices_df, config_dict = args
    
    try:
        universe_prices = prices_df[available]
        result = compute_universe_topo_diffusion(universe_prices, config_dict, window)
        
        return {
            "window": window,
            "universe": universe_name,
            "results": result,
            "error": None
        }
    except Exception as e:
        return {
            "window": window,
            "universe": universe_name,
            "results": {},
            "error": str(e)
        }


def run_trainer(hf_token: Optional[str] = None) -> Dict:
    """Run the full Topo-Score-Diffusion pipeline."""
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — will skip HuggingFace upload.")

    logger.info("🔄 Loading master data from HuggingFace...")
    try:
        prices_df, macro_df = load_master_data(token)
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(f"✅ Loaded {len(prices_df)} days, {len(prices_df.columns)} ETFs")

    run_date = datetime.now().strftime("%Y-%m-%d")

    engine_config = {
        **config.DIFFUSION,
        **config.TDA,
        **config.GUIDANCE,
    }

    results_tab1 = {"run_date": run_date, "universes": {}}
    results_tab2 = {"run_date": run_date, "universes": {}}

    # ── Prepare parallel tasks ───────────────────────────────────────────────
    tasks = []
    windows = config.WINDOWS
    max_workers = max(1, int(mp.cpu_count() * 0.75))
    logger.info(f"🚀 Using {max_workers} parallel workers")

    for universe_name, tickers in config.UNIVERSES.items():
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue

        for window in windows:
            tasks.append((window, universe_name, available, prices_df, engine_config))

    # ── Run parallel processing ──────────────────────────────────────────────
    logger.info(f"📋 Total tasks: {len(tasks)}")
    all_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_window, task): task for task in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            try:
                result = future.result(timeout=1800)
                if result.get("error"):
                    logger.warning(f"   ⚠️ {result['universe']} @ {result['window']}d failed: {result['error']}")
                    continue
                key = f"{result['universe']}_{result['window']}"
                all_results[key] = result
                logger.info(f"   ✅ [{completed}/{len(tasks)}] {result['universe']} @ {result['window']}d")
            except Exception as e:
                logger.error(f"   ❌ Task failed: {e}")

    logger.info(f"✅ Completed {len(all_results)}/{len(tasks)} tasks")

    # ── Build results ──────────────────────────────────────────────────────────
    for universe_name in config.UNIVERSES.keys():
        available = [t for t in config.UNIVERSES[universe_name] if t in prices_df.columns]
        if not available:
            continue

        universe_results = {}
        for key, result in all_results.items():
            if result.get("universe") == universe_name:
                universe_results[str(result["window"])] = result

        if not universe_results:
            continue

        # Build Tab 1: Best window per ETF
        best_window_per_etf = {}
        for ticker in available:
            best_z = -999
           
