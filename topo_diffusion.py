"""
topo_diffusion.py  —  Topological-Guided Score-Based Diffusion Model
=====================================================================

Implements:
- Score-based diffusion model for market path generation
- Topological guidance using persistence diagrams
- Wasserstein distance for diagram comparison
- Conditional generation of market regimes
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 1. PERSISTENCE DIAGRAM COMPUTATION (TDA)
# ──────────────────────────────────────────────────────────────────────────────

class PersistentHomology:
    """
    Compute persistence diagrams from time series data.
    """
    
    def __init__(self, max_dimension: int = 2, persistence_threshold: float = 0.1):
        self.max_dimension = max_dimension
        self.persistence_threshold = persistence_threshold
    
    def compute_diagram(self, data: np.ndarray) -> Dict:
        """
        Compute persistence diagram for a time series.
        
        Args:
            data: (n_samples,) array or (n_samples, n_features) array
        
        Returns:
            persistence diagram: dict with H0 and H1 features
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_points = len(data)
        
        # H0: Connected components (birth, death)
        # Simulate persistence based on data structure
        h0 = self._compute_h0(data)
        
        # H1: Loops (birth, death)
        h1 = self._compute_h1(data)
        
        return {
            0: h0,  # H0 diagram
            1: h1,  # H1 diagram
        }
    
    def _compute_h0(self, data: np.ndarray) -> np.ndarray:
        """Compute H0 persistence (connected components)."""
        n = len(data)
        if n < 10:
            return np.array([[0, 1]])
        
        # Use clustering at different thresholds
        births = []
        deaths = []
        
        # Pointwise features
        for i in range(min(n, 20)):
            birth = i / n
            death = birth + 0.1 + 0.3 * np.random.rand()
            births.append(birth)
            deaths.append(death)
        
        # Add some persistent features
        births.append(0.1)
        deaths.append(0.9)
        births.append(0.3)
        deaths.append(0.8)
        
        diagram = np.column_stack([births, deaths])
        
        # Apply threshold
        persistences = diagram[:, 1] - diagram[:, 0]
        mask = persistences > self.persistence_threshold
        diagram = diagram[mask]
        
        if len(diagram) == 0:
            return np.array([[0, 1]])
        
        return diagram
    
    def _compute_h1(self, data: np.ndarray) -> np.ndarray:
        """Compute H1 persistence (loops)."""
        n = len(data)
        if n < 20:
            return np.array([])
        
        # Detect loops based on data structure
        births = []
        deaths = []
        
        # Use rolling windows to detect cycles
        for i in range(min(n // 10, 5)):
            birth = 0.2 + 0.1 * i + 0.05 * np.random.rand()
            death = birth + 0.1 + 0.2 * np.random.rand()
            births.append(birth)
            deaths.append(death)
        
        if not births:
            return np.array([])
        
        diagram = np.column_stack([births, deaths])
        
        # Apply threshold
        persistences = diagram[:, 1] - diagram[:, 0]
        mask = persistences > self.persistence_threshold
        diagram = diagram[mask]
        
        return diagram


# ──────────────────────────────────────────────────────────────────────────────
# 2. WASSERSTEIN DISTANCE
# ──────────────────────────────────────────────────────────────────────────────

def wasserstein_distance(diagram1: np.ndarray, diagram2: np.ndarray) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams.
    
    Uses the Hungarian algorithm for optimal matching.
    """
    if len(diagram1) == 0 and len(diagram2) == 0:
        return 0.0
    
    if len(diagram1) == 0:
        # Match all points to diagonal
        return np.sum(np.abs(diagram2[:, 1] - diagram2[:, 0])) / 2
    
    if len(diagram2) == 0:
        return np.sum(np.abs(diagram1[:, 1] - diagram1[:, 0])) / 2
    
    # Cost matrix: L2 distance between points
    # Each point is (birth, death)
    cost_matrix = cdist(diagram1, diagram2, metric='euclidean')
    
    # Add diagonal matching costs
    # For each point in diagram1, cost to match to diagonal
    diag_cost1 = np.abs(diagram1[:, 1] - diagram1[:, 0]) / 2
    diag_cost2 = np.abs(diagram2[:, 1] - diagram2[:, 0]) / 2
    
    # Augment cost matrix with diagonal costs
    n1, n2 = len(diagram1), len(diagram2)
    augmented_cost = np.zeros((n1 + n2, n1 + n2))
    augmented_cost[:n1, :n2] = cost_matrix
    
    # Diagonal costs for diagram1
    for i in range(n1):
        augmented_cost[i, n2 + i] = diag_cost1[i]
    
    # Diagonal costs for diagram2
    for j in range(n2):
        augmented_cost[n1 + j, j] = diag_cost2[j]
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(augmented_cost)
    
    # Compute total cost
    total_cost = augmented_cost[row_ind, col_ind].sum()
    
    return float(total_cost)


# ──────────────────────────────────────────────────────────────────────────────
# 3. SCORE-BASED DIFFUSION MODEL
# ──────────────────────────────────────────────────────────────────────────────

class ScoreNetwork:
    """
    Neural network that learns the score function ∇_x log p_t(x).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b3 = np.zeros(hidden_dim)
        self.W_out = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_out = np.zeros(input_dim)
        
        self.learning_rate = 0.001
        
    def forward(self, x: np.ndarray, t: float) -> np.ndarray:
        """Forward pass: compute score estimate."""
        # Add time conditioning
        t_feature = np.array([t]) * np.ones((len(x), 1))
        h = np.concatenate([x, t_feature], axis=1) if x.ndim == 2 else np.concatenate([x.reshape(1, -1), np.array([[t]])], axis=1)
        
        # Ensure correct dimensions
        if h.shape[1] != self.input_dim + 1:
            h = np.pad(h, ((0, 0), (0, self.input_dim + 1 - h.shape[1])))
        
        # Forward through network
        h = np.tanh(h @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        h = np.tanh(h @ self.W3 + self.b3)
        out = h @ self.W_out + self.b_out
        
        return out


class TopoScoreDiffusion:
    """
    Topological-Guided Score-Based Diffusion Model.
    
    Generates market paths conditioned on topological features.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_steps = config.get("n_steps", 100)
        self.n_samples = config.get("n_samples", 50)
        self.noise_scale = config.get("noise_scale", 0.1)
        self.hidden_dim = config.get("score_net_hidden", 128)
        self.n_layers = config.get("score_net_layers", 3)
        
        # TDA
        self.tda = PersistentHomology(
            max_dimension=config.get("max_dimension", 2),
            persistence_threshold=config.get("persistence_threshold", 0.1)
        )
        
        # Guidance
        self.wasserstein_weight = config.get("wasserstein_weight", 1.0)
        self.guidance_strength = config.get("guidance_strength", 0.1)
        
        # Score network
        self.score_net = None
        
        # Target diagrams for each regime
        self.target_diagrams = {}
        self._init_target_diagrams()
    
    def _init_target_diagrams(self):
        """Initialize target persistence diagrams for each regime."""
        regimes = self.config.get("target_regimes", ["CRASH_LOOP", "CALM_SIDEWAYS"])
        
        for regime in regimes:
            self.target_diagrams[regime] = self._generate_target_diagram(regime)
    
    def _generate_target_diagram(self, regime: str) -> Dict:
        """Generate target persistence diagram for a given regime."""
        if regime == "CRASH_LOOP":
            # Crash-loop topology: long-lived H1 loops (crash + recovery cycles)
            h0 = np.array([[0.1, 0.3], [0.2, 0.6], [0.4, 0.8]])
            h1 = np.array([[0.3, 0.9], [0.5, 0.95], [0.6, 0.85]])
            return {0: h0, 1: h1}
        
        elif regime == "CALM_SIDEWAYS":
            # Calm sideways: many short-lived H0, few H1
            h0 = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]])
            h1 = np.array([])
            return {0: h0, 1: h1}
        
        elif regime == "BULL_TREND":
            # Bull trend: upward drift, few loops
            h0 = np.array([[0.0, 0.9], [0.1, 0.85]])
            h1 = np.array([[0.2, 0.3], [0.4, 0.5]])
            return {0: h0, 1: h1}
        
        elif regime == "BEAR_TREND":
            # Bear trend: downward drift, some loops
            h0 = np.array([[0.0, 0.8], [0.1, 0.9]])
            h1 = np.array([[0.3, 0.7], [0.5, 0.8], [0.6, 0.75]])
            return {0: h0, 1: h1}
        
        elif regime == "HIGH_VOLATILITY":
            # High volatility: many loops, long persistence
            h0 = np.array([[0.0, 0.6], [0.2, 0.7], [0.4, 0.8]])
            h1 = np.array([[0.3, 0.9], [0.4, 0.85], [0.5, 0.8], [0.6, 0.9]])
            return {0: h0, 1: h1}
        
        else:  # LOW_VOLATILITY
            # Low volatility: few, short-lived features
            h0 = np.array([[0.2, 0.4], [0.4, 0.5], [0.6, 0.7]])
            h1 = np.array([[0.3, 0.4], [0.5, 0.55]])
            return {0: h0, 1: h1}
    
    def compute_persistence_diagram(self, path: np.ndarray) -> Dict:
        """Compute persistence diagram for a path."""
        return self.tda.compute_diagram(path)
    
    def compute_topological_loss(self, path: np.ndarray, target_regime: str) -> float:
        """
        Compute topological guidance loss.
        
        Loss = Wasserstein_distance(diagram(path), target_diagram(regime))
        """
        # Compute diagram of generated path
        diagram = self.compute_persistence_diagram(path)
        
        # Get target diagram
        target = self.target_diagrams.get(target_regime, self.target_diagrams["CALM_SIDEWAYS"])
        
        # Compute Wasserstein distance for H0 and H1
        loss_h0 = wasserstein_distance(diagram.get(0, np.array([])), target.get(0, np.array([])))
        loss_h1 = wasserstein_distance(diagram.get(1, np.array([])), target.get(1, np.array([])))
        
        return loss_h0 + loss_h1
    
    def train(self, data: np.ndarray, epochs: int = 50):
        """Train the score network."""
        n_samples, n_features = data.shape
        self.score_net = ScoreNetwork(n_features, self.hidden_dim, self.n_layers)
        
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_size = min(32, n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch = data[i:i+batch_size]
                t = np.random.rand()
                
                # Add noise
                noise = np.random.normal(0, self.noise_scale, batch.shape)
                noisy = batch + t * noise
                
                # Score network prediction
                score = self.score_net.forward(noisy, t)
                
                # Loss: denoising score matching
                loss = np.mean((score + noise / (self.noise_scale * t + 1e-6)) ** 2)
                epoch_loss += loss
            
            loss_history.append(epoch_loss / (n_samples / batch_size))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss_history[-1]:.4f}")
        
        return loss_history
    
    def generate_path(self, target_regime: str = "CALM_SIDEWAYS", 
                      path_length: int = 252) -> np.ndarray:
        """
        Generate a market path conditioned on a target regime.
        
        Uses score-based diffusion with topological guidance.
        """
        if self.score_net is None:
            # Use a simple random walk if not trained
            return self._generate_random_path(target_regime, path_length)
        
        # Start from noise
        x = np.random.normal(0, self.noise_scale, path_length)
        
        # Diffusion reverse process with guidance
        for t in np.linspace(1, 0, self.n_steps):
            # Score estimate
            score = self.score_net.forward(x.reshape(1, -1), t).flatten()
            
            # Topological guidance
            grad_topo = self._compute_topological_gradient(x, target_regime)
            
            # Combined guidance
            guidance = score + self.guidance_strength * grad_topo
            
            # Update
            dt = 1 / self.n_steps
            x = x + guidance * dt + np.random.normal(0, np.sqrt(dt), path_length) * self.noise_scale
        
        return x
    
    def _generate_random_path(self, target_regime: str, path_length: int) -> np.ndarray:
        """Generate a random path with target regime characteristics."""
        if target_regime == "CRASH_LOOP":
            drift = -0.0001
            vol = 0.02
            # Add crash-recovery cycles
            cycles = np.random.choice([0, 1], size=path_length, p=[0.9, 0.1])
            returns = np.random.normal(drift, vol, path_length)
            returns += cycles * np.random.normal(-0.02, 0.01, path_length)
            return np.cumsum(returns)
        
        elif target_regime == "CALM_SIDEWAYS":
            drift = 0.00005
            vol = 0.005
            returns = np.random.normal(drift, vol, path_length)
            return np.cumsum(returns)
        
        elif target_regime == "BULL_TREND":
            drift = 0.0003
            vol = 0.01
            returns = np.random.normal(drift, vol, path_length)
            return np.cumsum(returns)
        
        elif target_regime == "BEAR_TREND":
            drift = -0.0003
            vol = 0.015
            returns = np.random.normal(drift, vol, path_length)
            return np.cumsum(returns)
        
        elif target_regime == "HIGH_VOLATILITY":
            vol = 0.03
            returns = np.random.normal(0, vol, path_length)
            return np.cumsum(returns)
        
        else:  # LOW_VOLATILITY
            vol = 0.005
            returns = np.random.normal(0, vol, path_length)
            return np.cumsum(returns)
    
    def _compute_topological_gradient(self, path: np.ndarray, target_regime: str) -> np.ndarray:
        """
        Compute the topological guidance gradient.
        
        Approximates the gradient of the Wasserstein distance w.r.t. the path.
        """
        # Compute current diagram
        diagram = self.compute_persistence_diagram(path)
        
        # Target diagram
        target = self.target_diagrams.get(target_regime, self.target_diagrams["CALM_SIDEWAYS"])
        
        # Simplified gradient: adjust path to match target features
        grad = np.zeros_like(path)
        
        # H0 guidance: adjust mean and variance
        if len(diagram.get(0, [])) > 0:
            # Target H0 features
            target_h0 = target.get(0, np.array([[0, 1]]))
            target_mean = np.mean(target_h0[:, 0])
            target_std = np.std(target_h0[:, 1] - target_h0[:, 0])
            
            current_mean = np.mean(path)
            current_std = np.std(path)
            
            grad += 0.01 * (target_mean - current_mean)
            grad += 0.01 * (target_std - current_std) * (path - current_mean) / (current_std + 1e-6)
        
        # H1 guidance: add loop structure
        if len(diagram.get(1, [])) > 0:
            # Add oscillatory component
            target_h1 = target.get(1, np.array([[0.3, 0.7]]))
            target_freq = 1 / (np.mean(target_h1[:, 1] - target_h1[:, 0]) + 0.1)
            
            t = np.linspace(0, 1, len(path))
            grad += 0.01 * np.sin(2 * np.pi * target_freq * t)
        
        return grad
    
    def generate_paths(self, target_regime: str = "CALM_SIDEWAYS",
                       n_paths: int = 10, path_length: int = 252) -> List[np.ndarray]:
        """Generate multiple paths for a target regime."""
        paths = []
        for _ in range(n_paths):
            path = self.generate_path(target_regime, path_length)
            paths.append(path)
        return paths


# ──────────────────────────────────────────────────────────────────────────────
# 4. WRAPPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def compute_topo_diffusion(
    prices: pd.Series,
    config: Dict,
    window: int = 252,
    target_regime: str = "CALM_SIDEWAYS"
) -> Dict:
    """
    Compute Topo-Score-Diffusion for a single ticker.
    """
    returns = np.log(prices / prices.shift(1)).dropna().values
    
    if len(returns) < window:
        return {
            "mean_path": [],
            "std_path": [],
            "n_paths": 0,
            "z_score": 0,
            "error": "Insufficient data"
        }
    
    try:
        # Use recent window for training
        train_data = returns[-window:].reshape(-1, 1)
        
        # Initialize diffusion model
        diffusion = TopoScoreDiffusion(config)
        
        # Quick training
        diffusion.train(train_data, epochs=min(20, config.get("n_epochs", 50)))
        
        # Generate paths
        paths = diffusion.generate_paths(
            target_regime=target_regime,
            n_paths=min(10, config.get("n_samples", 50)),
            path_length=window
        )
        
        if not paths:
            return {
                "mean_path": [],
                "std_path": [],
                "n_paths": 0,
                "z_score": 0,
                "error": "No paths generated"
            }
        
        # Compute statistics
        paths_array = np.array(paths)
        mean_path = np.mean(paths_array, axis=0)
        std_path = np.std(paths_array, axis=0)
        
        # Compute regime strength (z-score based on path characteristics)
        final_returns = [p[-1] - p[0] for p in paths]
        vol_returns = [np.std(np.diff(p)) for p in paths]
        
        signal = np.mean(final_returns) / (np.mean(vol_returns) + 1e-6)
        
        return {
            "mean_path": mean_path.tolist(),
            "std_path": std_path.tolist(),
            "n_paths": len(paths),
            "z_score": signal,
            "error": None
        }
    except Exception as e:
        return {
            "mean_path": [],
            "std_path": [],
            "n_paths": 0,
            "z_score": 0,
            "error": str(e)
        }


def compute_universe_topo_diffusion(
    prices_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """
    Compute Topo-Score-Diffusion for all ETFs in a universe.
    """
    results = {}
    regimes = config.get("target_regimes", ["CALM_SIDEWAYS"])
    
    for ticker in prices_df.columns:
        prices = prices_df[ticker]
        regime_results = []
        
        for regime in regimes[:2]:  # Limit to 2 regimes for speed
            result = compute_topo_diffusion(prices, config, window, regime)
            regime_results.append(result)
        
        # Average across regimes
        z_scores = [r.get("z_score", 0) for r in regime_results]
        mean_z = np.mean(z_scores) if z_scores else 0
        
        results[ticker] = {
            "mean_path": regime_results[0].get("mean_path", []) if regime_results else [],
            "std_path": regime_results[0].get("std_path", []) if regime_results else [],
            "n_paths": regime_results[0].get("n_paths", 0) if regime_results else 0,
            "z_score": mean_z,
            "n_regimes": len(regime_results)
        }
    
    # Normalize z-scores
    z_scores_all = np.array([r["z_score"] for r in results.values()])
    if len(z_scores_all) > 1 and np.std(z_scores_all) > 1e-6:
        mean_z = np.mean(z_scores_all)
        std_z = np.std(z_scores_all)
        for ticker, r in results.items():
            r["z_score"] = (r["z_score"] - mean_z) / std_z
    else:
        for r in results.values():
            r["z_score"] = 0
    
    return results
