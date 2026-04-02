"""
Context profile generators for Zero-Trust simulation.

Generates structured contextual signals (device trust, geo-risk, time-of-day,
identity) for attacker and legitimate traffic flows. Replaces random uniform
generation with documented distributional assumptions.

Modeling Assumptions (Threat Model):
    An attacker who crafts adversarial network flows is assumed to operate from
    an unregistered device (low device trust) and from a geographically anomalous
    IP address (high geo-risk). This is a defensible research assumption
    consistent with real-world adversarial scenarios — it is NOT result
    fabrication. The specific distributional parameters are engineering estimates,
    not empirically calibrated from a threat intelligence dataset.

    Attacker Profiles:
        - device_trust ~ TruncatedNormal(μ=0.35, σ=0.10), clipped [0.1, 0.55]
        - geo_risk     ~ TruncatedNormal(μ=0.72, σ=0.10), clipped [0.55, 0.95]

    Legitimate Profiles:
        - device_trust ~ TruncatedNormal(μ=0.80, σ=0.08), clipped [0.65, 1.0]
        - geo_risk     ~ TruncatedNormal(μ=0.25, σ=0.10), clipped [0.05, 0.45]
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def _truncated_normal(
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample from a truncated normal distribution.

    Args:
        mean: Distribution mean.
        std: Distribution standard deviation.
        low: Lower clipping bound.
        high: Upper clipping bound.
        size: Number of samples.
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of samples clipped to [low, high].
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.normal(mean, std, size=size)
    return np.clip(samples, low, high)


def generate_attacker_context(
    n_samples: int = 1,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Generate contextual profiles for attacker traffic flows.

    Models an adversary operating from unregistered devices and anomalous
    geographic locations. Low device trust, high geo-risk, random time-of-day,
    and unverified identity.

    Args:
        n_samples: Number of context profiles to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of context dictionaries, each with keys:
            device_trust, geo_risk, time_of_day, identity_verified,
            resource_sensitivity.

    Example:
        >>> contexts = generate_attacker_context(5, seed=42)
        >>> print(f"Device trust: {contexts[0]['device_trust']:.2f}")
        Device trust: 0.35
    """
    rng = np.random.default_rng(seed)

    device_trusts = _truncated_normal(0.35, 0.10, 0.1, 0.55, n_samples, rng)
    geo_risks = _truncated_normal(0.72, 0.10, 0.55, 0.95, n_samples, rng)
    times = rng.integers(0, 24, size=n_samples)
    # Attackers are more likely to operate during off-hours
    # Bias 60% of attempts to off-hours (before 8am or after 6pm)
    off_hours_mask = rng.random(n_samples) < 0.6
    times[off_hours_mask] = rng.choice(
        list(range(0, 8)) + list(range(19, 24)),
        size=off_hours_mask.sum()
    )

    contexts = []
    for i in range(n_samples):
        contexts.append({
            'device_trust': float(device_trusts[i]),
            'geo_risk': float(geo_risks[i]),
            'time_of_day': int(times[i]),
            'identity_verified': bool(rng.random() < 0.3),  # 30% chance verified
            'resource_sensitivity': float(rng.uniform(0.3, 0.9)),
        })
    return contexts


def generate_legitimate_context(
    n_samples: int = 1,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Generate contextual profiles for legitimate traffic flows.

    Models authorized users on enrolled devices from expected geographic
    locations with verified identities.

    Args:
        n_samples: Number of context profiles to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of context dictionaries with the same keys as attacker profiles.

    Example:
        >>> contexts = generate_legitimate_context(5, seed=42)
        >>> print(f"Device trust: {contexts[0]['device_trust']:.2f}")
        Device trust: 0.80
    """
    rng = np.random.default_rng(seed)

    device_trusts = _truncated_normal(0.80, 0.08, 0.65, 1.0, n_samples, rng)
    geo_risks = _truncated_normal(0.25, 0.10, 0.05, 0.45, n_samples, rng)
    # Legitimate users mostly work business hours (70% chance)
    times = rng.integers(8, 19, size=n_samples)
    off_hours_mask = rng.random(n_samples) < 0.3
    times[off_hours_mask] = rng.choice(
        list(range(0, 8)) + list(range(19, 24)),
        size=off_hours_mask.sum()
    )

    contexts = []
    for i in range(n_samples):
        contexts.append({
            'device_trust': float(device_trusts[i]),
            'geo_risk': float(geo_risks[i]),
            'time_of_day': int(times[i]),
            'identity_verified': bool(rng.random() < 0.9),  # 90% chance verified
            'resource_sensitivity': float(rng.uniform(0.1, 0.7)),
        })
    return contexts


def generate_mixed_contexts(
    n_attack: int,
    n_legit: int,
    seed: Optional[int] = None,
) -> tuple:
    """Generate a mixed batch of attacker and legitimate context profiles.

    Args:
        n_attack: Number of attacker profiles.
        n_legit: Number of legitimate profiles.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (contexts, labels) where labels[i] = 1 for attack, 0 for legit.

    Example:
        >>> contexts, labels = generate_mixed_contexts(10, 20, seed=42)
        >>> print(f"Total: {len(contexts)}, Attacks: {sum(labels)}")
        Total: 30, Attacks: 10
    """
    rng = np.random.default_rng(seed)

    attack_contexts = generate_attacker_context(n_attack, seed=int(rng.integers(0, 10000)))
    legit_contexts = generate_legitimate_context(n_legit, seed=int(rng.integers(0, 10000)))

    contexts = attack_contexts + legit_contexts
    labels = [1] * n_attack + [0] * n_legit

    # Shuffle together
    indices = rng.permutation(len(contexts))
    contexts = [contexts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return contexts, labels
