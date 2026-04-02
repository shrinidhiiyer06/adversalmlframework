# Deployment Guide

This document provides considerations for deploying the Zero-Trust Adversarial IDS in a production or staging environment.

## Requirements

### Software

- **Python**: 3.10+
- **PyTorch**: 2.0+
- **Streamlit**: 1.28+ (for dashboard)
- **Dependencies**: See `requirements.txt` for the full list of pinned versions.

### Hardware

- **CPU**: Most modern multi-core processors can handle inference and training for NSL-KDD (≈ 3 mins training).
- **RAM**: 8GB recommended.
- **Disk**: 500MB (including dataset).

## Configuration

Configuration is managed in `src/config.py`. Key environment variables you may want to set:

| Variable | Description | Default |
|---|---|---|
| `RANDOM_SEED` | Ensures reproducible results | `42` |
| `CI_ACCURACY_THRESHOLD` | Minimum accuracy for CI tests | `0.80` |
| `CONFIDENCE_THRESHOLD` | Threshold for uncertainty rejection | `0.15` |
| `STREAMLIT_PORT` | Port for the dashboard | `8501` |

## Production Best Practices

1. **Secure Your Policy Engine**: The `src/policy/zero_trust_engine.py` is the heart of the system. Ensure that its rules are regularly audited and that identity/geo-risk signals come from trusted, authenticated providers.
2. **Model Versioning**: Save and version your trained models in the `models/` directory. Use the `SCRIPTS_DIR/train_baseline.py` to regenerate models if the feature set changes.
3. **Telemetry Logs**: The system generates JSON audit trails. Ensure these are shipped to a secure log management system (e.g., ELK stack, Sentinel) for long-term retention and forensics.
4. **Adversarial Monitoring**: Monitor the "Evasion Gap" in the Blue Team tab. A widening gap between ML detection and System detection indicates that attackers are actively using adversarial techniques against your classifier.
