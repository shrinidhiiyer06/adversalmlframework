# Troubleshooting Guide

## Common Issues and Solutions

This guide covers common issues you may encounter when using the Zero-Trust Network system and their solutions.

---

## Installation Issues

### Issue: Module Not Found Errors

**Error:**

```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**

```bash
pip install torch numpy pandas scikit-learn streamlit plotly
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

---

### Issue: Python Version Incompatibility

**Error:**

```
SyntaxError: invalid syntax
```

**Solution:**
Ensure you're using Python 3.8 or higher:

```bash
python --version
```

If version is < 3.8, upgrade Python or use a virtual environment with Python 3.8+.

---

## Dataset Issues

### Issue: Dataset Files Not Found

**Error:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/KDDTest+.txt'
```

**Solution:**

1. Download NSL-KDD dataset from [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)
2. Place files in `data/` directory:
   - `data/KDDTrain+.txt`
   - `data/KDDTest+.txt`

---

### Issue: Dataset Preprocessing Errors

**Error:**

```
ValueError: could not convert string to float
KeyError: 'protocol_type'
```

**Solution:**
Ensure dataset files are in correct format (NSL-KDD format). Check that:

- Files are comma-separated
- Last column contains labels
- No header row

---

## Model Issues

### Issue: Model File Not Found

**Error:**

```
FileNotFoundError: models/network_risk_classifier.pth
```

**Solution:**
Train the model first:

```bash
python scripts\train_baseline.py
```

This will create `models/network_risk_classifier.pth`.

---

### Issue: Model Loading Errors

**Error:**

```
RuntimeError: Error(s) in loading state_dict
```

**Solution:**

1. Delete existing model file
2. Retrain from scratch:

```bash
rm models/network_risk_classifier.pth
python scripts\train_baseline.py
```

---

### Issue: CUDA/GPU Errors

**Error:**

```
RuntimeError: CUDA out of memory
RuntimeError: No CUDA GPUs are available
```

**Solution:**
The system is designed to run on CPU. If you see CUDA errors:

1. Ensure PyTorch is installed for CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

1. Or modify code to force CPU usage:

```python
device = torch.device('cpu')
model = model.to(device)
```

---

## Dashboard Issues

### Issue: Streamlit Command Not Found

**Error:**

```
'streamlit' is not recognized as an internal or external command
```

**Solution:**
Install Streamlit:

```bash
pip install streamlit
```

---

### Issue: Dashboard Won't Load

**Error:**

```
streamlit run src\dashboard\app.py
# No output or errors
```

**Solution:**

1. Check if port 8501 is already in use
2. Try a different port:

```bash
streamlit run src\dashboard\app.py --server.port 8502
```

---

### Issue: Dashboard Shows Import Errors

**Error in browser:**

```
ImportError: cannot import name 'ZeroTrustNetworkSystem'
```

**Solution:**

1. Ensure you're running from project root directory
2. Check that all source files exist in `src/` directory
3. Verify Python path includes project root

---

### Issue: Zero-Trust Tab Shows Errors

**Error in dashboard:**

```
Failed to load Zero-Trust system
Failed to load network data
```

**Solution:**

1. Train the model first:

```bash
python scripts\train_baseline.py
```

1. Ensure dataset files exist in `data/` directory

2. Check that preprocessors exist:
   - `models/label_encoders.pkl`
   - `models/scaler.pkl`

---

## Runtime Issues

### Issue: Out of Memory

**Error:**

```
MemoryError: Unable to allocate array
```

**Solution:**

1. Reduce batch size in training:

```python
# In train_baseline.py
batch_size = 128  # Reduce from 256
```

1. Process fewer flows in dashboard:
   - Use slider to select 5-10 flows instead of 50

---

### Issue: Slow Performance

**Symptom:** Dashboard is very slow or unresponsive

**Solution:**

1. Reduce number of flows being processed
2. Close other applications to free memory
3. Use smaller epsilon values for attacks (fewer iterations)
4. Clear browser cache

---

## Testing Issues

### Issue: Test Script Fails

**Error:**

```
python scripts\test_zero_trust_system.py
# Fails with various errors
```

**Solution:**

1. Ensure model is trained
2. Check dataset files exist
3. Run from project root directory:

```bash
cd "c:\Zero trust project"
python scripts\test_zero_trust_system.py
```

---

### Issue: Adversarial Attacks Fail

**Error:**

```
RuntimeError: Function 'BackwardHookFunctionBackward' returned nan values
```

**Solution:**

1. Reduce epsilon value (try 0.01 instead of 0.05)
2. Check that model is in eval mode
3. Ensure gradients are enabled:

```python
x.requires_grad = True
```

---

## Data Issues

### Issue: Preprocessor Not Found

**Error:**

```
FileNotFoundError: label_encoders.pkl
FileNotFoundError: scaler.pkl
```

**Solution:**
These are created during first data load. Run:

```bash
python scripts\train_baseline.py
```

This will create and save preprocessors.

---

### Issue: Feature Dimension Mismatch

**Error:**

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x40 and 41x128)
```

**Solution:**
Ensure all 41 features are present. Check that:

1. Dataset has correct format
2. Preprocessing is applied correctly
3. No features are dropped accidentally

---

## Policy Issues

### Issue: All Flows Denied

**Symptom:** Dashboard shows 100% DENY rate

**Solution:**

1. Lower ML risk threshold (try 0.9 instead of 0.8)
2. Lower device trust minimum (try 0.3 instead of 0.5)
3. Check if in "Lockdown" mode (switch to "Standard")

---

### Issue: All Flows Allowed

**Symptom:** Dashboard shows 100% ALLOW rate

**Solution:**

1. Increase ML risk threshold (try 0.7 instead of 0.8)
2. Increase device trust minimum (try 0.6 instead of 0.5)
3. Check model is loaded correctly

---

## Logging Issues

### Issue: Telemetry Logs Not Saved

**Error:**

```
FileNotFoundError: logs/zero_trust_telemetry.json
```

**Solution:**

1. Create logs directory:

```bash
mkdir logs
```

1. Ensure write permissions for logs directory

---

### Issue: Logs Are Empty

**Symptom:** Telemetry file exists but contains no data

**Solution:**

1. Process some network flows first
2. Check that `export_telemetry()` is called
3. Verify logs are not cleared accidentally

---

## Advanced Troubleshooting

### Enable Debug Mode

Add debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System Requirements

Verify your system meets minimum requirements:

- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Windows/Linux/Mac OS

### Verify Installation

Run this diagnostic script:

```python
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch not installed")

try:
    import streamlit
    print(f"Streamlit version: {streamlit.__version__}")
except ImportError:
    print("Streamlit not installed")

try:
    import pandas
    print(f"Pandas version: {pandas.__version__}")
except ImportError:
    print("Pandas not installed")
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check Documentation:**
   - [Quick Start Guide](QUICK_START.md)
   - [User Guide](USER_GUIDE.md)
   - [API Reference](API_REFERENCE.md)

2. **Review Logs:**
   - Check terminal output for error messages
   - Review `logs/` directory for telemetry

3. **Verify Setup:**
   - Ensure all dependencies installed
   - Check that dataset and models exist
   - Run from correct directory

4. **Common Fixes:**
   - Restart dashboard
   - Retrain model
   - Clear browser cache
   - Reinstall dependencies

---

## FAQ

### Q: How long does training take?

**A:** 2-3 minutes on CPU for 20 epochs

### Q: Can I use GPU?

**A:** Yes, PyTorch will automatically use GPU if available. No code changes needed.

### Q: How much memory is required?

**A:** Minimum 4GB RAM, 8GB recommended

### Q: Can I use my own dataset?

**A:** Yes, but you'll need to modify `network_loader.py` to match your data format

### Q: How do I change policy thresholds?

**A:** Use the sliders in the dashboard's Zero-Trust tab, or modify `zero_trust_engine.py`

### Q: Can I add more attack types?

**A:** Yes, extend `network_adversarial.py` or `evasion_scenarios.py`

### Q: How do I export results?

**A:** Click "Export JSON" in the dashboard's telemetry section

---

*For additional support, see [User Guide](USER_GUIDE.md) or [API Reference](API_REFERENCE.md)*
