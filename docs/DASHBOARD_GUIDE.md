# Running the Zero-Trust Dashboard

## Quick Start

### 1. Launch the Dashboard

```bash
cd "c:\Zero trust project"
streamlit run src\dashboard\app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 2. Navigate to Zero-Trust Network Tab

Click on the **🔵 Zero-Trust Network** tab (the 4th tab)

### 3. Process Network Flows

**Left Panel - Network Monitor:**

- Select **Traffic Type**: Choose "Malicious Flows", "Benign Flows", or "Random Mix"
- Set **Number of Flows**: 1-50 flows to process
- Configure **Policy Thresholds**:
  - ML Risk Deny Threshold (default: 0.8)
  - Min Device Trust (default: 0.5)
- Click **🚀 Process Network Flows**

**Right Panel - Results:**

- See live processing with progress bar
- View latest access decision (ALLOW/DENY/MFA) with glowing card
- See decision breakdown metrics
- Review detailed results table

### 4. Test Adversarial Attacks

**Left Panel - Adversarial Testing:**

- Set **Attack Strength (Epsilon)**: 0.01 - 0.20
- Click **🎯 Test Adversarial Evasion**

**Right Panel - Attack Results:**

- Evasion Success Rate
- Clean vs Adversarial deny rates
- Risk score distribution box plot
- Success/warning message based on evasion rate

### 5. View Telemetry Logs

**Bottom Section:**

- Export logs as JSON
- See total decision count
- Clear logs
- View recent 20 log entries in table

## Dashboard Features

### Tab 1: 🟢 Operations (SOC)

- Live traffic monitoring
- Security mode selection
- Incident response tracking
- Threat intelligence feed

### Tab 2: 🔴 Red Team (Adversarial)

- Black-box attacks (HopSkipJump)
- White-box attacks (FGM)
- Multi-seed validation
- Robustness curves

### Tab 3: 🟣 Blue Team (Defense)

- Model resilience metrics
- Drift tolerance testing
- Adversarial fortification
- Explainability (SHAP)

### Tab 4: 🔵 Zero-Trust Network (NEW!)

- Real NSL-KDD network traffic
- ML risk scoring
- Context-aware policies
- Adversarial evasion testing
- SOC telemetry

## Example Workflow

1. **Start Dashboard**: `streamlit run src\dashboard\app.py`
2. **Go to Zero-Trust tab**
3. **Process 10 malicious flows** → See how many are DENIED
4. **Test adversarial attack** with epsilon=0.05
5. **Review evasion rate** → Should be low (~20%)
6. **Export telemetry** for SOC analysis
7. **Adjust policy thresholds** and retest

## Troubleshooting

**If dashboard fails to load:**

- Ensure all dependencies are installed: `pip install streamlit plotly torch pandas numpy scikit-learn`
- Check that models exist in `models/` directory
- Verify NSL-KDD data is in `data/` directory

**If Zero-Trust tab shows errors:**

- Run `python scripts/train_baseline.py` first to create the model
- Ensure `network_risk_classifier.pth` exists in `models/`

## Tips

- **Start with small flow counts** (5-10) for faster testing
- **Use "Malicious Flows"** to see DENY decisions
- **Try different epsilon values** (0.01, 0.05, 0.10, 0.20) to see attack strength impact
- **Export telemetry regularly** for analysis
- **Adjust policy thresholds** to see how they affect decisions

Enjoy exploring your Zero-Trust Network Security Dashboard! 🛡️
