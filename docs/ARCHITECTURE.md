# Zero-Trust Network Architecture: Adversarial Attack Detection in ML-Based ZT

## System Overview

This document describes the complete architecture of the Zero-Trust Network Adversarial Detection System.

## High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                 ZERO-TRUST NETWORK SYSTEM                   │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Network Flow    │ ← NSL-KDD Dataset (41 features)
│  (NSL-KDD)       │   - Duration, protocol, service, flags
└────────┬─────────┘   - Byte counts, packet stats
         │             - Connection patterns
         ▼
┌──────────────────────┐
│ Feature Extraction   │ ← Preprocessing:
│  & Normalization     │   - Label encoding (categorical)
└────────┬─────────────┘   - StandardScaler (numerical)
         │
         ▼
┌──────────────────────────┐
│ ML Risk Classifier       │ ← Neural Network:
│  (NetworkRiskModel)      │   Input(41) → 128 → 64 → 32 → 1
│                          │   Dropout(0.3), ReLU, Sigmoid
└────────┬─────────────────┘
         │
         ├─→ Risk Score (0-1)
         │
         ▼
┌──────────────────────────────┐
│ Context Enrichment           │ ← Logic-Driven Trust (Posture):
│  - User identity             │   - Patch compliance (0-1)
│  - Device trust (0-1)        │   - Behavioral anomaly (0-1)
│  - Geo risk (0-1)            │   - Verification age (days)
│  - Time-of-day risk          │   - Resource-level sensitivity
│  - Requested segment         │     (Micro-segmentation)
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ Zero-Trust Policy Engine             │ ← Multi-Factor & Segment Rules:
│                                      │
│  Rules:                              │
│  • Resource ML Threshold → DENY      │ (Micro-segmentation Rule 0)
│  • Global ML risk > 0.8 → DENY       │
│  • Device trust < 0.5 → MFA/DENY     │
│  • Geo risk > 0.7 → MFA              │
│  • Admin segment → strict checks     │
│  • Otherwise → ALLOW/RATE_LIMIT      │
└────────┬─────────────────────────────┘
         │
         ├─→ Decision: ALLOW/DENY/MFA/RATE_LIMIT
         │
         ▼
┌──────────────────────────┐
│ SOC Telemetry Logging    │ ← Audit Trail:
│  - Flow ID               │   - All decisions
│  - User, IP, segment     │   - Risk scores
│  - All scores & decision │   - Context data
│  - Timestamp & reason    │   - JSON export
└──────────────────────────┘

═══════════════════════════════════════════════
        ADVERSARIAL ATTACK LAYER
═══════════════════════════════════════════════

┌──────────────────────────────┐
│ Adversarial Attacker         │ ← Attack Methods:
│  - FGSM on network flows     │   - Gradient-based
│  - PGD multi-iteration       │   - Network constraints
│  - Evasion scenarios         │   - Realistic bounds
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Evasion Evaluation           │ ← Metrics:
│  - Attack Success Rate       │   - Evasion success
│  - Risk score reduction      │   - L2/Linf distance
│  - Policy bypass rate        │   - Detection rates
└──────────────────────────────┘
```

## Component Details

### 1. Network Data Layer

**Purpose:** Load and preprocess NSL-KDD network traffic data

**Key Features:**

- 41 network flow features
- Label encoding for categorical data
- StandardScaler normalization
- Train/test split handling

**Input:** Raw NSL-KDD CSV files
**Output:** Preprocessed numpy arrays (X, y)

### 2. ML Risk Classifier

**Purpose:** Detect malicious network traffic

**Architecture:**

```text
Layer 1: Linear(41 → 128) + ReLU + Dropout(0.3)
Layer 2: Linear(128 → 64) + ReLU + Dropout(0.3)
Layer 3: Linear(64 → 32) + ReLU
Layer 4: Linear(32 → 1) + Sigmoid
```

**Performance:**

- Accuracy: 78.5%
- Precision: 97.2%
- Recall: 64.1%
- F1: 77.3%

### 3. Context Enrichment

**Purpose:** Add Zero-Trust metadata to network flows

**Context Fields:**

- `user_identity`: Simulated user ID
- `device_trust_score`: 0-1 (Derives from posture logic)
- `device_posture`: Raw telemetry (Compliance, Anomaly, Age)
- `geo_risk_score`: 0-1 (geographic risk)
- `time_of_day_risk`: 0-1 (temporal risk)
- `requested_segment`: Resource level (e.g., 'db', 'web')

**Trust Sources (Simulated Logic):**

- **Patch Compliance**: weighted 40%
- **Behavioral Anomaly**: weighted 40% (Inverse: 1.0 - score)
- **Verification Age**: weighted 20% (Linear decay over 365 days)

### 4. Zero-Trust Policy Engine

**Purpose:** Make access control decisions

**Policy Rules:**

| Priority | Condition | Decision | Reason |
| :--- | :--- | :--- | :--- |
| 0 | ML Risk > Resource Threshold | DENY | Micro-segmentation breach |
| 1 | Global ML Risk > 0.8 | DENY | High risk score |
| 2 | Device Trust < 0.5 AND ML Risk > 0.4 | DENY | Untrusted device + risk |
| 3 | Device Trust < 0.5 | STEP_UP_AUTH | Require MFA |
| 4 | Geo Risk > 0.7 | STEP_UP_AUTH | High geo risk |
| 5 | Sensitive Segment AND ML Risk > 0.6 | STEP_UP_AUTH | Sensitive access |
| 6 | Sensitive Segment AND Device Trust < 0.7 | DENY | Insufficient trust |
| 7 | ML Risk > 0.6 | RATE_LIMIT | Moderate risk |
| 8 | Default | ALLOW | All checks passed |

**Resource Thresholds (example):**

- admin: 0.4
- database: 0.5
- api: 0.6
- internal: 0.7
- web: 0.8

**Access Decisions:**

- `ALLOW`: Full access
- `DENY`: Block access
- `STEP_UP_AUTH`: Require MFA
- `RATE_LIMIT`: Throttle access
- `ISOLATE`: Quarantine

### 5. Adversarial Attack Layer

**Purpose:** Test system robustness against evasion

**Attack Methods:**

#### FGSM (Fast Gradient Sign Method)

```python
perturbation = epsilon * gradient.sign()
x_adv = x - perturbation  # Minimize loss
```

#### PGD (Projected Gradient Descent)

```python
for i in range(num_iter):
    perturbation = alpha * gradient.sign()
    x_adv = x - perturbation
    x_adv = project_to_epsilon_ball(x_adv)
```

**Network Constraints:**

- **Gradient Masking**: Categorical features protected from drift
- **Feature Clipping**: Per-feature valid ranges (duration >= 0, etc.)
- **Integer Rounding**: Packet/Byte counts rounded
- **Protocol Bounds**: Statistical range enforcement

**Evasion Scenarios:**

1. Slow rate limiting
2. Port hopping
3. Mimicry attacks
4. Packet fragmentation

## Data Flow

### Normal Flow Processing

```text
1. Network packet arrives
2. Extract 41 features
3. Preprocess (encode, scale)
4. ML risk scoring → 0.7
5. Build context:
   - User: user_abc123
   - Device trust: 0.8
   - Geo risk: 0.2
   - Segment: web
6. Policy evaluation:
   - ML risk 0.7 > 0.6 → RATE_LIMIT
7. Log decision
8. Return: RATE_LIMIT
```

### Adversarial Flow Processing

```text
1. Malicious flow detected (risk: 0.9)
2. Attacker generates adversarial example
3. FGSM perturbation applied
4. Network constraints enforced
5. Adversarial flow submitted (risk: 0.5)
6. Context enrichment:
   - Device trust: 0.4 (low)
   - Geo risk: 0.8 (high)
7. Policy catches via context:
   - Device trust < 0.5 → DENY
8. Attack fails despite ML evasion
```

## Threat Model

### Attacker Capabilities

**Knowledge:**

- Black-box access to ML model
- Can query model for predictions
- Knows feature space

**Constraints:**

- Must maintain valid network flows
- Limited perturbation budget
- Cannot modify identity/device context

**Goals:**

- Evade ML detection
- Bypass Zero-Trust policies
- Gain unauthorized access

### Defense Mechanisms

### Layer 1: ML Detection

- High precision (97.2%)
- Detects known attack patterns

### Layer 2: Zero-Trust Context

- Device trust verification
- Geo-risk assessment
- Segment-based policies

### Layer 3: Policy Enforcement

- Multi-factor decision making
- Defense-in-depth
- Adaptive thresholds

## System Metrics

### Performance Metrics

- **Throughput:** ~1000 flows/second (single-threaded)
- **Latency:** <10ms per decision
- **Memory:** ~500MB (model + data)

### Security Metrics

- **False Positive Rate:** 2.8%
- **False Negative Rate:** 35.9%
- **ROC AUC (Clean):** 0.96
- **ROC AUC (Adversarial):** 0.78 (Degradation analysis)
- **Adversarial Evasion Success:** 20%
- **Policy Bypass Rate:** 0% (context catches evaded flows)

## Deployment Architecture

```text
┌─────────────────────────────────────────┐
│         Production Deployment           │
└─────────────────────────────────────────┘

Network Traffic
    ↓
┌─────────────────┐
│ Traffic Mirror  │ ← Span port / TAP
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Extract │ ← Real-time parsing
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Zero-Trust System   │ ← This system
│  (Docker Container) │
└────────┬────────────┘
         │
         ├─→ Access Decision
         │
         ▼
┌─────────────────┐
│ SIEM / SOC      │ ← Splunk, ELK
│  (Telemetry)    │
└─────────────────┘
```

## Future Enhancements

1. **Real-Time Integration**
   - Live packet capture
   - Streaming inference
   - Sub-millisecond latency

2. **Advanced ML**
   - Adversarial training
   - Ensemble models
   - Online learning

3. **Enhanced Context**
   - Behavioral analytics
   - Historical risk scoring
   - Peer group analysis

4. **Explainability**
   - SHAP values
   - Feature importance
   - Decision explanations

5. **Automation**
   - Auto-remediation
   - Dynamic policy updates
   - Threat intelligence integration

## References

- **NSL-KDD Dataset:** [UNB CIC](https://www.unb.ca/cic/datasets/nsl.html)
- **Zero-Trust:** [NIST SP 800-207](https://csrc.nist.gov/publications/detail/sp/800-207/final)
- **Adversarial ML:** [ART Framework](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
