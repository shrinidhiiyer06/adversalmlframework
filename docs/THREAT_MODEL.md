# Threat Model: Adversarial Attack Detection in ML-Based Zero-Trust Network

## Overview

This document formalizes the threat model for the Zero-Trust Network Security system, defining attacker capabilities, attack vectors, defense mechanisms, and security assumptions.

---

## 1. System Context

### Protected Asset

A **Zero-Trust Network Infrastructure** that uses ML-based intrusion detection to classify network traffic and enforce context-aware access control policies.

### Security Objective

Maintain **confidentiality, integrity, and availability** of network resources by:

1. Detecting malicious network traffic with high accuracy
2. Preventing adversarial evasion attacks
3. Enforcing least-privilege access through Zero-Trust policies
4. Providing comprehensive audit trails for security operations

---

## 2. Attacker Profile

### Attacker Types

#### Type 1: External Threat Actor

- **Goal**: Gain unauthorized network access or exfiltrate data
- **Capability**: Can send network traffic to the system
- **Knowledge**: Black-box access (can query system, observe decisions)
- **Resources**: Limited query budget (<1000 queries)

#### Type 2: Advanced Persistent Threat (APT)

- **Goal**: Evade detection while maintaining persistent access
- **Capability**: Can craft sophisticated adversarial perturbations
- **Knowledge**: May have partial knowledge of ML model architecture
- **Resources**: Significant computational resources, patient approach

#### Type 3: Insider Threat

- **Goal**: Abuse legitimate credentials for malicious purposes
- **Capability**: Has valid identity but untrusted device/location
- **Knowledge**: Understands network topology and policies
- **Resources**: Legitimate access credentials

---

## 3. Attacker Capabilities

### Knowledge Level

| Level | Description | Research Context |
| :--- | :--- | :--- |
| **Black-Box** | No access to model internals | Can only query and observe decisions |
| **Gray-Box** | Partial knowledge | Knows feature space and general architecture |
| **White-Box** | Full knowledge | Has model weights (not realistic for production) |

**This project focuses on Black-Box and Gray-Box scenarios.**

### Access Level

| Access Type | Description | Constraints |
| :--- | :--- | :--- |
| **Oracle Access** | Can query the system | Limited to <1000 queries |
| **Network Access** | Can send traffic | Must maintain protocol compliance |
| **No Direct Access** | Cannot access model files | Realistic production scenario |

### Perturbation Constraints

Adversarial perturbations must maintain **network flow validity**:

1. **Integer Features**: Byte counts, packet counts must be integers
2. **Non-Negative**: Duration, sizes cannot be negative
3. **Protocol Compliance**: Flags, services must be valid
4. **Bounded Magnitude**: L∞ norm < ε (typically ε ≤ 0.20)
5. **Realistic Ranges**: All features within observed bounds

---

## 4. Attack Vectors

### Attack Vector 1: ML Evasion (FGSM)

### Fast Gradient Sign Method (FGSM)

- **Type**: White-box gradient-based attack
- **Mechanism**: Perturb features in direction of gradient
- **Constraint**: ε-bounded perturbation (L∞ ≤ ε)
- **Goal**: Reduce ML risk score below detection threshold
- **Effectiveness**: 20% evasion success rate (ε=0.05)

### Attack Process

```text
1. Get malicious flow x with label y=1 (attack)
2. Compute gradient ∇_x L(x, y=0) (minimize loss for benign)
3. Generate perturbation: δ = ε * sign(∇_x L)
4. Create adversarial example: x_adv = x - δ
5. Apply network constraints (rounding, clipping)
```

---

## 5. Defense Mechanisms

### Defense Layer 1: ML Risk Classifier

### Neural Network Intrusion Detection

- **Strength**: 97.2% precision (very few false positives)
- **Weakness**: 64.1% recall (misses 36% of attacks)
- **Robustness**: 80% of adversarial attacks still detected

### Defense Layer 2: Zero-Trust Context

### Multi-Factor Verification

- **Device Trust**: Verify device health and compliance
- **Geo-Risk**: Assess geographic risk score
- **Identity**: Validate user credentials
- **Temporal**: Consider time-of-day risk

**Effectiveness**: Catches attacks even when ML is evaded (0% policy bypass rate)

---

## 6. Risk Assessment Matrix

| Threat Vector | Impact | Likelihood | Current Mitigation | Residual Risk |
| :--- | :--- | :--- | :--- | :--- |
| **FGSM Evasion** | High | Medium | ML + Context layers | Low |
| **PGD Evasion** | High | Medium | ML + Context layers | Low |
| **Context Manipulation** | Critical | Low | External verification | Medium |
| **Insider Threat** | Critical | Low | Device trust + geo-risk | Medium |

---

## Conclusion

The Zero-Trust Network system provides **defense-in-depth** against adversarial evasion attacks through:

1. **ML Risk Scoring**: High-precision intrusion detection
2. **Context Enrichment**: Multi-factor verification
3. **Policy Enforcement**: Segment-based access control
4. **Telemetry Logging**: Comprehensive audit trails

---

*Last Updated: February 2026*
