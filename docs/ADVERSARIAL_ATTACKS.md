# Adversarial Attacks

This document details the implementation and methodology of the adversarial attacks used to evaluate the Zero-Trust Adversarial IDS.

## Attack Methodologies

### Fast Gradient Method (FGM)

FGM is a white-box attack that uses the gradient of the loss function with respect to the input data to create adversarial examples. It is efficient and serves as a baseline for adversarial robustness.

- **Implementation**: `src/attacks/whitebox.py`
- **Key Parameters**: `epsilon` (ε), which controls the perturbation magnitude.

### Projected Gradient Descent (PGD)

PGD is a more powerful iterative version of FGM. It performs multiple small steps in the direction of the gradient and projects the resulting perturbation back into the ε-ball around the original input.

- **Implementation**: `src/attacks/whitebox.py`
- **Key Parameters**: `iterations`, `alpha` (step size), and `restarts`.

### HopSkipJump (Black-box)

The project also includes support for black-box attacks where the attacker only has access to the model's output (risk scores) and not its internal gradients.

- **Implementation**: `src/attacks/blackbox.py`

## Domain Constraints

Unlike image adversarial attacks, network traffic features must adhere to domain-specific constraints (e.g., non-negative packet counts, categorical value ranges).

- **Constraint Validator**: `src/attacks/validate_constraints.py`
- **Mechanism**: All adversarial perturbations are validated to ensure they result in realistic, "valid" network flows that would not be immediately rejected by protocol-level checks.

## Evasion Scenarios

We evaluate system performance across several evasion scenarios:

1. **Random Noise**: Baseline comparison using non-adversarial perturbations.
2. **Epsilon Sweeps**: Evaluating the model's performance as the attack power (ε) increases from 0.01 to 0.20.
3. **Cross-Model Transferability**: Generating attacks on a surrogate model and evaluating their success on the target deployment model.

For detailed results of these attacks, see the [Research Methodology](RESEARCH_METHODOLOGY.md).
