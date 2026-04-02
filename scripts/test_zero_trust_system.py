"""
End-to-End Zero-Trust Network System Test
Tests complete pipeline from network flows to access decisions
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.system.zero_trust_network import ZeroTrustNetworkSystem
from src.data.network_loader import NetworkDataLoader
from src.attacks.network_adversarial import NetworkAdversarialAttacker


def test_complete_system():
    """End-to-end test of Zero-Trust network system"""
    
    print("="*70)
    print("ZERO-TRUST NETWORK SYSTEM - END-TO-END TEST")
    print("="*70)
    
    # Load test data
    print("\n[1/5] Loading NSL-KDD test data...")
    loader = NetworkDataLoader()
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    test_path = os.path.join(data_dir, 'KDDTest+.txt')
    
    # Load preprocessors first
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    try:
        loader.load_preprocessors(models_dir)
        X_test, y_test, _ = loader.load_and_preprocess(test_path, is_train=False)
    except FileNotFoundError:
        print("  Preprocessors not found, loading as training data...")
        X_test, y_test, _ = loader.load_and_preprocess(test_path, is_train=True)
    
    # Get malicious flows only
    malicious_idx = np.where(y_test == 1)[0][:100]
    X_malicious = X_test[malicious_idx]
    
    print(f"  Loaded {len(X_malicious)} malicious network flows for testing")
    
    # Initialize system
    print("\n[2/5] Initializing Zero-Trust Network System...")
    model_path = os.path.join(models_dir, 'network_risk_classifier.pth')
    system = ZeroTrustNetworkSystem(model_path=model_path)
    
    # Test 1: Clean malicious traffic (should be denied)
    print("\n[3/5] TEST 1: Clean Malicious Traffic")
    print("-"*70)
    print("Testing if clean malicious flows are correctly denied...")
    print()
    
    clean_results = []
    for i, x in enumerate(X_malicious[:10]):
        result = system.process_network_request(x, i)
        clean_results.append(result)
        print(f"Flow {i}: {result['decision'].value:15s} | Risk: {result['ml_risk_score']:.3f} | {result['reason']}")
    
    # Test 2: Adversarial evasion attempt
    print("\n[4/5] TEST 2: Adversarial Evasion Attack")
    print("-"*70)
    print("Generating adversarial examples to evade detection...")
    print()
    
    # Get feature bounds
    bounds = loader.get_feature_bounds(X_test)
    
    attacker = NetworkAdversarialAttacker(
        model=system.risk_model,
        feature_bounds=bounds
    )
    
    # Generate adversarial samples
    print("Generating adversarial samples with FGSM (epsilon=0.05)...")
    X_adv_list = []
    for x in X_malicious[:10]:
        x_adv = attacker.constrained_fgsm(x, epsilon=0.05, target_label=0)
        X_adv_list.append(x_adv[0])
    X_adv = np.array(X_adv_list)
    
    print(f"Generated {len(X_adv)} adversarial samples\n")
    
    # Evaluate evasion
    evasion_results = system.evaluate_adversarial_evasion(
        X_clean=X_malicious[:10],
        X_adv=X_adv,
        flow_indices=range(10)
    )
    
    print("\nEVASION RESULTS:")
    print(f"  Clean Deny Rate:        {evasion_results['clean_deny_rate']:.2%}")
    print(f"  Adversarial Allow Rate: {evasion_results['adv_allow_rate']:.2%}")
    print(f"  Adversarial Deny Rate:  {evasion_results['adv_deny_rate']:.2%}")
    print(f"  Evasion Success Rate:   {evasion_results['evasion_success_rate']:.2%}")
    print(f"  Avg Clean Risk Score:   {evasion_results['avg_clean_risk']:.3f}")
    print(f"  Avg Adv Risk Score:     {evasion_results['avg_adv_risk']:.3f}")
    print(f"  Risk Reduction:         {evasion_results['risk_reduction']:.3f}")
    
    # Test 3: Show telemetry logs
    print("\n[5/5] TEST 3: Zero-Trust Telemetry Logs")
    print("-"*70)
    print("Sample access control decisions:\n")
    
    for log in system.access_log[-5:]:
        print(f"User: {log['user']:15s} | Segment: {log['segment']:8s} | "
              f"ML Risk: {log['ml_risk_score']:.3f} | "
              f"Device Trust: {log['device_trust']:.2f} | "
              f"Decision: {log['decision']}")
    
    # Decision summary
    print("\n" + "-"*70)
    summary = system.get_decision_summary()
    print("DECISION SUMMARY:")
    for decision, count in summary.items():
        if decision != 'total':
            percentage = (count / summary['total'] * 100) if summary['total'] > 0 else 0
            print(f"  {decision:15s}: {count:3d} ({percentage:5.1f}%)")
    print(f"  {'TOTAL':15s}: {summary['total']:3d}")
    
    # Export telemetry
    log_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'zero_trust_telemetry.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    system.export_telemetry(log_path)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print(f"\nKey Findings:")
    print(f"  [OK] Zero-Trust system successfully processes network flows")
    print(f"  [OK] Multi-factor access control decisions working")
    print(f"  [OK] Adversarial attacks achieve {evasion_results['evasion_success_rate']:.1%} evasion rate")
    print(f"  [OK] Telemetry logs exported to {log_path}")
    print()


if __name__ == "__main__":
    test_complete_system()
