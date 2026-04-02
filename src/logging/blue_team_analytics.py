"""
Blue Team Analytics Module

Analyzes attack logs to identify patterns, vulnerabilities, and recommend
defense improvements. Supports adversarial retraining and automated hardening.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class BlueTeamAnalytics:
    """
    Blue Team analytics system for attack pattern analysis and defense hardening.
    
    Analyzes logged attack data to identify weaknesses and recommend improvements.
    """
    
    def __init__(self, log_directory: str = "logs/sessions"):
        """
        Initialize the analytics system.
        
        Args:
            log_directory: Directory containing log files
        """
        self.log_directory = Path(log_directory)
        self.logs_data: List[Dict[str, Any]] = []
        self.analysis_results: Dict[str, Any] = {}
        
        logger.info(f"Blue Team Analytics initialized - Monitoring: {self.log_directory}")
    
    def load_logs_from_file(self, filepath: str) -> None:
        """
        Load logs from a JSON file.
        
        Args:
            filepath: Path to the log file
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            if 'logs' in data:
                self.logs_data.extend(data['logs'])
                logger.info(f"Loaded {len(data['logs'])} events from {filepath}")
            else:
                logger.warning(f"No 'logs' key found in {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to load logs from {filepath}: {e}")
            raise
    
    def load_all_logs(self, pattern: str = "*.json") -> int:
        """
        Load all log files matching pattern from the log directory.
        
        Args:
            pattern: File pattern to match (default: "*.json")
            
        Returns:
            Number of log files loaded
        """
        log_files = list(self.log_directory.glob(pattern))
        
        if not log_files:
            logger.warning(f"No log files found in {self.log_directory}")
            return 0
        
        for log_file in log_files:
            try:
                self.load_logs_from_file(str(log_file))
            except Exception as e:
                logger.error(f"Failed to load {log_file}: {e}")
                continue
        
        logger.info(f"Loaded total of {len(self.logs_data)} events from {len(log_files)} files")
        return len(log_files)
    
    def analyze_attack_patterns(self) -> Dict[str, Any]:
        """
        Analyze attack patterns to identify dominant vectors and success rates.
        
        Returns:
            Dictionary containing attack pattern analysis
        """
        attack_events = [log for log in self.logs_data if log.get('event_type') == 'attack']
        batch_attacks = [log for log in self.logs_data if log.get('event_type') == 'batch_attack']
        
        if not attack_events and not batch_attacks:
            logger.warning("No attack events found in logs")
            return {
                "total_attacks": 0,
                "attack_types": {},
                "recommendations": ["No attack data available for analysis"]
            }
        
        # Attack type distribution
        attack_types = Counter()
        epsilon_values = []
        query_counts = []
        success_flags = []
        
        for event in attack_events:
            attack_data = event.get('attack', {})
            attack_type = attack_data.get('type', 'unknown')
            attack_types[attack_type] += 1
            
            if 'epsilon' in attack_data:
                epsilon_values.append(float(attack_data['epsilon']))
            
            if 'queries' in attack_data:
                query_counts.append(int(attack_data['queries']))
            
            if 'success' in attack_data:
                success_flags.append(bool(attack_data['success']))
        
        # Analyze batch attacks
        for event in batch_attacks:
            attack_type = event.get('attack_type', 'unknown')
            attack_types[attack_type] += event.get('num_samples', 1)
            
            summary = event.get('summary', {})
            if 'mean_evasion_rate' in summary:
                # Approximate success rate from evasion rate
                success_flags.extend([True] * int(summary.get('num_samples', 0) * summary['mean_evasion_rate']))
        
        # Calculate statistics
        analysis = {
            "total_attacks": len(attack_events) + sum(e.get('num_samples', 0) for e in batch_attacks),
            "attack_types": dict(attack_types),
            "most_common_attack": attack_types.most_common(1)[0][0] if attack_types else "N/A",
            "epsilon_stats": {
                "min": float(np.min(epsilon_values)) if epsilon_values else 0.0,
                "max": float(np.max(epsilon_values)) if epsilon_values else 0.0,
                "mean": float(np.mean(epsilon_values)) if epsilon_values else 0.0,
                "median": float(np.median(epsilon_values)) if epsilon_values else 0.0
            } if epsilon_values else {},
            "query_stats": {
                "min": int(np.min(query_counts)) if query_counts else 0,
                "max": int(np.max(query_counts)) if query_counts else 0,
                "mean": float(np.mean(query_counts)) if query_counts else 0.0
            } if query_counts else {},
            "success_rate": float(np.mean(success_flags)) if success_flags else 0.0
        }
        
        self.analysis_results['attack_patterns'] = analysis
        return analysis
    
    def analyze_defense_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze defense system effectiveness against logged attacks.
        
        Returns:
            Dictionary containing defense performance metrics
        """
        attack_events = [log for log in self.logs_data 
                        if log.get('event_type') == 'attack' and log.get('defense')]
        
        if not attack_events:
            logger.warning("No attack events with defense data found")
            return {
                "total_defended_attacks": 0,
                "recommendations": ["No defense data available for analysis"]
            }
        
        # Analyze defense decisions
        defense_decisions = []
        detection_layers = {
            'rf_attack': [],
            'anomaly': [],
            'uncertainty': []
        }
        
        for event in attack_events:
            defense = event.get('defense', {})
            
            # Defense decision
            decision = defense.get('final_decision', '')
            if decision:
                defense_decisions.append(decision.lower())
            
            # Detection layer flags
            if 'isolation_flag' in defense:
                detection_layers['anomaly'].append(bool(defense['isolation_flag']))
            
            if 'confidence_threshold_triggered' in defense:
                detection_layers['uncertainty'].append(bool(defense['confidence_threshold_triggered']))
        
        # Calculate metrics
        total_attacks = len(defense_decisions)
        blocked_count = sum(1 for d in defense_decisions if d in ['deny', 'denied', 'block'])
        
        analysis = {
            "total_defended_attacks": total_attacks,
            "blocked": blocked_count,
            "allowed": total_attacks - blocked_count,
            "block_rate": float(blocked_count / total_attacks) if total_attacks > 0 else 0.0,
            "detection_layer_performance": {
                layer: {
                    "triggered": sum(flags),
                    "trigger_rate": float(np.mean(flags)) if flags else 0.0
                }
                for layer, flags in detection_layers.items() if flags
            }
        }
        
        self.analysis_results['defense_effectiveness'] = analysis
        return analysis
    
    def identify_vulnerabilities(self) -> Dict[str, Any]:
        """
        Identify system vulnerabilities based on successful attacks.
        
        Returns:
            Dictionary containing vulnerability analysis and recommendations
        """
        attack_patterns = self.analysis_results.get('attack_patterns', {})
        defense_effectiveness = self.analysis_results.get('defense_effectiveness', {})
        
        vulnerabilities = []
        recommendations = []
        
        # Check attack success rate
        success_rate = attack_patterns.get('success_rate', 0.0)
        if success_rate > 0.3:
            vulnerabilities.append({
                "type": "high_attack_success_rate",
                "severity": "high",
                "description": f"Attack success rate is {success_rate:.1%}, indicating weak defense",
                "affected_metric": "overall_robustness"
            })
            recommendations.append({
                "action": "increase_confidence_threshold",
                "description": "Increase confidence threshold to 0.20 or higher",
                "expected_impact": "Reduce false negatives by 15-25%"
            })
        
        # Check block rate
        block_rate = defense_effectiveness.get('block_rate', 0.0)
        if block_rate < 0.7:
            vulnerabilities.append({
                "type": "low_block_rate",
                "severity": "medium",
                "description": f"Only {block_rate:.1%} of attacks are blocked",
                "affected_metric": "defense_effectiveness"
            })
            recommendations.append({
                "action": "enable_lockdown_mode",
                "description": "Consider enabling lockdown mode for high-risk periods",
                "expected_impact": "Increase block rate to 85-90%"
            })
        
        # Check epsilon distribution (perturbation strength)
        epsilon_stats = attack_patterns.get('epsilon_stats', {})
        if epsilon_stats and epsilon_stats.get('mean', 0) > 0.2:
            vulnerabilities.append({
                "type": "vulnerable_to_large_perturbations",
                "severity": "medium",
                "description": f"Model vulnerable to perturbations with epsilon={epsilon_stats['mean']:.3f}",
                "affected_metric": "adversarial_robustness"
            })
            recommendations.append({
                "action": "adversarial_retraining",
                "description": "Retrain model with adversarial examples at epsilon=0.1-0.3",
                "expected_impact": "Improve robustness by 20-30%"
            })
        
        # Check most common attack type
        most_common = attack_patterns.get('most_common_attack', '')
        attack_types = attack_patterns.get('attack_types', {})
        if most_common and attack_types.get(most_common, 0) > len(self.logs_data) * 0.5:
            vulnerabilities.append({
                "type": "single_attack_vector_dominance",
                "severity": "medium",
                "description": f"{most_common} attacks dominate ({attack_types[most_common]} occurrences)",
                "affected_metric": "defense_diversity"
            })
            recommendations.append({
                "action": "specialized_defense",
                "description": f"Implement specialized defenses against {most_common} attacks",
                "expected_impact": "Reduce success rate of dominant attack by 40-50%"
            })
        
        analysis = {
            "total_vulnerabilities": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "recommendations": recommendations,
            "priority_actions": sorted(recommendations, 
                                      key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(
                                          [v for v in vulnerabilities if v.get('type') in str(x)][0].get('severity', 'low') 
                                          if [v for v in vulnerabilities if v.get('type') in str(x)] else 'low', 
                                          0
                                      ),
                                      reverse=True)[:3]
        }
        
        self.analysis_results['vulnerabilities'] = analysis
        return analysis
    
    def generate_hardening_config(self) -> Dict[str, Any]:
        """
        Generate recommended configuration changes for defense hardening.
        
        Returns:
            Dictionary with recommended config updates
        """
        vulnerabilities = self.analysis_results.get('vulnerabilities', {})
        attack_patterns = self.analysis_results.get('attack_patterns', {})
        
        current_threshold = 0.15  # Default from config
        recommended_threshold = current_threshold
        
        # Adjust threshold based on attack success rate
        success_rate = attack_patterns.get('success_rate', 0.0)
        if success_rate > 0.3:
            recommended_threshold = 0.20
        elif success_rate > 0.4:
            recommended_threshold = 0.25
        
        hardening_config = {
            "confidence_threshold": {
                "current": current_threshold,
                "recommended": recommended_threshold,
                "change": recommended_threshold - current_threshold,
                "reason": f"Attack success rate is {success_rate:.1%}"
            },
            "isolation_contamination": {
                "current": 0.05,
                "recommended": 0.03 if success_rate > 0.3 else 0.05,
                "reason": "Increase anomaly detection sensitivity"
            },
            "enable_adversarial_retraining": {
                "recommended": success_rate > 0.25,
                "reason": "High attack success indicates need for robust training"
            },
            "recommended_actions": [
                rec['action'] for rec in vulnerabilities.get('recommendations', [])
            ]
        }
        
        self.analysis_results['hardening_config'] = hardening_config
        return hardening_config
    
    def extract_adversarial_samples(self, output_dir: str = "logs/analytics") -> str:
        """
        Extract adversarial samples from logs for retraining.
        
        Args:
            output_dir: Directory to save adversarial samples
            
        Returns:
            Path to saved adversarial samples file
        """
        attack_events = [log for log in self.logs_data if log.get('event_type') == 'attack']
        
        if not attack_events:
            logger.warning("No attack events found to extract samples")
            return ""
        
        # Extract attack parameters and metadata for potential retraining
        adversarial_data = []
        
        for event in attack_events:
            attack = event.get('attack', {})
            model_behavior = event.get('model_behavior', {})
            
            sample_data = {
                "attack_type": attack.get('type', ''),
                "epsilon": attack.get('epsilon', 0.0),
                "success": attack.get('success', False),
                "original_prediction": model_behavior.get('original_prediction', 0),
                "adversarial_prediction": model_behavior.get('adversarial_prediction', 0),
                "confidence": model_behavior.get('confidence', 0.0),
                "timestamp": event.get('timestamp', '')
            }
            
            adversarial_data.append(sample_data)
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"adversarial_samples_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump({
                "metadata": {
                    "total_samples": len(adversarial_data),
                    "extraction_date": timestamp,
                    "source_logs": len(self.logs_data)
                },
                "samples": adversarial_data
            }, f, indent=2)
        
        logger.info(f"Extracted {len(adversarial_data)} adversarial samples to {filepath}")
        return filepath
    
    def generate_comprehensive_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive blue team analysis report.
        
        Args:
            output_file: Optional file to save report (JSON format)
            
        Returns:
            Complete analysis report
        """
        # Run all analyses if not already done
        if 'attack_patterns' not in self.analysis_results:
            self.analyze_attack_patterns()
        
        if 'defense_effectiveness' not in self.analysis_results:
            self.analyze_defense_effectiveness()
        
        if 'vulnerabilities' not in self.analysis_results:
            self.identify_vulnerabilities()
        
        if 'hardening_config' not in self.analysis_results:
            self.generate_hardening_config()
        
        report = {
            "report_metadata": {
                "generated_at": pd.Timestamp.now().isoformat(),
                "total_events_analyzed": len(self.logs_data),
                "analysis_version": "1.0"
            },
            "attack_patterns": self.analysis_results.get('attack_patterns', {}),
            "defense_effectiveness": self.analysis_results.get('defense_effectiveness', {}),
            "vulnerabilities": self.analysis_results.get('vulnerabilities', {}),
            "hardening_recommendations": self.analysis_results.get('hardening_config', {}),
            "executive_summary": self._generate_executive_summary()
        }
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Comprehensive report saved to {output_file}")
        
        return report
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of findings."""
        attack_patterns = self.analysis_results.get('attack_patterns', {})
        defense_effectiveness = self.analysis_results.get('defense_effectiveness', {})
        vulnerabilities = self.analysis_results.get('vulnerabilities', {})
        
        total_attacks = attack_patterns.get('total_attacks', 0)
        success_rate = attack_patterns.get('success_rate', 0.0)
        block_rate = defense_effectiveness.get('block_rate', 0.0)
        num_vulnerabilities = vulnerabilities.get('total_vulnerabilities', 0)
        
        # Determine overall security posture
        if success_rate < 0.15 and block_rate > 0.85:
            posture = "STRONG"
            risk_level = "LOW"
        elif success_rate < 0.30 and block_rate > 0.70:
            posture = "MODERATE"
            risk_level = "MEDIUM"
        else:
            posture = "WEAK"
            risk_level = "HIGH"
        
        return {
            "security_posture": posture,
            "risk_level": risk_level,
            "total_attacks_analyzed": total_attacks,
            "attack_success_rate": float(success_rate),
            "defense_block_rate": float(block_rate),
            "critical_vulnerabilities": num_vulnerabilities,
            "top_recommendation": vulnerabilities.get('priority_actions', [{}])[0].get('action', 'None') 
                                  if vulnerabilities.get('priority_actions') else 'None',
            "requires_immediate_action": risk_level == "HIGH"
        }
    
    def create_retraining_dataset(
        self,
        epsilon_range: Tuple[float, float] = (0.1, 0.3),
        output_dir: str = "logs/analytics"
    ) -> str:
        """
        Create a dataset specification for adversarial retraining.
        
        Args:
            epsilon_range: Range of epsilon values to focus on
            output_dir: Directory to save dataset spec
            
        Returns:
            Path to dataset specification file
        """
        attack_patterns = self.analysis_results.get('attack_patterns', {})
        
        # Determine recommended epsilon values based on attack data
        epsilon_stats = attack_patterns.get('epsilon_stats', {})
        if epsilon_stats:
            # Focus on the range where attacks succeeded
            recommended_epsilons = [
                epsilon_stats.get('mean', 0.2),
                epsilon_stats.get('median', 0.15),
                epsilon_stats.get('max', 0.3)
            ]
        else:
            recommended_epsilons = list(np.linspace(epsilon_range[0], epsilon_range[1], 5))
        
        dataset_spec = {
            "purpose": "adversarial_retraining",
            "based_on_analysis": {
                "total_attacks_analyzed": attack_patterns.get('total_attacks', 0),
                "dominant_attack_type": attack_patterns.get('most_common_attack', 'FGM'),
                "observed_epsilon_range": [
                    epsilon_stats.get('min', epsilon_range[0]),
                    epsilon_stats.get('max', epsilon_range[1])
                ] if epsilon_stats else list(epsilon_range)
            },
            "recommended_training_config": {
                "epsilon_values": [float(e) for e in recommended_epsilons],
                "attack_types_to_generate": list(attack_patterns.get('attack_types', {}).keys()) or ['FGM'],
                "augmentation_ratio": 0.2 if attack_patterns.get('success_rate', 0) > 0.3 else 0.1,
                "training_epochs": 100,
                "batch_size": 128
            },
            "expected_improvements": {
                "robustness_increase": "20-30%",
                "evasion_rate_reduction": "15-25%",
                "confidence_stability": "improved"
            }
        }
        
        # Save spec
        os.makedirs(output_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"retraining_spec_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(dataset_spec, f, indent=2)
        
        logger.info(f"Retraining dataset specification saved to {filepath}")
        return filepath


def analyze_logs_and_generate_report(
    log_directory: str = "logs/sessions",
    output_dir: str = "logs/analytics"
) -> str:
    """
    Convenience function to run complete analysis pipeline.
    
    Args:
        log_directory: Directory containing log files
        output_dir: Directory to save reports
        
    Returns:
        Path to comprehensive report file
    """
    analytics = BlueTeamAnalytics(log_directory=log_directory)
    analytics.load_all_logs()
    
    # Run complete analysis
    analytics.analyze_attack_patterns()
    analytics.analyze_defense_effectiveness()
    analytics.identify_vulnerabilities()
    analytics.generate_hardening_config()
    
    # Generate reports
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"blue_team_report_{timestamp}.json")
    
    analytics.generate_comprehensive_report(output_file=report_path)
    
    # Also extract adversarial samples and create retraining spec
    analytics.extract_adversarial_samples(output_dir=output_dir)
    analytics.create_retraining_dataset(output_dir=output_dir)
    
    return report_path
