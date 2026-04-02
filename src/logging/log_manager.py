"""
SOC/Red Team Log Manager Module

Provides comprehensive logging infrastructure for attack simulations,
defense operations, and security event tracking with multi-format export.
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Log format types
LogFormat = Literal['json', 'txt', 'md', 'csv']


class LogManager:
    """
    Centralized logging system for SOC operations and red team attacks.
    
    Supports multiple export formats and provides structured event tracking
    for comprehensive security analysis and audit trails.
    """
    
    def __init__(self, base_dir: str = "logs"):
        """
        Initialize the log manager.
        
        Args:
            base_dir: Base directory for all logs (default: "logs")
        """
        self.base_dir = Path(base_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.sessions_dir = self.base_dir / "sessions"
        self.attacks_dir = self.base_dir / "attacks"
        self.analytics_dir = self.base_dir / "analytics"
        
        for directory in [self.sessions_dir, self.attacks_dir, self.analytics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # In-memory log storage for current session
        self.current_session_logs: List[Dict[str, Any]] = []
        self.session_metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_events": 0,
            "attack_count": 0,
            "defense_count": 0
        }
        
        logger.info(f"Log Manager initialized - Session ID: {self.session_id}")
    
    def log_attack_event(
        self,
        attack_type: str,
        attack_params: Dict[str, Any],
        model_behavior: Dict[str, Any],
        defense_response: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a red team attack event with comprehensive details.
        
        Args:
            attack_type: Type of attack (e.g., 'HopSkipJump', 'FGM')
            attack_params: Attack parameters (epsilon, queries, etc.)
            model_behavior: Model predictions and confidence scores
            defense_response: Defense system response (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Event ID (timestamp-based unique identifier)
        """
        event_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        event = {
            "event_id": event_id,
            "event_type": "attack",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "attack": {
                "type": attack_type,
                **attack_params
            },
            "model_behavior": model_behavior,
            "defense": defense_response or {},
            "metadata": metadata or {}
        }
        
        self.current_session_logs.append(event)
        self.session_metadata["attack_count"] += 1
        self.session_metadata["total_events"] += 1
        
        logger.debug(f"Logged attack event: {event_id} ({attack_type})")
        return event_id
    
    def log_defense_event(
        self,
        defense_mode: str,
        input_sample: Dict[str, Any],
        decision: Dict[str, Any],
        detection_flags: Dict[str, bool],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a SOC defense operation event.
        
        Args:
            defense_mode: Security mode (e.g., 'Standard', 'Lockdown')
            input_sample: Information about the analyzed sample
            decision: Final defense decision (Allow/Deny)
            detection_flags: Flags from different detection layers
            metadata: Additional metadata (optional)
            
        Returns:
            Event ID
        """
        event_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        event = {
            "event_id": event_id,
            "event_type": "defense",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "defense_mode": defense_mode,
            "input_sample": input_sample,
            "decision": decision,
            "detection_flags": detection_flags,
            "metadata": metadata or {}
        }
        
        self.current_session_logs.append(event)
        self.session_metadata["defense_count"] += 1
        self.session_metadata["total_events"] += 1
        
        logger.debug(f"Logged defense event: {event_id}")
        return event_id
    
    def log_batch_attack(
        self,
        attack_type: str,
        summary_stats: Dict[str, Any],
        individual_results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a batch attack operation (multiple samples at once).
        
        Args:
            attack_type: Type of attack
            summary_stats: Aggregate statistics
            individual_results: Per-sample results (optional)
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        event_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        event = {
            "event_id": event_id,
            "event_type": "batch_attack",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "attack_type": attack_type,
            "summary": summary_stats,
            "num_samples": len(individual_results),
            "individual_results": individual_results[:100],  # Limit to 100 for space
            "metadata": metadata or {}
        }
        
        self.current_session_logs.append(event)
        self.session_metadata["attack_count"] += 1
        self.session_metadata["total_events"] += 1
        
        logger.info(f"Logged batch attack: {attack_type} ({len(individual_results)} samples)")
        return event_id
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def export_logs(
        self,
        format: LogFormat = 'json',
        filename: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Export current session logs to specified format.
        
        Args:
            format: Export format ('json', 'txt', 'md', 'csv')
            filename: Custom filename (auto-generated if None)
            output_dir: Output directory (uses sessions_dir if None)
            
        Returns:
            Path to exported file
        """
        if not self.current_session_logs:
            logger.warning("No logs to export")
            return ""
        
        # Finalize session metadata
        self.session_metadata["end_time"] = datetime.now().isoformat()
        
        # Determine output directory and filename
        if output_dir is None:
            output_dir = str(self.sessions_dir)
        
        if filename is None:
            filename = f"session_{self.session_id}.{format}"
        elif not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        filepath = os.path.join(output_dir, filename)
        
        # Convert numpy types to serializable
        serializable_logs = self._convert_to_serializable(self.current_session_logs)
        serializable_metadata = self._convert_to_serializable(self.session_metadata)
        
        # Export based on format
        if format == 'json':
            self._export_json(filepath, serializable_logs, serializable_metadata)
        elif format == 'txt':
            self._export_txt(filepath, serializable_logs, serializable_metadata)
        elif format == 'md':
            self._export_markdown(filepath, serializable_logs, serializable_metadata)
        elif format == 'csv':
            self._export_csv(filepath, serializable_logs, serializable_metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(serializable_logs)} log entries to {filepath}")
        return filepath
    
    def _export_json(
        self,
        filepath: str,
        logs: List[Dict],
        metadata: Dict
    ) -> None:
        """Export logs as JSON."""
        data = {
            "metadata": metadata,
            "logs": logs
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=False)
    
    def _export_txt(
        self,
        filepath: str,
        logs: List[Dict],
        metadata: Dict
    ) -> None:
        """Export logs as plain text."""
        with open(filepath, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"SOC/RED TEAM LOG - Session {metadata['session_id']}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write metadata
            f.write("SESSION METADATA:\n")
            f.write("-" * 80 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Write logs
            f.write("LOG ENTRIES:\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, log in enumerate(logs, 1):
                f.write(f"[{idx}] Event ID: {log['event_id']}\n")
                f.write(f"    Type: {log['event_type']}\n")
                f.write(f"    Timestamp: {log['timestamp']}\n")
                
                if log['event_type'] == 'attack':
                    f.write(f"    Attack Type: {log['attack']['type']}\n")
                    f.write(f"    Attack Params: {log['attack']}\n")
                    f.write(f"    Model Behavior: {log['model_behavior']}\n")
                    if log.get('defense'):
                        f.write(f"    Defense Response: {log['defense']}\n")
                
                elif log['event_type'] == 'defense':
                    f.write(f"    Defense Mode: {log['defense_mode']}\n")
                    f.write(f"    Decision: {log['decision']}\n")
                    f.write(f"    Detection Flags: {log['detection_flags']}\n")
                
                elif log['event_type'] == 'batch_attack':
                    f.write(f"    Attack Type: {log['attack_type']}\n")
                    f.write(f"    Num Samples: {log['num_samples']}\n")
                    f.write(f"    Summary: {log['summary']}\n")
                
                f.write("\n" + "-" * 80 + "\n\n")
    
    def _export_markdown(
        self,
        filepath: str,
        logs: List[Dict],
        metadata: Dict
    ) -> None:
        """Export logs as Markdown."""
        with open(filepath, 'w') as f:
            # Write header
            f.write(f"# SOC/Red Team Session Log\n\n")
            f.write(f"**Session ID:** `{metadata['session_id']}`\n\n")
            
            # Write metadata table
            f.write("## Session Metadata\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in metadata.items():
                f.write(f"| {key} | {value} |\n")
            f.write("\n")
            
            # Write logs
            f.write("## Event Log\n\n")
            
            for idx, log in enumerate(logs, 1):
                f.write(f"### Event {idx}: {log['event_type'].upper()}\n\n")
                f.write(f"- **Event ID:** `{log['event_id']}`\n")
                f.write(f"- **Timestamp:** {log['timestamp']}\n")
                
                if log['event_type'] == 'attack':
                    f.write(f"- **Attack Type:** {log['attack']['type']}\n")
                    f.write(f"- **Parameters:**\n")
                    for key, value in log['attack'].items():
                        if key != 'type':
                            f.write(f"  - {key}: {value}\n")
                    
                    f.write(f"- **Model Behavior:**\n")
                    for key, value in log['model_behavior'].items():
                        f.write(f"  - {key}: {value}\n")
                    
                    if log.get('defense'):
                        f.write(f"- **Defense Response:**\n")
                        for key, value in log['defense'].items():
                            f.write(f"  - {key}: {value}\n")
                
                elif log['event_type'] == 'defense':
                    f.write(f"- **Defense Mode:** {log['defense_mode']}\n")
                    f.write(f"- **Decision:** {log['decision']}\n")
                    f.write(f"- **Detection Flags:**\n")
                    for key, value in log['detection_flags'].items():
                        f.write(f"  - {key}: {value}\n")
                
                elif log['event_type'] == 'batch_attack':
                    f.write(f"- **Attack Type:** {log['attack_type']}\n")
                    f.write(f"- **Samples Processed:** {log['num_samples']}\n")
                    f.write(f"- **Summary Statistics:**\n")
                    for key, value in log['summary'].items():
                        f.write(f"  - {key}: {value}\n")
                
                f.write("\n---\n\n")
    
    def _export_csv(
        self,
        filepath: str,
        logs: List[Dict],
        metadata: Dict
    ) -> None:
        """Export logs as CSV (flattened structure)."""
        # Flatten log entries for CSV export
        flattened_logs = []
        
        for log in logs:
            flat_log = {
                'event_id': log['event_id'],
                'event_type': log['event_type'],
                'timestamp': log['timestamp'],
                'session_id': log['session_id']
            }
            
            if log['event_type'] == 'attack':
                flat_log.update({
                    'attack_type': log['attack'].get('type', ''),
                    'epsilon': log['attack'].get('epsilon', ''),
                    'queries': log['attack'].get('queries', ''),
                    'success': log['attack'].get('success', ''),
                    'original_prediction': log['model_behavior'].get('original_prediction', ''),
                    'adversarial_prediction': log['model_behavior'].get('adversarial_prediction', ''),
                    'confidence': log['model_behavior'].get('confidence', ''),
                    'defense_decision': log.get('defense', {}).get('final_decision', '')
                })
            
            elif log['event_type'] == 'defense':
                flat_log.update({
                    'defense_mode': log.get('defense_mode', ''),
                    'decision': log.get('decision', {}).get('action', ''),
                    'rf_flagged': log.get('detection_flags', {}).get('rf_attack', ''),
                    'anomaly_flagged': log.get('detection_flags', {}).get('anomaly', ''),
                    'uncertainty_flagged': log.get('detection_flags', {}).get('uncertain', '')
                })
            
            elif log['event_type'] == 'batch_attack':
                flat_log.update({
                    'attack_type': log.get('attack_type', ''),
                    'num_samples': log.get('num_samples', ''),
                    'mean_evasion': log.get('summary', {}).get('mean_evasion_rate', ''),
                    'mean_accuracy': log.get('summary', {}).get('mean_accuracy', '')
                })
            
            flattened_logs.append(flat_log)
        
        # Write CSV - collect all possible fieldnames from all logs
        if flattened_logs:
            # Gather all unique fieldnames across all flattened logs
            all_fieldnames = set()
            for log in flattened_logs:
                all_fieldnames.update(log.keys())
            
            fieldnames = sorted(list(all_fieldnames))
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(flattened_logs)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for current session."""
        attack_types = {}
        defense_decisions = {'allow': 0, 'deny': 0}
        
        for log in self.current_session_logs:
            if log['event_type'] == 'attack':
                attack_type = log['attack'].get('type', 'unknown')
                attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
            
            elif log['event_type'] == 'defense':
                decision = log.get('decision', {}).get('action', 'unknown')
                if decision.lower() in ['allow', 'allowed']:
                    defense_decisions['allow'] += 1
                elif decision.lower() in ['deny', 'denied']:
                    defense_decisions['deny'] += 1
        
        return {
            'session_id': self.session_id,
            'total_events': len(self.current_session_logs),
            'attack_types': attack_types,
            'defense_decisions': defense_decisions,
            'session_metadata': self.session_metadata
        }
    
    def clear_session(self) -> None:
        """Clear current session logs and start new session."""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_logs = []
        self.session_metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_events": 0,
            "attack_count": 0,
            "defense_count": 0
        }
        logger.info(f"Session cleared - New Session ID: {self.session_id}")


# Convenience function for quick logging setup
def create_log_manager(base_dir: str = "logs") -> LogManager:
    """Create and return a LogManager instance."""
    return LogManager(base_dir=base_dir)
