"""
SOC Logging and Blue Team Analytics Module

Provides comprehensive logging and analytics capabilities for adversarial ML security operations.

Modules:
- log_manager: Multi-format event logging (JSON, TXT, MD, CSV)
- blue_team_analytics: Automated vulnerability detection and hardening recommendations
- attack_logging_wrappers: Drop-in replacements for attack functions with integrated logging
"""

from .log_manager import LogManager
from .blue_team_analytics import BlueTeamAnalytics, analyze_logs_and_generate_report
from .attack_logging_wrappers import (
    run_blackbox_attack_with_logging,
    run_whitebox_attack_with_logging,
    ensemble_defense_predict_with_logging,
    create_logged_attack_session
)

__all__ = [
    'LogManager',
    'BlueTeamAnalytics',
    'analyze_logs_and_generate_report',
    'run_blackbox_attack_with_logging',
    'run_whitebox_attack_with_logging',
    'ensemble_defense_predict_with_logging',
    'create_logged_attack_session'
]
