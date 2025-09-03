"""
Dashboard Configuration
Centralized configuration for the NDIS Carer Data Pipeline Dashboard
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "data"

# Dashboard settings
DASHBOARD_CONFIG = {
    "title": "NDIS Carer Data Pipeline Dashboard",
    "subtitle": "Data Quality & Pipeline Health Monitoring",
    "version": "1.0.0",
    "refresh_interval": 30,  # seconds
    "max_records_display": 1000,
    "pagination_size": 50,
}

# KPI thresholds
KPI_THRESHOLDS = {
    "pass_rate": {
        "excellent": 95.0,
        "good": 85.0,
        "warning": 70.0,
        "critical": 50.0
    },
    "fallback_ratio": {
        "excellent": 5.0,
        "good": 10.0,
        "warning": 20.0,
        "critical": 30.0
    },
    "throughput": {
        "excellent": 100.0,  # records per minute
        "good": 50.0,
        "warning": 20.0,
        "critical": 10.0
    },
    "avg_narrative_length": {
        "min_acceptable": 50,
        "max_acceptable": 500,
        "optimal_min": 100,
        "optimal_max": 300
    },
    "field_coverage": {
        "excellent": 95.0,
        "good": 85.0,
        "warning": 70.0,
        "critical": 60.0
    }
}

# Chart colors
CHART_COLORS = {
    "excellent": "#28a745",  # green
    "good": "#17a2b8",       # blue
    "warning": "#ffc107",    # yellow
    "critical": "#dc3545",   # red
    "neutral": "#6c757d",    # gray
    "primary": "#007bff",    # primary blue
    "secondary": "#6c757d",  # secondary gray
}

# Data sources
DATA_SOURCES = {
    "output_files": {
        "pattern": "*.{json,jsonl,csv}",
        "directory": OUTPUT_DIR
    },
    "validation_reports": {
        "pattern": "*validation*.json",
        "directory": OUTPUT_DIR
    },
    "logs": {
        "pattern": "*.log",
        "directory": LOGS_DIR
    }
}

# Quality gates configuration
QUALITY_GATES = {
    "parsing_errors": {
        "name": "Data Parsing Errors",
        "description": "Records that failed to parse correctly",
        "threshold": 5.0  # percentage
    },
    "validation_failures": {
        "name": "Schema Validation Failures", 
        "description": "Records that failed schema validation",
        "threshold": 10.0
    },
    "duplicate_records": {
        "name": "Duplicate Records",
        "description": "Records identified as duplicates",
        "threshold": 2.0
    },
    "privacy_risks": {
        "name": "Privacy Risk Detections",
        "description": "Records with potential privacy risks",
        "threshold": 1.0
    },
    "narrative_quality": {
        "name": "Poor Narrative Quality",
        "description": "Records with low-quality narrative content",
        "threshold": 15.0
    }
}

# Export settings
EXPORT_CONFIG = {
    "formats": ["json", "csv", "xlsx"],
    "max_export_records": 10000,
    "include_metadata": True
}

# Alert settings  
ALERT_CONFIG = {
    "enabled": True,
    "email_notifications": False,
    "thresholds": {
        "pass_rate_drop": 10.0,  # percentage drop
        "failure_spike": 20.0,   # percentage increase
        "throughput_drop": 30.0  # percentage drop
    }
}

def get_dashboard_config() -> Dict[str, Any]:
    """Get complete dashboard configuration"""
    return {
        "dashboard": DASHBOARD_CONFIG,
        "kpi_thresholds": KPI_THRESHOLDS,
        "chart_colors": CHART_COLORS,
        "data_sources": DATA_SOURCES,
        "quality_gates": QUALITY_GATES,
        "export": EXPORT_CONFIG,
        "alerts": ALERT_CONFIG
    }

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [OUTPUT_DIR, LOGS_DIR, REPORTS_DIR, DASHBOARD_DATA_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Test configuration
    config = get_dashboard_config()
    print("Dashboard Configuration loaded successfully")
    print(f"Title: {config['dashboard']['title']}")
    print(f"Data sources: {len(config['data_sources'])} configured")
    print(f"Quality gates: {len(config['quality_gates'])} configured")



