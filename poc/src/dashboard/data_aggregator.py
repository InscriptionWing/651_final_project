"""
Data Aggregator for Dashboard
ETL job that aggregates outputs, logs, and metrics into a metrics store
"""

import json
import pandas as pd
import sqlite3
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import glob
import re
from dataclasses import asdict

from config import get_dashboard_config, OUTPUT_DIR, LOGS_DIR, DASHBOARD_DATA_DIR
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# 尝试导入英文数据模式，如果失败则使用默认模式
try:
    from english_data_schema import CarerServiceRecord, ServiceType, ServiceOutcome, LocationType, EnglishDataValidator as DataValidator
    logger.info("Using English data schema")
except ImportError:
    try:
        from carer_data_schema import CarerServiceRecord, ServiceType, ServiceOutcome, LocationType, DataValidator
        logger.info("Using default data schema")
    except ImportError:
        logger.error("Could not import data schema")
        raise

class DataAggregator:
    """Aggregates data from various sources for dashboard consumption"""
    
    def __init__(self):
        self.config = get_dashboard_config()
        self.db_path = DASHBOARD_DATA_DIR / "metrics.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Main metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_records INTEGER,
                    passed_records INTEGER,
                    failed_records INTEGER,
                    fallback_records INTEGER,
                    generation_time_seconds REAL,
                    validation_time_seconds REAL,
                    overall_score REAL,
                    privacy_score REAL,
                    realism_score REAL,
                    throughput_per_minute REAL
                )
            """)
            
            # Record details table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS record_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT UNIQUE,
                    carer_id TEXT,
                    carer_name TEXT,
                    participant_id TEXT,
                    service_date DATE,
                    service_type TEXT,
                    duration_hours REAL,
                    narrative_length INTEGER,
                    narrative_notes TEXT,
                    location_type TEXT,
                    service_outcome TEXT,
                    support_techniques TEXT,
                    challenges_encountered TEXT,
                    validation_status TEXT,
                    source_type TEXT,
                    created_timestamp DATETIME,
                    validation_errors TEXT
                )
            """)
            
            # Quality gates table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_gate_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    gate_name TEXT,
                    total_records INTEGER,
                    failed_records INTEGER,
                    failure_rate REAL,
                    details TEXT
                )
            """)
            
            # Template usage tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS template_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    template_type TEXT,
                    template_category TEXT,
                    usage_count INTEGER,
                    diversity_score REAL
                )
            """)
    
    def aggregate_all_data(self) -> Dict[str, Any]:
        """Run complete data aggregation process"""
        logger.info("Starting data aggregation process")
        
        try:
            # Aggregate output files
            output_metrics = self._aggregate_output_files()
            
            # Aggregate validation reports
            validation_metrics = self._aggregate_validation_reports()
            
            # Aggregate logs
            log_metrics = self._aggregate_logs()
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(output_metrics, validation_metrics)
            
            # Store in database
            self._store_metrics(output_metrics, validation_metrics, derived_metrics)
            
            # Generate summary
            summary = {
                "aggregation_timestamp": datetime.now().isoformat(),
                "output_metrics": output_metrics,
                "validation_metrics": validation_metrics,
                "log_metrics": log_metrics,
                "derived_metrics": derived_metrics,
                "status": "success"
            }
            
            logger.info("Data aggregation completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Data aggregation failed: {e}")
            return {
                "aggregation_timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def _aggregate_output_files(self) -> Dict[str, Any]:
        """Aggregate metrics from output files"""
        output_files = list(OUTPUT_DIR.glob("*.json")) + list(OUTPUT_DIR.glob("*.jsonl"))
        
        if not output_files:
            return {"total_files": 0, "total_records": 0, "latest_file": None}
        
        # Find latest file
        latest_file = max(output_files, key=lambda f: f.stat().st_mtime)
        
        # Load and analyze latest dataset
        records = self._load_records_from_file(latest_file)
        
        if not records:
            return {"total_files": len(output_files), "total_records": 0, "latest_file": str(latest_file)}
        
        # Calculate metrics
        metrics = {
            "total_files": len(output_files),
            "total_records": len(records),
            "latest_file": str(latest_file),
            "file_timestamp": datetime.fromtimestamp(latest_file.stat().st_mtime),
            "service_type_distribution": self._calculate_service_type_distribution(records),
            "outcome_distribution": self._calculate_outcome_distribution(records),
            "location_distribution": self._calculate_location_distribution(records),
            "duration_stats": self._calculate_duration_stats(records),
            "narrative_stats": self._calculate_narrative_stats(records),
            "date_range": self._calculate_date_range(records),
            "unique_carers": len(set(r.get("carer_id") for r in records if r.get("carer_id"))),
            "unique_participants": len(set(r.get("participant_id") for r in records if r.get("participant_id")))
        }
        
        return metrics
    
    def _aggregate_validation_reports(self) -> Dict[str, Any]:
        """Aggregate metrics from validation reports"""
        validation_files = list(OUTPUT_DIR.glob("*validation*.json"))
        
        if not validation_files:
            return {"total_reports": 0, "latest_validation": None}
        
        # Find latest validation report
        latest_report = max(validation_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_report, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
            
            metrics = {
                "total_reports": len(validation_files),
                "latest_report": str(latest_report),
                "report_timestamp": datetime.fromtimestamp(latest_report.stat().st_mtime),
                "overall_score": validation_data.get("overall_score", 0),
                "privacy_score": validation_data.get("privacy_analysis", {}).get("anonymization_score", 0),
                "realism_score": validation_data.get("utility_analysis", {}).get("realism_score", 0),
                "schema_compliance": validation_data.get("schema_validation", {}).get("compliance_rate", 0),
                "quality_gates": self._extract_quality_gate_results(validation_data),
                "validation_errors": validation_data.get("validation_errors", []),
                "recommendations": validation_data.get("recommendations", [])
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load validation report {latest_report}: {e}")
            return {"total_reports": len(validation_files), "error": str(e)}
    
    def _aggregate_logs(self) -> Dict[str, Any]:
        """Aggregate metrics from log files"""
        log_files = list(LOGS_DIR.glob("*.log"))
        
        if not log_files:
            return {"total_logs": 0, "recent_errors": []}
        
        # Analyze recent log entries
        recent_errors = []
        recent_warnings = []
        generation_times = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-1000:]  # Last 1000 lines
                
                for line in lines:
                    if "ERROR" in line:
                        recent_errors.append(line.strip())
                    elif "WARNING" in line:
                        recent_warnings.append(line.strip())
                    
                    # Extract generation times
                    time_match = re.search(r'生成用时:\s*([\d.]+)秒', line)
                    if time_match:
                        generation_times.append(float(time_match.group(1)))
                        
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")
        
        return {
            "total_logs": len(log_files),
            "recent_errors": recent_errors[-10:],  # Last 10 errors
            "recent_warnings": recent_warnings[-10:],  # Last 10 warnings
            "avg_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0,
            "total_generations": len(generation_times)
        }
    
    def _calculate_derived_metrics(self, output_metrics: Dict, validation_metrics: Dict) -> Dict[str, Any]:
        """Calculate derived metrics from aggregated data"""
        total_records = output_metrics.get("total_records", 0)
        overall_score = validation_metrics.get("overall_score", 0)
        
        # Calculate pass rate (assuming scores > 70 are "passed")
        pass_rate = min(overall_score / 70.0 * 100, 100) if overall_score > 0 else 0
        
        # Calculate fallback ratio (estimated from validation errors)
        validation_errors = len(validation_metrics.get("validation_errors", []))
        fallback_ratio = (validation_errors / total_records * 100) if total_records > 0 else 0
        
        # Calculate throughput (records per minute)
        generation_time = output_metrics.get("generation_time_seconds", 0)
        throughput = (total_records / (generation_time / 60)) if generation_time > 0 else 0
        
        # Calculate field coverage
        narrative_stats = output_metrics.get("narrative_stats", {})
        avg_narrative_length = narrative_stats.get("avg_length", 0)
        field_coverage = self._calculate_field_coverage_score(output_metrics)
        
        return {
            "pass_rate": round(pass_rate, 2),
            "fallback_ratio": round(fallback_ratio, 2),
            "throughput_per_minute": round(throughput, 2),
            "avg_narrative_length": round(avg_narrative_length, 2),
            "field_coverage": round(field_coverage, 2),
            "quality_status": self._determine_quality_status(pass_rate, fallback_ratio, throughput),
            "trend_analysis": self._calculate_trends()
        }
    
    def _calculate_field_coverage_score(self, output_metrics: Dict) -> float:
        """Calculate field coverage score based on data completeness"""
        total_records = output_metrics.get("total_records", 0)
        if total_records == 0:
            return 0
        
        # Estimate coverage based on available data
        service_types = len(output_metrics.get("service_type_distribution", {}))
        outcomes = len(output_metrics.get("outcome_distribution", {}))
        locations = len(output_metrics.get("location_distribution", {}))
        
        # Simple scoring based on diversity
        coverage_score = min((service_types / 6 + outcomes / 4 + locations / 9) / 3 * 100, 100)
        return coverage_score
    
    def _determine_quality_status(self, pass_rate: float, fallback_ratio: float, throughput: float) -> str:
        """Determine overall quality status"""
        thresholds = self.config["kpi_thresholds"]
        
        if (pass_rate >= thresholds["pass_rate"]["excellent"] and 
            fallback_ratio <= thresholds["fallback_ratio"]["excellent"] and
            throughput >= thresholds["throughput"]["excellent"]):
            return "excellent"
        elif (pass_rate >= thresholds["pass_rate"]["good"] and
              fallback_ratio <= thresholds["fallback_ratio"]["good"] and
              throughput >= thresholds["throughput"]["good"]):
            return "good"
        elif (pass_rate >= thresholds["pass_rate"]["warning"] and
              fallback_ratio <= thresholds["fallback_ratio"]["warning"]):
            return "warning"
        else:
            return "critical"
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate trend analysis from historical data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get last 24 hours of data
                query = """
                    SELECT * FROM pipeline_metrics 
                    WHERE timestamp >= datetime('now', '-24 hours')
                    ORDER BY timestamp DESC
                """
                df = pd.read_sql_query(query, conn)
                
                if len(df) < 2:
                    return {"trend_available": False, "reason": "insufficient_data"}
                
                # Calculate trends
                latest = df.iloc[0]
                previous = df.iloc[-1]
                
                trends = {
                    "trend_available": True,
                    "pass_rate_trend": self._calculate_trend_direction(latest["overall_score"], previous["overall_score"]),
                    "throughput_trend": self._calculate_trend_direction(latest["throughput_per_minute"], previous["throughput_per_minute"]),
                    "records_trend": self._calculate_trend_direction(latest["total_records"], previous["total_records"]),
                    "time_period_hours": 24,
                    "data_points": len(df)
                }
                
                return trends
                
        except Exception as e:
            logger.error(f"Failed to calculate trends: {e}")
            return {"trend_available": False, "error": str(e)}
    
    def _calculate_trend_direction(self, current: float, previous: float) -> str:
        """Calculate trend direction"""
        if previous == 0:
            return "stable"
        
        change_percent = ((current - previous) / previous) * 100
        
        if change_percent > 5:
            return "up"
        elif change_percent < -5:
            return "down"
        else:
            return "stable"
    
    def _load_records_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load records from JSON or JSONL file"""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                records = data if isinstance(data, list) else [data]
            
            elif file_path.suffix == '.jsonl':
                records = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line.strip()))
            else:
                return []
            
            # 规范化记录格式，确保所有必需字段存在
            normalized_records = []
            for record in records:
                # 确保记录包含基本字段
                normalized_record = {
                    'record_id': record.get('record_id', 'UNKNOWN'),
                    'carer_id': record.get('carer_id', 'UNKNOWN'),
                    'carer_name': record.get('carer_name', record.get('carer_id', 'Unknown Carer')),
                    'participant_id': record.get('participant_id', 'UNKNOWN'),
                    'service_date': record.get('service_date', '2025-01-01'),
                    'service_type': record.get('service_type', 'Unknown'),
                    'duration_hours': float(record.get('duration_hours', 0)),
                    'narrative_notes': record.get('narrative_notes', ''),
                    'location_type': record.get('location_type', 'Unknown'),
                    'service_outcome': record.get('service_outcome', 'neutral'),
                    'support_techniques_used': record.get('support_techniques_used', []),
                    'challenges_encountered': record.get('challenges_encountered', []),
                    'created_timestamp': record.get('created_timestamp', datetime.now().isoformat())
                }
                
                # 添加其他可选字段
                for key, value in record.items():
                    if key not in normalized_record:
                        normalized_record[key] = value
                
                normalized_records.append(normalized_record)
            
            return normalized_records
            
        except Exception as e:
            logger.error(f"Failed to load records from {file_path}: {e}")
            return []
    
    def _calculate_service_type_distribution(self, records: List[Dict]) -> Dict[str, int]:
        """Calculate service type distribution"""
        distribution = {}
        for record in records:
            service_type = record.get("service_type", "Unknown")
            distribution[service_type] = distribution.get(service_type, 0) + 1
        return distribution
    
    def _calculate_outcome_distribution(self, records: List[Dict]) -> Dict[str, int]:
        """Calculate service outcome distribution"""
        distribution = {}
        for record in records:
            outcome = record.get("service_outcome", "Unknown")
            distribution[outcome] = distribution.get(outcome, 0) + 1
        return distribution
    
    def _calculate_location_distribution(self, records: List[Dict]) -> Dict[str, int]:
        """Calculate location type distribution"""
        distribution = {}
        for record in records:
            location = record.get("location_type", "Unknown")
            distribution[location] = distribution.get(location, 0) + 1
        return distribution
    
    def _calculate_duration_stats(self, records: List[Dict]) -> Dict[str, float]:
        """Calculate duration statistics"""
        durations = [float(r.get("duration_hours", 0)) for r in records if r.get("duration_hours")]
        
        if not durations:
            return {"avg": 0, "min": 0, "max": 0, "median": 0}
        
        durations.sort()
        return {
            "avg": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "median": durations[len(durations) // 2]
        }
    
    def _calculate_narrative_stats(self, records: List[Dict]) -> Dict[str, float]:
        """Calculate narrative statistics"""
        narratives = [r.get("narrative_notes", "") for r in records if r.get("narrative_notes")]
        
        if not narratives:
            return {"avg_length": 0, "min_length": 0, "max_length": 0}
        
        lengths = [len(n) for n in narratives]
        return {
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths)
        }
    
    def _calculate_date_range(self, records: List[Dict]) -> Dict[str, str]:
        """Calculate date range of records"""
        dates = []
        for record in records:
            date_str = record.get("service_date")
            if date_str:
                try:
                    dates.append(datetime.fromisoformat(date_str).date())
                except:
                    continue
        
        if not dates:
            return {"start_date": None, "end_date": None, "span_days": 0}
        
        dates.sort()
        return {
            "start_date": dates[0].isoformat(),
            "end_date": dates[-1].isoformat(),
            "span_days": (dates[-1] - dates[0]).days
        }
    
    def _extract_quality_gate_results(self, validation_data: Dict) -> Dict[str, Any]:
        """Extract quality gate results from validation data"""
        gates = {}
        
        # Schema validation gate
        schema_validation = validation_data.get("schema_validation", {})
        gates["schema_validation"] = {
            "passed": schema_validation.get("passed_records", 0),
            "failed": schema_validation.get("failed_records", 0),
            "failure_rate": schema_validation.get("failure_rate", 0)
        }
        
        # Privacy gate
        privacy_analysis = validation_data.get("privacy_analysis", {})
        gates["privacy_risks"] = {
            "score": privacy_analysis.get("anonymization_score", 0),
            "risks_detected": len(privacy_analysis.get("potential_pii", [])),
            "failure_rate": 100 - privacy_analysis.get("anonymization_score", 0)
        }
        
        # Utility gate
        utility_analysis = validation_data.get("utility_analysis", {})
        gates["utility_quality"] = {
            "score": utility_analysis.get("realism_score", 0),
            "failure_rate": 100 - utility_analysis.get("realism_score", 0)
        }
        
        return gates
    
    def _store_metrics(self, output_metrics: Dict, validation_metrics: Dict, derived_metrics: Dict):
        """Store metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store main metrics
                conn.execute("""
                    INSERT INTO pipeline_metrics (
                        total_records, passed_records, failed_records, fallback_records,
                        overall_score, privacy_score, realism_score, throughput_per_minute
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    output_metrics.get("total_records", 0),
                    int(output_metrics.get("total_records", 0) * derived_metrics.get("pass_rate", 0) / 100),
                    len(validation_metrics.get("validation_errors", [])),
                    int(output_metrics.get("total_records", 0) * derived_metrics.get("fallback_ratio", 0) / 100),
                    validation_metrics.get("overall_score", 0),
                    validation_metrics.get("privacy_score", 0),
                    validation_metrics.get("realism_score", 0),
                    derived_metrics.get("throughput_per_minute", 0)
                ))
                
                conn.commit()
                logger.info("Metrics stored successfully")
                
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM pipeline_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """
                cursor = conn.execute(query)
                row = cursor.fetchone()
                
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to get latest metrics: {e}")
            return {}

if __name__ == "__main__":
    # Test data aggregation
    aggregator = DataAggregator()
    result = aggregator.aggregate_all_data()
    
    print("Data Aggregation Test Results:")
    print(f"Status: {result.get('status')}")
    print(f"Total Records: {result.get('output_metrics', {}).get('total_records', 0)}")
    print(f"Overall Score: {result.get('validation_metrics', {}).get('overall_score', 0)}")
