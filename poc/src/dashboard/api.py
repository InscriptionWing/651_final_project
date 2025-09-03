"""
Dashboard API
Simple REST API for dashboard data access
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from config import get_dashboard_config, DASHBOARD_DATA_DIR
from data_aggregator import DataAggregator

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize components
config = get_dashboard_config()
aggregator = DataAggregator()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": config["dashboard"]["version"]
    })

@app.route('/api/overview', methods=['GET'])
def get_overview():
    """Get dashboard overview with KPIs"""
    try:
        # Run data aggregation
        metrics = aggregator.aggregate_all_data()
        
        if metrics.get("status") != "success":
            return jsonify({"error": "Failed to aggregate data"}), 500
        
        # Extract KPIs
        derived_metrics = metrics.get("derived_metrics", {})
        output_metrics = metrics.get("output_metrics", {})
        validation_metrics = metrics.get("validation_metrics", {})
        
        overview = {
            "kpis": {
                "pass_rate": {
                    "value": derived_metrics.get("pass_rate", 0),
                    "status": _get_kpi_status("pass_rate", derived_metrics.get("pass_rate", 0)),
                    "trend": derived_metrics.get("trend_analysis", {}).get("pass_rate_trend", "stable")
                },
                "fallback_ratio": {
                    "value": derived_metrics.get("fallback_ratio", 0),
                    "status": _get_kpi_status("fallback_ratio", derived_metrics.get("fallback_ratio", 0)),
                    "trend": "stable"  # Could be enhanced with historical data
                },
                "throughput": {
                    "value": derived_metrics.get("throughput_per_minute", 0),
                    "status": _get_kpi_status("throughput", derived_metrics.get("throughput_per_minute", 0)),
                    "trend": derived_metrics.get("trend_analysis", {}).get("throughput_trend", "stable")
                },
                "avg_narrative_length": {
                    "value": derived_metrics.get("avg_narrative_length", 0),
                    "status": _get_narrative_length_status(derived_metrics.get("avg_narrative_length", 0)),
                    "trend": "stable"
                },
                "field_coverage": {
                    "value": derived_metrics.get("field_coverage", 0),
                    "status": _get_kpi_status("field_coverage", derived_metrics.get("field_coverage", 0)),
                    "trend": "stable"
                }
            },
            "summary": {
                "total_records": output_metrics.get("total_records", 0),
                "unique_carers": output_metrics.get("unique_carers", 0),
                "unique_participants": output_metrics.get("unique_participants", 0),
                "overall_quality_status": derived_metrics.get("quality_status", "unknown"),
                "last_updated": metrics.get("aggregation_timestamp"),
                "data_span_days": output_metrics.get("date_range", {}).get("span_days", 0)
            }
        }
        
        return jsonify(overview)
        
    except Exception as e:
        logger.error(f"Failed to get overview: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/quality-gates', methods=['GET'])
def get_quality_gates():
    """Get quality gate analysis"""
    try:
        metrics = aggregator.aggregate_all_data()
        validation_metrics = metrics.get("validation_metrics", {})
        quality_gates = validation_metrics.get("quality_gates", {})
        
        # Format quality gates for dashboard
        gates_data = []
        for gate_name, gate_data in quality_gates.items():
            gates_data.append({
                "name": gate_name.replace("_", " ").title(),
                "failure_rate": gate_data.get("failure_rate", 0),
                "failed_records": gate_data.get("failed", 0),
                "total_records": gate_data.get("passed", 0) + gate_data.get("failed", 0),
                "status": _get_quality_gate_status(gate_data.get("failure_rate", 0)),
                "details": gate_data
            })
        
        return jsonify({
            "quality_gates": gates_data,
            "overall_score": validation_metrics.get("overall_score", 0),
            "recommendations": validation_metrics.get("recommendations", [])
        })
        
    except Exception as e:
        logger.error(f"Failed to get quality gates: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/records', methods=['GET'])
def get_records():
    """Get records with pagination and filtering"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        status_filter = request.args.get('status')
        source_filter = request.args.get('source')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        search_query = request.args.get('search')
        
        # Build SQL query
        query = "SELECT * FROM record_details WHERE 1=1"
        params = []
        
        if status_filter:
            query += " AND validation_status = ?"
            params.append(status_filter)
        
        if source_filter:
            query += " AND source_type = ?"
            params.append(source_filter)
        
        if date_from:
            query += " AND service_date >= ?"
            params.append(date_from)
        
        if date_to:
            query += " AND service_date <= ?"
            params.append(date_to)
        
        if search_query:
            query += " AND (record_id LIKE ? OR carer_id LIKE ? OR participant_id LIKE ?)"
            search_param = f"%{search_query}%"
            params.extend([search_param, search_param, search_param])
        
        # Add ordering and pagination
        query += " ORDER BY created_timestamp DESC"
        
        # Count total records
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        
        with sqlite3.connect(aggregator.db_path) as conn:
            # Get total count
            total_records = conn.execute(count_query, params).fetchone()[0]
            
            # Get paginated results
            query += " LIMIT ? OFFSET ?"
            params.extend([per_page, (page - 1) * per_page])
            
            cursor = conn.execute(query, params)
            columns = [description[0] for description in cursor.description]
            records = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify({
            "records": records,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_records": total_records,
                "total_pages": (total_records + per_page - 1) // per_page
            },
            "filters_applied": {
                "status": status_filter,
                "source": source_filter,
                "date_from": date_from,
                "date_to": date_to,
                "search": search_query
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get records: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/record/<record_id>', methods=['GET'])
def get_record_details(record_id: str):
    """Get detailed information about a specific record"""
    try:
        with sqlite3.connect(aggregator.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM record_details WHERE record_id = ?", 
                (record_id,)
            )
            columns = [description[0] for description in cursor.description]
            record = cursor.fetchone()
            
            if not record:
                return jsonify({"error": "Record not found"}), 404
            
            record_data = dict(zip(columns, record))
            
            # Parse validation errors if present
            if record_data.get("validation_errors"):
                try:
                    record_data["validation_errors"] = json.loads(record_data["validation_errors"])
                except:
                    pass
            
            return jsonify({"record": record_data})
        
    except Exception as e:
        logger.error(f"Failed to get record details: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/distributions', methods=['GET'])
def get_distributions():
    """Get data distributions for charts"""
    try:
        metrics = aggregator.aggregate_all_data()
        output_metrics = metrics.get("output_metrics", {})
        
        distributions = {
            "service_types": output_metrics.get("service_type_distribution", {}),
            "outcomes": output_metrics.get("outcome_distribution", {}),
            "locations": output_metrics.get("location_distribution", {}),
            "duration_stats": output_metrics.get("duration_stats", {}),
            "narrative_stats": output_metrics.get("narrative_stats", {})
        }
        
        return jsonify(distributions)
        
    except Exception as e:
        logger.error(f"Failed to get distributions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/trends', methods=['GET'])
def get_trends():
    """Get trend data for time series charts"""
    try:
        # Get time range from query parameters
        hours = int(request.args.get('hours', 24))
        
        with sqlite3.connect(aggregator.db_path) as conn:
            query = """
                SELECT 
                    timestamp,
                    total_records,
                    overall_score,
                    throughput_per_minute,
                    privacy_score,
                    realism_score
                FROM pipeline_metrics 
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp ASC
            """.format(hours)
            
            cursor = conn.execute(query)
            columns = [description[0] for description in cursor.description]
            trends = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        return jsonify({
            "trends": trends,
            "time_range_hours": hours
        })
        
    except Exception as e:
        logger.error(f"Failed to get trends: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/templates', methods=['GET'])
def get_template_usage():
    """Get template usage statistics"""
    try:
        # This would be enhanced with actual template tracking
        # For now, return mock data based on service type distribution
        metrics = aggregator.aggregate_all_data()
        service_distribution = metrics.get("output_metrics", {}).get("service_type_distribution", {})
        
        template_data = {
            "narrative_templates": {
                "total_templates": 50,
                "categories": list(service_distribution.keys()),
                "usage_by_category": service_distribution,
                "diversity_score": 85.5
            },
            "action_templates": {
                "total_templates": 30,
                "usage_count": sum(service_distribution.values()),
                "diversity_score": 78.2
            },
            "factor_templates": {
                "total_templates": 25,
                "near_duplicate_rate": 5.2
            }
        }
        
        return jsonify(template_data)
        
    except Exception as e:
        logger.error(f"Failed to get template usage: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export_data():
    """Export data in requested format"""
    try:
        data = request.get_json()
        export_format = data.get('format', 'json')
        filters = data.get('filters', {})
        
        # Get filtered records (reuse logic from get_records)
        # This is a simplified implementation
        with sqlite3.connect(aggregator.db_path) as conn:
            query = "SELECT * FROM record_details LIMIT 1000"  # Limit for safety
            cursor = conn.execute(query)
            columns = [description[0] for description in cursor.description]
            records = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        if export_format.lower() == 'csv':
            # Convert to CSV format
            import pandas as pd
            df = pd.DataFrame(records)
            csv_data = df.to_csv(index=False)
            
            return jsonify({
                "status": "success",
                "format": "csv",
                "data": csv_data,
                "record_count": len(records)
            })
        else:
            # Return as JSON
            return jsonify({
                "status": "success",
                "format": "json",
                "data": records,
                "record_count": len(records)
            })
        
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    """Manually trigger data refresh"""
    try:
        result = aggregator.aggregate_all_data()
        return jsonify({
            "status": "success",
            "message": "Data refresh completed",
            "timestamp": datetime.now().isoformat(),
            "metrics_updated": result.get("status") == "success"
        })
        
    except Exception as e:
        logger.error(f"Failed to refresh data: {e}")
        return jsonify({"error": str(e)}), 500

# Helper functions
def _get_kpi_status(kpi_name: str, value: float) -> str:
    """Get KPI status based on thresholds"""
    thresholds = config["kpi_thresholds"].get(kpi_name, {})
    
    if kpi_name == "fallback_ratio":
        # Lower is better for fallback ratio
        if value <= thresholds.get("excellent", 0):
            return "excellent"
        elif value <= thresholds.get("good", 0):
            return "good"
        elif value <= thresholds.get("warning", 0):
            return "warning"
        else:
            return "critical"
    else:
        # Higher is better for other KPIs
        if value >= thresholds.get("excellent", 0):
            return "excellent"
        elif value >= thresholds.get("good", 0):
            return "good"
        elif value >= thresholds.get("warning", 0):
            return "warning"
        else:
            return "critical"

def _get_narrative_length_status(length: float) -> str:
    """Get status for narrative length"""
    thresholds = config["kpi_thresholds"]["avg_narrative_length"]
    
    if (length >= thresholds["optimal_min"] and 
        length <= thresholds["optimal_max"]):
        return "excellent"
    elif (length >= thresholds["min_acceptable"] and 
          length <= thresholds["max_acceptable"]):
        return "good"
    else:
        return "warning"

def _get_quality_gate_status(failure_rate: float) -> str:
    """Get quality gate status based on failure rate"""
    if failure_rate <= 5:
        return "excellent"
    elif failure_rate <= 10:
        return "good"
    elif failure_rate <= 20:
        return "warning"
    else:
        return "critical"

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create database if it doesn't exist
    aggregator._init_database()
    
    # Run development server
    app.run(host='0.0.0.0', port=5000, debug=True)



