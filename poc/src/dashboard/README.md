# NDIS Carer Data Pipeline Dashboard

## Overview

The NDIS Carer Data Pipeline Dashboard provides a comprehensive, real-time view of data quality and pipeline health for the synthetic carer service record generation system. This dashboard enables quick, reliable monitoring of data generation processes, validation results, and system performance.

## Purpose

The dashboard serves as a centralized monitoring solution that:

- **Monitors Data Quality**: Tracks pass rates, validation results, and quality gates in real-time
- **Ensures Pipeline Health**: Provides visibility into generation throughput, error rates, and system status  
- **Supports Decision Making**: Offers actionable insights through KPIs, trends, and detailed analytics
- **Enables Quality Assurance**: Facilitates record-level inspection and validation review

## Architecture

### Data Flow

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Output Files    │    │              │    │                 │    │              │
│ (JSON/CSV/JSONL)│───▶│ ETL Job      │───▶│ Metrics Store   │───▶│ Dashboard    │
│                 │    │ (Aggregator) │    │ (SQLite)        │    │ (Streamlit)  │
└─────────────────┘    │              │    │                 │    │              │
┌─────────────────┐    │              │    │                 │    │              │
│ Validation      │───▶│              │    │                 │    │              │
│ Reports         │    │              │    │                 │    │              │
└─────────────────┘    │              │    │                 │    │              │
┌─────────────────┐    │              │    │                 │    │              │
│ Log Files       │───▶│              │    │                 │    │              │
│                 │    │              │    │                 │    │              │
└─────────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘
```

The ETL job aggregates outputs (JSONL/CSV), logs, and metrics into a metrics store, which the dashboard reads via a simple API.

## Features

### 📊 Overview KPIs

Top-level metrics provide instant insight into system health:

- **Pass Rate**: Percentage of records that passed validation (Target: >85%)
- **Fallback Ratio**: Percentage of records using fallback templates (Target: <10%)
- **Throughput**: Records generated per minute (Target: >50/min)
- **Avg Narrative Length**: Average character count of narrative notes (Target: 100-300 chars)
- **Field Coverage**: Percentage of fields with complete data (Target: >85%)

### 🚦 Quality Gate View

Visual analysis of validation results:

- **Failure Reason Analysis**: Bar charts showing top failure categories
- **Trend Analysis**: Time series showing quality trends over time
- **Module Distribution**: Breakdown by validation module (parsing, rules, duplicates)
- **Gate Status**: Real-time status of all quality gates

### 📋 Record Explorer

Detailed record inspection capabilities:

- **Searchable Table**: Browse latest records with status, source, and metadata
- **Advanced Filtering**: Filter by date range, status, location, model ID, and source type
- **Record Details**: Drill-down view for individual record analysis
- **Validation Messages**: View detailed validation results for each record

### 🎯 Data Distributions

Statistical analysis and visualization:

- **Service Type Distribution**: Pie chart of service categories
- **Outcome Distribution**: Bar chart of service results
- **Location Analysis**: Geographic distribution of services
- **Duration Statistics**: Statistical summary of service durations

### 📝 Templates Viewer

Template pool monitoring:

- **Template Coverage**: Shows narrative/action/factor pool sizes
- **Category Coverage**: Usage statistics by service category
- **Diversity Metrics**: Near-duplicate rate and diversity scores
- **Usage Patterns**: Template utilization analysis

### 🔍 Filtering & Search

Comprehensive data exploration:

- **Date Range Filters**: Custom time period selection
- **Status Filters**: Filter by validation status (passed/failed/warning)
- **Source Filters**: Filter by generation source (LLM/template)
- **Text Search**: Search by record ID, carer ID, or keywords
- **Export Filters**: Apply filters to data exports

### 📤 Export & Reports

Data export and reporting capabilities:

- **Multiple Formats**: Export as JSON, CSV, or Excel
- **Filtered Exports**: Export current filtered view
- **Automated Reports**: Daily summary reports with key metrics
- **Custom Reports**: Generate reports for specific time periods or criteria

### ⚠️ Alerts (Optional)

Threshold-based monitoring:

- **Pass Rate Alerts**: Trigger when pass rate drops suddenly
- **Failure Spike Alerts**: Alert on significant increases in failures
- **Throughput Alerts**: Monitor for performance degradation
- **Custom Thresholds**: Configurable alert parameters

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see requirements.txt)
- Access to the NDIS carer data generation project

### Installation Steps

1. **Navigate to Dashboard Directory**
   ```bash
   cd /path/to/poc/new/dashboard
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Database**
   ```bash
   python data_aggregator.py
   ```

4. **Run Initial Data Aggregation**
   ```bash
   python -c "from data_aggregator import DataAggregator; DataAggregator().aggregate_all_data()"
   ```

### Running the Dashboard

#### Option 1: Streamlit Dashboard (Recommended)

```bash
streamlit run streamlit_app.py
```

The dashboard will be available at `http://localhost:8501`

#### Option 2: Flask API + Custom Frontend

```bash
python api.py
```

API will be available at `http://localhost:5000`

#### Option 3: Background ETL Scheduler

```bash
# Run ETL job once
python scheduler.py --run-once

# Run ETL scheduler with 5-minute intervals
python scheduler.py --interval 5

# Run as daemon
python scheduler.py --daemon --interval 10
```

## Configuration

### Dashboard Settings

Edit `config.py` to customize:

```python
# KPI thresholds
KPI_THRESHOLDS = {
    "pass_rate": {
        "excellent": 95.0,
        "good": 85.0,
        "warning": 70.0,
        "critical": 50.0
    },
    # ... other thresholds
}

# Dashboard appearance
DASHBOARD_CONFIG = {
    "title": "NDIS Carer Data Pipeline Dashboard",
    "refresh_interval": 30,  # seconds
    "max_records_display": 1000,
}
```

### Quality Gates

Configure validation thresholds:

```python
QUALITY_GATES = {
    "parsing_errors": {
        "threshold": 5.0  # percentage
    },
    "validation_failures": {
        "threshold": 10.0
    },
    "privacy_risks": {
        "threshold": 1.0
    }
}
```

## Usage Guide

### Daily Monitoring Workflow

1. **Check Overview KPIs**: Review top-level health indicators
2. **Examine Quality Gates**: Identify any failing validation rules
3. **Review Recent Records**: Inspect latest generated records
4. **Analyze Trends**: Look for patterns in failure rates or performance
5. **Export Reports**: Generate summaries for stakeholders

### Troubleshooting Common Issues

#### Low Pass Rate
- Check Quality Gates view for specific failure reasons
- Review recent error logs in System Status
- Examine failing records in Record Explorer

#### High Fallback Ratio
- Review template usage in Templates Viewer
- Check LLM service availability in logs
- Verify API configuration and quotas

#### Poor Throughput
- Monitor system resources and performance metrics
- Check for network or API latency issues
- Review ETL job scheduling and frequency

### Best Practices

1. **Regular Monitoring**: Check dashboard at least daily during active generation
2. **Threshold Tuning**: Adjust KPI thresholds based on actual performance patterns
3. **Data Retention**: Archive old metrics data to maintain performance
4. **Alert Configuration**: Set up alerts for critical threshold breaches
5. **Documentation**: Keep notes on recurring issues and resolutions

## API Reference

### REST Endpoints

- `GET /api/health` - Health check
- `GET /api/overview` - Dashboard overview with KPIs
- `GET /api/quality-gates` - Quality gate analysis
- `GET /api/records` - Paginated record listing with filters
- `GET /api/record/{id}` - Individual record details
- `GET /api/distributions` - Data distribution statistics
- `GET /api/trends` - Time series trend data
- `GET /api/templates` - Template usage statistics
- `POST /api/export` - Data export
- `POST /api/refresh` - Manual data refresh

### Query Parameters

#### Records Endpoint
```
GET /api/records?page=1&per_page=50&status=passed&source=llm&date_from=2024-01-01&search=SR12345
```

#### Trends Endpoint
```
GET /api/trends?hours=24
```

## Privacy & Safety

### Data Protection Measures

- **Synthetic Data Only**: All displayed data is artificially generated
- **PII Sanitization**: Automatic detection and flagging of potential PII
- **Access Logging**: All dashboard access is logged for audit trails
- **Aggregated Metrics**: Only statistical summaries shared outside team

### Compliance Features

- **NDIS Privacy Standards**: Adheres to NDIS privacy protection requirements
- **Data Retention**: Configurable data retention policies
- **Audit Trail**: Complete logging of data access and modifications
- **Secure Storage**: Encrypted storage of sensitive configuration data

## Technical Architecture

### Tech Stack

- **Frontend**: Streamlit for interactive dashboard UI
- **Backend**: Flask for REST API services
- **Database**: SQLite for metrics storage (easily upgradeable to PostgreSQL)
- **Visualization**: Plotly for interactive charts and graphs
- **Scheduling**: Python schedule library for ETL automation
- **Data Processing**: Pandas for data manipulation and analysis

### Database Schema

```sql
-- Main metrics table
CREATE TABLE pipeline_metrics (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_records INTEGER,
    passed_records INTEGER,
    failed_records INTEGER,
    overall_score REAL,
    throughput_per_minute REAL
);

-- Record details table
CREATE TABLE record_details (
    id INTEGER PRIMARY KEY,
    record_id TEXT UNIQUE,
    service_type TEXT,
    validation_status TEXT,
    created_timestamp DATETIME
);

-- Quality gates table
CREATE TABLE quality_gate_results (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    gate_name TEXT,
    failure_rate REAL
);
```

### Performance Considerations

- **Data Caching**: 1-minute cache for dashboard data
- **Pagination**: Large datasets paginated for performance
- **Background Processing**: ETL runs in background to avoid UI blocking
- **Resource Limits**: Configurable limits on data export sizes

## Troubleshooting

### Common Issues

#### Dashboard Won't Start
```bash
# Check Python version
python --version  # Should be 3.8+

# Install missing dependencies
pip install -r requirements.txt

# Check for port conflicts
netstat -an | grep :8501
```

#### No Data Displayed
```bash
# Run data aggregation manually
python data_aggregator.py

# Check output directory
ls -la ../output/

# Verify database creation
sqlite3 data/metrics.db ".tables"
```

#### Performance Issues
```bash
# Check database size
ls -lh data/metrics.db

# Clear old data
sqlite3 data/metrics.db "DELETE FROM pipeline_metrics WHERE timestamp < datetime('now', '-30 days');"

# Restart with clean cache
rm -rf .streamlit/
streamlit run streamlit_app.py
```

### Getting Help

For technical support or feature requests:

1. Check the logs in `logs/` directory
2. Review configuration in `config.py`
3. Consult the project documentation
4. Contact the development team

## Future Enhancements

### Planned Features

- **Real-time Updates**: WebSocket-based live data updates
- **Advanced Analytics**: Machine learning-based anomaly detection
- **Multi-tenant Support**: Support for multiple pipeline instances
- **Custom Dashboards**: User-configurable dashboard layouts
- **Integration APIs**: Webhooks for external system integration

### Scalability Roadmap

- **Database Migration**: PostgreSQL support for larger deployments
- **Distributed Processing**: Support for distributed ETL processing
- **High Availability**: Load balancing and failover capabilities
- **Enterprise Features**: SSO, RBAC, and audit compliance

---

## License

This dashboard is part of the NDIS Carer Data Generation project and follows the same licensing terms.

## Version History

- **v1.0.0** - Initial release with core dashboard functionality
- **v1.1.0** - Added export and reporting features
- **v1.2.0** - Enhanced filtering and search capabilities

---

*For questions or support, please refer to the main project documentation or contact the development team.*



