# Dashboard Setup Guide

## Quick Start

This guide will help you set up and run the NDIS Carer Data Pipeline Dashboard in under 10 minutes.

## Prerequisites

Before starting, ensure you have:

- Python 3.8 or higher installed
- The main NDIS carer data generation project set up
- At least 1GB of free disk space
- Internet connection for package installation

## Step-by-Step Setup

### Step 1: Prepare Environment

1. **Open Terminal/Command Prompt**
   ```bash
   # Navigate to the project directory
   cd /path/to/poc/new
   
   # Verify Python version
   python --version
   # Should show Python 3.8.x or higher
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   # Create virtual environment
   python -m venv dashboard_env
   
   # Activate virtual environment
   # On Windows:
   dashboard_env\Scripts\activate
   # On macOS/Linux:
   source dashboard_env/bin/activate
   ```

### Step 2: Install Dashboard Dependencies

```bash
# Navigate to dashboard directory
cd dashboard

# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, plotly, pandas; print('Dependencies installed successfully')"
```

### Step 3: Initialize Dashboard Database

```bash
# Create dashboard database and initial data
python data_aggregator.py
```

Expected output:
```
Data Aggregation Test Results:
Status: success
Total Records: [number]
Overall Score: [score]
```

### Step 4: Generate Initial Dashboard Data

```bash
# Run initial data aggregation
python -c "
from data_aggregator import DataAggregator
aggregator = DataAggregator()
result = aggregator.aggregate_all_data()
print(f'Aggregation completed: {result.get(\"status\", \"unknown\")}')
"
```

### Step 5: Launch Dashboard

```bash
# Start the Streamlit dashboard
streamlit run streamlit_app.py
```

The dashboard should automatically open in your browser at `http://localhost:8501`

## Verification

### Check Dashboard Functionality

1. **Overview KPIs**: Verify that metrics are displayed
2. **Quality Gates**: Check that validation data is shown
3. **Record Explorer**: Ensure records are listed
4. **Data Refresh**: Click the refresh button to update data

### Expected Initial State

- **Pass Rate**: Should show a percentage (may be 0% initially)
- **Total Records**: Should match your generated data count
- **Quality Gates**: Should display validation results
- **System Status**: Should show "healthy" or similar

## Configuration

### Basic Configuration

Edit `config.py` to customize dashboard settings:

```python
# Dashboard title and branding
DASHBOARD_CONFIG = {
    "title": "Your Custom Dashboard Title",
    "refresh_interval": 30,  # Auto-refresh interval in seconds
}

# Adjust KPI thresholds based on your requirements
KPI_THRESHOLDS = {
    "pass_rate": {
        "excellent": 95.0,  # Green threshold
        "good": 85.0,       # Blue threshold  
        "warning": 70.0,    # Yellow threshold
        "critical": 50.0    # Red threshold
    }
}
```

### Data Source Configuration

Ensure your data sources are correctly configured:

```python
# Verify paths in config.py
OUTPUT_DIR = PROJECT_ROOT / "output"     # Should point to your output directory
LOGS_DIR = PROJECT_ROOT / "logs"         # Should point to your logs directory
```

## Automated Setup (Optional)

### ETL Scheduler Setup

To automatically update dashboard data:

```bash
# Run ETL scheduler in background
python scheduler.py --daemon --interval 5

# Or run ETL job once every 10 minutes
python scheduler.py --interval 10
```

### Windows Service Setup

Create a batch file `start_dashboard.bat`:

```batch
@echo off
cd /d "C:\path\to\your\poc\new\dashboard"
call dashboard_env\Scripts\activate
streamlit run streamlit_app.py
pause
```

### Linux/macOS Service Setup

Create a systemd service file `/etc/systemd/system/ndis-dashboard.service`:

```ini
[Unit]
Description=NDIS Carer Data Dashboard
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/poc/new/dashboard
Environment=PATH=/path/to/dashboard_env/bin
ExecStart=/path/to/dashboard_env/bin/streamlit run streamlit_app.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl enable ndis-dashboard
sudo systemctl start ndis-dashboard
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Dashboard Won't Start

**Error**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
# Ensure virtual environment is activated
source dashboard_env/bin/activate  # or dashboard_env\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. No Data Displayed

**Error**: Dashboard shows "No data available"

**Solution**:
```bash
# Check if output files exist
ls -la ../output/

# Run data generation if no files exist
cd ..
python main.py --size 100

# Run data aggregation
cd dashboard
python data_aggregator.py
```

#### 3. Database Errors

**Error**: `sqlite3.OperationalError: no such table`

**Solution**:
```bash
# Delete and recreate database
rm -f data/metrics.db
python data_aggregator.py
```

#### 4. Port Already in Use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**:
```bash
# Use different port
streamlit run streamlit_app.py --server.port=8502

# Or kill existing process
# On Windows:
netstat -ano | findstr :8501
taskkill /PID [PID_NUMBER] /F

# On macOS/Linux:
lsof -ti:8501 | xargs kill -9
```

#### 5. Permission Errors

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# Check directory permissions
chmod 755 dashboard/
chmod 644 dashboard/*.py

# Ensure data directory is writable
mkdir -p dashboard/data
chmod 755 dashboard/data
```

### Performance Issues

#### Slow Dashboard Loading

1. **Reduce data scope**:
   ```python
   # In config.py, reduce max records
   DASHBOARD_CONFIG["max_records_display"] = 500
   ```

2. **Clear cache**:
   ```bash
   # Clear Streamlit cache
   rm -rf .streamlit/
   ```

3. **Optimize database**:
   ```bash
   # Compact database
   sqlite3 data/metrics.db "VACUUM;"
   ```

#### High Memory Usage

1. **Limit data retention**:
   ```sql
   -- Keep only last 30 days of data
   sqlite3 data/metrics.db "DELETE FROM pipeline_metrics WHERE timestamp < datetime('now', '-30 days');"
   ```

2. **Use pagination**:
   ```python
   # In config.py
   DASHBOARD_CONFIG["pagination_size"] = 25
   ```

## Validation

### Test Dashboard Features

Run this validation script:

```python
# validation_test.py
import requests
import streamlit as st

def test_dashboard():
    """Test dashboard functionality"""
    
    # Test 1: Data aggregation
    from data_aggregator import DataAggregator
    aggregator = DataAggregator()
    result = aggregator.aggregate_all_data()
    assert result.get("status") == "success", "Data aggregation failed"
    
    # Test 2: Database connectivity
    import sqlite3
    conn = sqlite3.connect("data/metrics.db")
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert len(tables) > 0, "No database tables found"
    conn.close()
    
    # Test 3: Configuration loading
    from config import get_dashboard_config
    config = get_dashboard_config()
    assert "dashboard" in config, "Configuration not loaded properly"
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_dashboard()
```

Run validation:
```bash
python validation_test.py
```

### Performance Benchmarks

Expected performance on standard hardware:

- **Dashboard Load Time**: < 5 seconds
- **Data Refresh**: < 3 seconds  
- **Record Search**: < 2 seconds
- **Export Generation**: < 10 seconds (for 1000 records)

## Security Considerations

### Access Control

1. **Network Security**:
   ```bash
   # Bind to localhost only (default)
   streamlit run streamlit_app.py --server.address=127.0.0.1
   
   # Or bind to specific interface
   streamlit run streamlit_app.py --server.address=192.168.1.100
   ```

2. **Authentication** (if needed):
   ```python
   # Add to streamlit_app.py
   import streamlit_authenticator as stauth
   
   # Configure authentication
   authenticator = stauth.Authenticate(
       credentials,
       'dashboard_auth',
       'auth_key',
       cookie_expiry_days=30
   )
   ```

### Data Privacy

1. **Synthetic Data Verification**:
   ```python
   # Verify no real PII in dashboard
   from data_aggregator import DataAggregator
   aggregator = DataAggregator()
   # Check privacy scores in validation results
   ```

2. **Audit Logging**:
   ```python
   # Enable access logging in config.py
   AUDIT_CONFIG = {
       "enabled": True,
       "log_file": "logs/dashboard_access.log"
   }
   ```

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   ```bash
   # Clean old logs
   find logs/ -name "*.log" -mtime +7 -delete
   
   # Compact database
   sqlite3 data/metrics.db "VACUUM;"
   ```

2. **Monthly**:
   ```bash
   # Archive old data
   sqlite3 data/metrics.db "DELETE FROM pipeline_metrics WHERE timestamp < datetime('now', '-90 days');"
   
   # Update dependencies
   pip install -r requirements.txt --upgrade
   ```

3. **Quarterly**:
   ```bash
   # Full system health check
   python validation_test.py
   
   # Review and update thresholds
   # Edit config.py KPI_THRESHOLDS
   ```

### Backup Procedures

```bash
# Create backup
cp data/metrics.db data/metrics_backup_$(date +%Y%m%d).db

# Automated backup script
#!/bin/bash
BACKUP_DIR="backups"
mkdir -p $BACKUP_DIR
cp data/metrics.db "$BACKUP_DIR/metrics_$(date +%Y%m%d_%H%M%S).db"
find $BACKUP_DIR -name "metrics_*.db" -mtime +30 -delete
```

## Next Steps

After successful setup:

1. **Customize Thresholds**: Adjust KPI thresholds based on your data patterns
2. **Set Up Alerts**: Configure alert thresholds for critical metrics
3. **Schedule ETL**: Set up automated data aggregation
4. **User Training**: Train team members on dashboard usage
5. **Documentation**: Create organization-specific usage guidelines

## Support

If you encounter issues not covered in this guide:

1. Check the main project documentation
2. Review log files in `logs/` directory
3. Verify configuration in `config.py`
4. Run the validation test script
5. Contact the development team

---

**Setup Complete!** 🎉

Your NDIS Carer Data Pipeline Dashboard should now be running and ready for monitoring your data quality and pipeline health.



