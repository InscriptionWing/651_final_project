# 🚀 Complete Guide: Data Generation & Real-time Dashboard Updates

## 📋 Overview
This guide will teach you how to use `main_english.py` to generate new English carer data and view real-time updates in the dashboard.

## 🎯 Quick Start (Recommended Methods)

### Method 1: One-Click Generate & Update (Simplest)

```bash
# Navigate to project directory
cd D:\651\poc\nsrc

# One-click generation of 30 records and dashboard update
python dashboard\generate_and_update.py --size 30
```

**This script will automatically:**
- ✅ Use `main_english.py` to generate high-quality English data
- ✅ Update dashboard database
- ✅ Display generated data statistics
- ✅ Check dashboard running status

### Method 2: Quick Demo Mode (Faster)

```bash
# Use demo mode to quickly generate 20 records
python dashboard\generate_and_update.py --demo --size 20
```

## 📖 Detailed Step-by-Step Instructions

### Step 1: Generate New Data

#### Option A: Using main_english.py (High Quality, Slower)
```bash
cd D:\651\poc\src
python main_english.py --size 25
```

**Parameter Explanations:**
- `--size`: Number of records to generate (default: 100)
- `--no-validate`: Skip data validation (optional)
- `--output-formats`: Output format selection (json, csv, jsonl)

#### Option B: Using Integration Script (One-Step Completion)
```bash
# Generate and integrate into dashboard
python dashboard\run_with_english_data.py --size 25

# If you only want to update dashboard (skip generation)
python dashboard\run_with_english_data.py --skip-generation
```

### Step 2: Update Dashboard Data

```bash
cd D:\651\poc\src\dashboard
python data_aggregator.py
```

### Step 3: Start/Refresh Dashboard

```bash
# If dashboard is not running
python dashboard\start_simple.py

# If already running, press F5 in browser to refresh
```

## 🎮 Common Command Combinations

### 1. Generate Small Batch Test Data
```bash
cd D:\651\poc\src
python main_english.py --size 10
python dashboard\data_aggregator.py
```

### 2. Generate Large Batch Production Data
```bash
cd D:\651\poc\src
python main_english.py --size 100 --output-formats json jsonl csv
python dashboard\data_aggregator.py
```

### 3. Validate Existing Data Only
```bash
cd D:\651\poc\src
python main_english.py --validate-file output\your_data_file.json
```

### 4. Quick Demo Workflow
```bash
cd D:\651\poc\src
python dashboard\generate_and_update.py --demo --size 15
```

## 🔄 Real-time Update Workflows

### Scenario 1: Regular New Data Generation
```bash
# Each time generating new data
cd D:\651\poc\src

# 1. Generate data
python main_english.py --size 50

# 2. Update dashboard
python dashboard\data_aggregator.py

# 3. Refresh browser at http://localhost:8501
```

### Scenario 2: Continuous Data Generation and Monitoring
```bash
# First run
python dashboard\generate_and_update.py --size 30

# Subsequent additions of more data
python dashboard\generate_and_update.py --size 20
python dashboard\generate_and_update.py --size 25

# After each run, dashboard will display all accumulated data
```

## 📊 Dashboard Access and Features

### Access URL
```
http://localhost:8501
```

### Main Feature Pages
1. **📈 Overview** - KPI Summary
2. **🔍 Quality Gates** - Quality Analysis
3. **📋 Records** - Record Browser
4. **🛠️ Diagnostics** - Single Record Diagnostics
5. **📝 Templates** - Template Viewer
6. **⚙️ System Status** - System Status and Export

### Real-time View of New Data
After generating new data, you will see in the dashboard:
- ✅ Updated total record count
- ✅ New KPI metrics
- ✅ Latest service records
- ✅ Updated quality analysis
- ✅ New data distribution charts

## 🚨 Troubleshooting

### Issue 1: ImportError
```bash
# Ensure running in correct directory
cd D:\651\poc\src
python main_english.py --size 10
```

### Issue 2: Dashboard Data Not Updating
```bash
# Manually run data aggregation
cd D:\651\poc\src\dashboard
python data_aggregator.py

# Then refresh browser page
```

### Issue 3: Dashboard Inaccessible
```bash
# Restart dashboard
cd D:\651\poc\src
python dashboard\start_simple.py
```

### Issue 4: Generation Timeout
```bash
# Use smaller batches
python main_english.py --size 10

# Or use demo mode
python dashboard\generate_and_update.py --demo --size 15
```

## 📁 Generated File Locations

### Data Files
```
D:\651\poc\new\output\
├── pure_llm_english_carers_YYYYMMDD_HHMMSS_Xrecords.json
├── pure_llm_english_carers_YYYYMMDD_HHMMSS_Xrecords.jsonl
└── pure_llm_english_carers_YYYYMMDD_HHMMSS_Xrecords.csv
```

### Dashboard Database
```
D:\651\poc\src\dashboard\metrics.db
```

### Log Files
```
D:\651\poc\src\logs\generator.log
```

## 🎉 Recommended Complete Workflow

```bash
# 1. Navigate to project directory
cd D:\651\poc\src

# 2. One-click generate and update (recommended)
python dashboard\generate_and_update.py --size 30

# 3. Open browser to access dashboard
# http://localhost:8501

# 4. Generate more data as needed
python dashboard\generate_and_update.py --size 20

# 5. View updated dashboard data
```

## 💡 Best Practices

1. **Batch Size Recommendations**:
   - Testing: 10-20 records
   - Daily use: 30-50 records
   - Large batch: 100+ records

2. **Performance Optimization**:
   - Use `--demo` flag for quick testing
   - Split large batch generation into multiple runs
   - Regularly clean up old output files

3. **Data Quality**:
   - Always keep data validation enabled
   - Regularly check quality metrics
   - Monitor abnormal patterns

4. **Dashboard Usage**:
   - Refresh browser after generating new data
   - Use filters to view specific data
   - Regularly export data backups

## 🔗 Related Files and Scripts

- `main_english.py` - Main data generation script
- `pure_llm_english_generator.py` - LLM generator
- `english_data_schema.py` - Data schema definitions
- `dashboard/data_aggregator.py` - Data aggregator
- `dashboard/streamlit_app.py` - Dashboard application
- `dashboard/generate_and_update.py` - One-click generate and update
- `dashboard/run_with_english_data.py` - Complete integration script

## 🆕 New Quick-Start Scripts

### Ultra-Simple One-Click Script
```bash
# Generate data and launch dashboard in one command
python dashboard\quick_generate_and_view.py --size 25

# Quick demo mode (faster)
python dashboard\quick_generate_and_view.py --demo --size 15

# Generate only (no dashboard launch)
python dashboard\quick_generate_and_view.py --size 30 --no-dashboard
```

### Windows Batch File (Double-click to run)
```
D:\651\poc\new\dashboard\快速生成数据.bat
```

**This batch file provides:**
- Interactive menu with 4 preset modes
- Quick demo mode (15 records, ~1 minute)
- Standard mode (30 records, ~3 minutes)
- Large batch mode (100 records, ~10 minutes)
- Custom quantity option

## 🔧 Advanced Usage

### Continuous Data Generation Pipeline
```bash
# Setup continuous generation (every hour)
while true; do
    python dashboard\generate_and_update.py --size 20
    sleep 3600  # Wait 1 hour
done
```

### Batch Processing with Validation
```bash
# Generate multiple datasets with different sizes
for size in 10 20 30 50; do
    echo "Generating $size records..."
    python main_english.py --size $size
    python dashboard\data_aggregator.py
    echo "Dashboard updated with $size new records"
done
```

### Quality Monitoring Workflow
```bash
# Generate data with enhanced validation
python main_english.py --size 50 --output-formats json csv jsonl
python dashboard\data_aggregator.py

# Check quality metrics
curl -s http://localhost:5000/api/metrics | python -m json.tool
```

## 📈 Dashboard Features Deep Dive

### Overview Tab
- **Pass Rate**: Percentage of records passing validation
- **Fallback Ratio**: Ratio of records using fallback templates
- **Throughput**: Records generated per minute
- **Average Narrative Length**: Mean length of narrative content
- **Field Coverage**: Percentage of fields populated

### Quality Gates Tab
- **Validation Results**: Pass/fail status for each record
- **Error Distribution**: Common validation error types
- **Quality Trends**: Historical quality metrics over time
- **Module Performance**: Quality breakdown by generation module

### Records Tab
- **Latest Records**: Most recently generated service records
- **Filter Options**: By date, service type, outcome, location
- **Quick View**: Expandable record details
- **Export Options**: JSON, CSV, summary reports

### Diagnostics Tab
- **Single Record Analysis**: Detailed validation messages
- **Re-validation Preview**: Test validation rules on specific records
- **Quality Score Breakdown**: Individual quality components
- **Template Usage**: Which templates were used for generation

### Templates Tab
- **Template Pool Overview**: Available narrative/action/factor templates
- **Coverage Analysis**: Usage distribution across categories
- **Duplicate Detection**: Near-duplicate template identification
- **Template Effectiveness**: Success rates by template type

### System Status Tab
- **System Health**: Overall pipeline status
- **Data Freshness**: Time since last data generation
- **Export Functions**: JSON, CSV, summary report generation
- **Performance Metrics**: Generation speed, error rates
- **Storage Information**: Database size, file counts

## 🛡️ Security and Privacy

### Data Safety Features
- **Synthetic Data Only**: All generated records are synthetic
- **PII Sanitization**: No real personal information included
- **Access Logging**: All dashboard access logged
- **Aggregated Metrics**: Only summary statistics shared
- **Local Storage**: All data stored locally, no external transmission

### Compliance Considerations
- **NDIS Guidelines**: Generated data follows NDIS service standards
- **Data Protection**: Synthetic data eliminates privacy concerns
- **Audit Trail**: Complete generation and validation history
- **Quality Assurance**: Multi-layer validation ensures data integrity

---

## 🎯 **You can now start generating data and viewing real-time results in the dashboard!**

### Quick Commands Reference Card
```bash
# Ultra-fast start
python dashboard\quick_generate_and_view.py --demo --size 15

# Production quality
python dashboard\generate_and_update.py --size 50

# Manual control
python main_english.py --size 30
python dashboard\data_aggregator.py
python dashboard\start_simple.py

# Validation only
python main_english.py --validate-file output\data.json
```

### Dashboard URL
```
🌐 http://localhost:8501
```

**Happy data generation and dashboard monitoring!** 🚀

