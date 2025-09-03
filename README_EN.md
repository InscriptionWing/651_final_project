# NDIS Carers Data Generator

## 📋 Project Overview

A synthetic data generator designed specifically for the Australian National Disability Insurance Scheme (NDIS) carer services. This project uses local LLM technology to generate high-quality, privacy-compliant English carer service records for safe data analysis, system testing, and business analytics without using real participant sensitive data.

## 🎯 Core Features

### ✅ Implemented Features
- **🤖 Pure LLM Data Generation**: Complete carer service record generation using Ollama local LLM
- **🌍 English Output**: All generated content in English, meeting international standards
- **👤 Complete Carer Information**: Includes carer names, IDs, and detailed service records
- **📊 Multi-format Export**: Supports JSON, JSONL, CSV formats
- **✅ Data Validation**: Built-in data quality validation and statistical analysis
- **🔒 Privacy Protection**: Completely synthetic data with no real personal information

### 📈 Generated Data Includes
- Carer basic information (name, ID)
- Participant information (de-identified)
- Service types and duration
- Detailed service narratives
- Support techniques and challenge records
- Participant responses and follow-up requirements
- NDIS billing codes and supervision notes

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to project directory
cd D:\651\poc\new

# Install dependencies
pip install -r requirements.txt

# Setup Ollama (if not already installed)
# Visit https://ollama.ai to download and install Ollama
# Run a suitable model, for example:
ollama pull gpt-oss:20b
```

### 2. Generate Data

```bash
# Generate 20 English carer records
python main_english.py --size 20

# Generate 100 records and skip validation (faster)
python main_english.py --size 100 --no-validate

# Validate existing data file
python main_english.py --validate-file output/your_data_file.json
```

### 3. View Results

Generated data will be saved in the `output/` directory in three formats:
- `*.json` - Standard JSON format
- `*.jsonl` - One JSON record per line
- `*.csv` - CSV table format

## 📁 Project Structure

```
D:\651\poc\new\
├── README.md                          # Project documentation
├── README_EN.md                       # English documentation
├── requirements.txt                   # Dependency list
├── requirements-minimal.txt           # Minimal dependencies
├── main_english.py                   # Main program entry
├── pure_llm_english_generator.py     # Pure LLM English generator
├── english_data_schema.py            # English data structure definitions
├── config.py                         # Project configuration
├── output/                           # Output directory
├── logs/                             # Log directory
├── dashboard/                        # Data dashboard (optional)
└── *.py                              # Other support modules
```

## 🔧 Core Components

### Data Generators
- **PureLLMEnglishGenerator**: Pure LLM-driven English data generator
- **EnglishTemplateGenerator**: Template-driven English data generator (alternative)

### Data Validation
- **EnglishDataValidator**: English data quality validator
- Supports statistical analysis, distribution checks, integrity validation

### Data Structures
- **CarerServiceRecord**: Main carer service record structure
- **CarerProfile**: Carer profile information
- **ParticipantProfile**: Participant profile information

## 📊 Generation Example

```json
{
  "record_id": "SR72682989",
  "carer_id": "CR191161", 
  "carer_name": "Joshua Walker",
  "participant_id": "PT791798",
  "service_date": "2025-07-19",
  "service_type": "Household Tasks",
  "duration_hours": 1.27,
  "narrative_notes": "Joshua Walker provided household tasks support...",
  "location_type": "Healthcare Facility",
  "service_outcome": "neutral",
  "support_techniques_used": ["Visual cues", "Task sequencing"],
  "challenges_encountered": [],
  "participant_response": "Positive engagement",
  "follow_up_required": false,
  "billing_code": "NDIS_HOUSEHOLD_TASKS_2612"
}
```

## ⚙️ Configuration Options

### Command Line Arguments
- `--size`: Number of records to generate (default: 100)
- `--no-validate`: Skip data validation for faster generation
- `--validate-file`: Validate existing data file
- `--output-formats`: Specify output formats (json, jsonl, csv)

### Environment Configuration
- Ensure Ollama service is running on `localhost:11434`
- Project automatically detects available LLM models

## 📈 Performance Metrics

### Typical Generation Performance
- **Generation Speed**: ~2-5 seconds/record (depends on LLM performance)
- **Success Rate**: >95% (with stable network)
- **Data Quality**: Average narrative length 730 characters, rich and realistic content

### Recommended Configuration
- **Small batch testing**: 10-20 records
- **Regular use**: 50-100 records
- **Large batch generation**: 500+ records (recommend batch processing)

## 🔍 Data Validation

Generated data undergoes automatic validation:
- ✅ Field completeness check
- ✅ Data type validation
- ✅ Narrative content length validation
- ✅ Service type distribution analysis
- ✅ Time range reasonableness check
- ✅ Carer-participant allocation statistics

## 🎨 Optional Features

### Data Dashboard
Project includes an optional Streamlit data dashboard:

```bash
cd dashboard
python run_dashboard.py --mode simple
```

### Data Aggregation Analysis
```bash
python dashboard/data_aggregator.py
```

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check Ollama service status
   curl http://localhost:11434/api/tags
   
   # Restart Ollama service
   ollama serve
   ```

2. **Generation Timeout**
   - Increase timeout settings
   - Check network connection
   - Use smaller batch sizes

3. **Out of Memory**
   - Reduce generation batch size
   - Close unnecessary applications

## 📝 Development Guide

### Adding New Service Types
1. Edit `ServiceType` enum in `english_data_schema.py`
2. Update LLM prompt templates in `pure_llm_english_generator.py`
3. Test new service type generation

### Custom Validation Rules
1. Modify `EnglishDataValidator` class
2. Add new validation methods
3. Update output formats

## 📄 License

This project is for educational and research purposes only.

## 🤝 Contributing

Welcome to submit issue reports and feature suggestions!

## 📞 Support

For technical support, please check:
1. Project documentation and examples
2. FAQ section
3. Error log analysis

---

**⚠️ Important Notice**: 
- Generated data is synthetic only and should not be used in production
- Please ensure compliance with local data protection regulations
- Regularly update LLM models to maintain data quality
