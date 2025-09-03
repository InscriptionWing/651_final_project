#!/usr/bin/env python3
"""
Dashboard Launcher
Convenient script to launch the NDIS Carer Data Pipeline Dashboard
"""

import sys
import subprocess
import os
import time
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        logger.error(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'plotly', 'pandas', 'flask']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    return True

def initialize_database():
    """Initialize dashboard database"""
    try:
        logger.info("Initializing dashboard database...")
        from data_aggregator import DataAggregator
        
        aggregator = DataAggregator()
        result = aggregator.aggregate_all_data()
        
        if result.get("status") == "success":
            logger.info("Database initialized successfully")
            total_records = result.get("output_metrics", {}).get("total_records", 0)
            logger.info(f"Loaded {total_records} records")
            return True
        else:
            logger.error(f"Database initialization failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def run_streamlit_dashboard(port=8501, host="localhost"):
    """Run the Streamlit dashboard"""
    try:
        logger.info(f"Starting Streamlit dashboard on http://{host}:{port}")
        
        # Ensure we're in the correct directory
        dashboard_dir = Path(__file__).parent
        os.chdir(dashboard_dir)
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        # Run streamlit
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(3)
        
        if process.poll() is None:
            logger.info("Dashboard started successfully!")
            logger.info(f"Open your browser to: http://{host}:{port}")
            logger.info("Press Ctrl+C to stop the dashboard")
            
            try:
                process.wait()
            except KeyboardInterrupt:
                logger.info("Shutting down dashboard...")
                process.terminate()
                process.wait()
        else:
            stdout, stderr = process.communicate()
            logger.error("Failed to start dashboard")
            if stderr:
                logger.error(f"Error: {stderr.decode()}")
            
    except Exception as e:
        logger.error(f"Failed to run dashboard: {e}")

def run_flask_api(port=5000, host="localhost"):
    """Run the Flask API server"""
    try:
        logger.info(f"Starting Flask API server on http://{host}:{port}")
        
        # Set environment variables
        os.environ['FLASK_ENV'] = 'development'
        os.environ['FLASK_DEBUG'] = '1'
        
        cmd = [sys.executable, "api.py"]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)
        
        if process.poll() is None:
            logger.info("API server started successfully!")
            logger.info(f"API available at: http://{host}:{port}/api/health")
            
            try:
                process.wait()
            except KeyboardInterrupt:
                logger.info("Shutting down API server...")
                process.terminate()
                process.wait()
        else:
            stdout, stderr = process.communicate()
            logger.error("Failed to start API server")
            if stderr:
                logger.error(f"Error: {stderr.decode()}")
                
    except Exception as e:
        logger.error(f"Failed to run API server: {e}")

def run_etl_scheduler(interval=5, daemon=False):
    """Run the ETL scheduler"""
    try:
        logger.info(f"Starting ETL scheduler with {interval} minute intervals")
        
        cmd = [sys.executable, "scheduler.py", "--interval", str(interval)]
        
        if daemon:
            cmd.append("--daemon")
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if daemon:
            logger.info("ETL scheduler started in daemon mode")
            return process
        else:
            try:
                process.wait()
            except KeyboardInterrupt:
                logger.info("Shutting down ETL scheduler...")
                process.terminate()
                process.wait()
                
    except Exception as e:
        logger.error(f"Failed to run ETL scheduler: {e}")

def run_health_check():
    """Run system health check"""
    logger.info("Running system health check...")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{check_name}: {status}")
            if not result:
                all_passed = False
        except Exception as e:
            logger.error(f"{check_name}: ❌ ERROR - {e}")
            all_passed = False
    
    # Check data availability
    try:
        from data_aggregator import DataAggregator
        aggregator = DataAggregator()
        result = aggregator.get_latest_metrics()
        
        if result:
            logger.info("Data Availability: ✅ PASS")
        else:
            logger.info("Data Availability: ⚠️ WARN - No data found, run initialization")
            
    except Exception as e:
        logger.error(f"Data Availability: ❌ ERROR - {e}")
        all_passed = False
    
    if all_passed:
        logger.info("🎉 All health checks passed!")
    else:
        logger.error("⚠️ Some health checks failed. Please address issues before running dashboard.")
    
    return all_passed

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="NDIS Carer Data Pipeline Dashboard Launcher")
    parser.add_argument("--mode", choices=["dashboard", "api", "scheduler", "health"], 
                       default="dashboard", help="Run mode (default: dashboard)")
    parser.add_argument("--port", type=int, default=8501, help="Port number (default: 8501)")
    parser.add_argument("--host", default="localhost", help="Host address (default: localhost)")
    parser.add_argument("--interval", type=int, default=5, help="ETL interval in minutes (default: 5)")
    parser.add_argument("--daemon", action="store_true", help="Run scheduler in daemon mode")
    parser.add_argument("--skip-init", action="store_true", help="Skip database initialization")
    parser.add_argument("--init-only", action="store_true", help="Only initialize database and exit")
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    logger.info("🚀 NDIS Carer Data Pipeline Dashboard Launcher")
    logger.info(f"Mode: {args.mode}")
    
    # Run health check first
    if args.mode != "health":
        logger.info("Running pre-flight health check...")
        if not run_health_check():
            logger.error("Health check failed. Use --mode health for detailed diagnostics.")
            sys.exit(1)
    
    # Initialize database unless skipped
    if not args.skip_init and args.mode != "health":
        if not initialize_database():
            logger.error("Database initialization failed")
            sys.exit(1)
    
    # Exit if init-only
    if args.init_only:
        logger.info("Database initialization complete. Exiting.")
        return
    
    # Run selected mode
    try:
        if args.mode == "dashboard":
            run_streamlit_dashboard(port=args.port, host=args.host)
        
        elif args.mode == "api":
            run_flask_api(port=args.port, host=args.host)
        
        elif args.mode == "scheduler":
            run_etl_scheduler(interval=args.interval, daemon=args.daemon)
        
        elif args.mode == "health":
            run_health_check()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
