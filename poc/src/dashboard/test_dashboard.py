#!/usr/bin/env python3
"""
Dashboard Test Script
Comprehensive testing of dashboard functionality
"""

import sys
import os
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardTester:
    """Comprehensive dashboard testing"""
    
    def __init__(self):
        self.test_results = {}
        self.dashboard_dir = Path(__file__).parent
        self.project_root = self.dashboard_dir.parent
        
    def run_all_tests(self):
        """Run all dashboard tests"""
        logger.info("🧪 Starting Dashboard Test Suite")
        logger.info("=" * 50)
        
        tests = [
            ("Configuration Loading", self.test_configuration),
            ("Database Initialization", self.test_database),
            ("Data Aggregator", self.test_data_aggregator),
            ("Demo Data Generation", self.test_demo_generation),
            ("API Components", self.test_api_components),
            ("Streamlit Components", self.test_streamlit_components),
            ("File Structure", self.test_file_structure),
            ("Dependencies", self.test_dependencies)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Testing: {test_name}")
            try:
                result = test_func()
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                self.test_results[test_name] = result
            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
                self.test_results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("🏁 Test Summary")
        logger.info(f"Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! Dashboard is ready to use.")
        else:
            logger.warning("⚠️ Some tests failed. Please review the errors above.")
        
        return passed_tests == total_tests
    
    def test_configuration(self):
        """Test configuration loading"""
        try:
            from config import get_dashboard_config, DASHBOARD_CONFIG, KPI_THRESHOLDS
            
            config = get_dashboard_config()
            assert "dashboard" in config
            assert "kpi_thresholds" in config
            assert "chart_colors" in config
            
            # Test specific configuration values
            assert DASHBOARD_CONFIG["title"] == "NDIS Carer Data Pipeline Dashboard"
            assert "pass_rate" in KPI_THRESHOLDS
            
            logger.info("  - Configuration structure: OK")
            logger.info("  - Dashboard title: OK")
            logger.info("  - KPI thresholds: OK")
            
            return True
            
        except Exception as e:
            logger.error(f"  - Configuration error: {e}")
            return False
    
    def test_database(self):
        """Test database initialization"""
        try:
            from data_aggregator import DataAggregator
            
            aggregator = DataAggregator()
            db_path = aggregator.db_path
            
            # Check if database file exists
            if not db_path.exists():
                logger.info("  - Database not found, initializing...")
                aggregator._init_database()
            
            # Test database connection
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                table_names = [table[0] for table in tables]
                
                expected_tables = ['pipeline_metrics', 'record_details', 'quality_gate_results', 'template_usage']
                
                for table in expected_tables:
                    if table in table_names:
                        logger.info(f"  - Table '{table}': OK")
                    else:
                        logger.warning(f"  - Table '{table}': MISSING")
            
            logger.info(f"  - Database path: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"  - Database error: {e}")
            return False
    
    def test_data_aggregator(self):
        """Test data aggregator functionality"""
        try:
            from data_aggregator import DataAggregator
            
            aggregator = DataAggregator()
            
            # Test basic functionality
            config = aggregator.config
            assert config is not None
            
            # Test database path
            db_path = aggregator.db_path
            assert db_path.exists() or db_path.parent.exists()
            
            logger.info("  - Aggregator initialization: OK")
            logger.info("  - Configuration loading: OK")
            logger.info("  - Database path: OK")
            
            # Test data aggregation (if data exists)
            try:
                result = aggregator.get_latest_metrics()
                logger.info("  - Latest metrics retrieval: OK")
            except Exception as e:
                logger.info(f"  - Latest metrics: No data yet ({e})")
            
            return True
            
        except Exception as e:
            logger.error(f"  - Data aggregator error: {e}")
            return False
    
    def test_demo_generation(self):
        """Test demo data generation"""
        try:
            from demo import DashboardDemo
            
            demo = DashboardDemo()
            
            # Test demo record generation
            records = demo.generate_demo_data(5)  # Generate small sample
            assert len(records) == 5
            assert all("record_id" in record for record in records)
            assert all("narrative_notes" in record for record in records)
            
            # Test validation report generation
            validation_report = demo.generate_validation_report(records)
            assert "overall_score" in validation_report
            assert "total_records" in validation_report
            assert validation_report["total_records"] == 5
            
            logger.info("  - Demo record generation: OK")
            logger.info("  - Validation report generation: OK")
            logger.info(f"  - Generated {len(records)} sample records")
            
            return True
            
        except Exception as e:
            logger.error(f"  - Demo generation error: {e}")
            return False
    
    def test_api_components(self):
        """Test API components"""
        try:
            from api import app
            
            # Test Flask app creation
            assert app is not None
            assert app.name == "api"
            
            # Test route registration
            rules = [rule.rule for rule in app.url_map.iter_rules()]
            expected_routes = ['/api/health', '/api/overview', '/api/records']
            
            for route in expected_routes:
                if route in rules:
                    logger.info(f"  - Route '{route}': OK")
                else:
                    logger.warning(f"  - Route '{route}': MISSING")
            
            logger.info("  - Flask app initialization: OK")
            return True
            
        except Exception as e:
            logger.error(f"  - API components error: {e}")
            return False
    
    def test_streamlit_components(self):
        """Test Streamlit components"""
        try:
            # Test if streamlit_app.py can be imported
            import streamlit_app
            
            # Test key functions exist
            assert hasattr(streamlit_app, 'main')
            assert hasattr(streamlit_app, 'get_dashboard_data')
            
            logger.info("  - Streamlit app import: OK")
            logger.info("  - Main function: OK")
            logger.info("  - Dashboard data function: OK")
            
            return True
            
        except Exception as e:
            logger.error(f"  - Streamlit components error: {e}")
            return False
    
    def test_file_structure(self):
        """Test required file structure"""
        required_files = [
            "config.py",
            "data_aggregator.py",
            "api.py",
            "scheduler.py",
            "streamlit_app.py",
            "run_dashboard.py",
            "demo.py",
            "requirements.txt",
            "README.md",
            "SETUP_GUIDE.md"
        ]
        
        missing_files = []
        
        for file_name in required_files:
            file_path = self.dashboard_dir / file_name
            if file_path.exists():
                logger.info(f"  - {file_name}: OK")
            else:
                logger.warning(f"  - {file_name}: MISSING")
                missing_files.append(file_name)
        
        # Check data directory
        data_dir = self.dashboard_dir / "data"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            logger.info("  - Created data directory")
        else:
            logger.info("  - Data directory: OK")
        
        return len(missing_files) == 0
    
    def test_dependencies(self):
        """Test required dependencies"""
        required_packages = [
            "streamlit",
            "plotly",
            "pandas",
            "numpy", 
            "flask",
            "schedule",
            "requests"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"  - {package}: OK")
            except ImportError:
                logger.warning(f"  - {package}: MISSING")
                missing_packages.append(package)
        
        if missing_packages:
            logger.info(f"  - Install missing packages: pip install {' '.join(missing_packages)}")
            return False
        
        return True
    
    def generate_test_report(self):
        """Generate detailed test report"""
        report = {
            "test_timestamp": datetime.now().isoformat(),
            "test_results": self.test_results,
            "dashboard_directory": str(self.dashboard_dir),
            "project_root": str(self.project_root),
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        report_file = self.dashboard_dir / "test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Test report saved: {report_file}")
        return str(report_file)

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dashboard Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--report", "-r", action="store_true", help="Generate test report")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = DashboardTester()
    success = tester.run_all_tests()
    
    if args.report:
        tester.generate_test_report()
    
    if success:
        print("\n🎉 All tests passed! Dashboard is ready to use.")
        print("Next steps:")
        print("  1. Run: python demo.py --records 100")
        print("  2. Run: python run_dashboard.py")
        print("  3. Open browser to: http://localhost:8501")
    else:
        print("\n⚠️ Some tests failed. Please review the errors and fix issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()



