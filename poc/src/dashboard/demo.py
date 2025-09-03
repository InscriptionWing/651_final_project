#!/usr/bin/env python3
"""
Dashboard Demo Script
Generates sample data and demonstrates dashboard capabilities
"""

import json
import random
from datetime import datetime, date, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 尝试导入英文数据模式，如果失败则使用默认模式
try:
    from english_data_schema import CarerServiceRecord, ServiceType, ServiceOutcome, LocationType
    logger.info("Using English data schema for demo")
except ImportError:
    from carer_data_schema import CarerServiceRecord, ServiceType, ServiceOutcome, LocationType
    logger.info("Using default data schema for demo")
from data_aggregator import DataAggregator
from config import get_dashboard_config, OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardDemo:
    """Demo data generator for dashboard"""
    
    def __init__(self):
        self.config = get_dashboard_config()
        self.aggregator = DataAggregator()
        
    def generate_demo_data(self, num_records: int = 200) -> List[Dict[str, Any]]:
        """Generate demo data for dashboard"""
        logger.info(f"Generating {num_records} demo records for dashboard")
        
        records = []
        start_date = date.today() - timedelta(days=30)
        
        # Sample narratives for different service types
        narrative_templates = {
            ServiceType.PERSONAL_CARE: [
                "Assisted participant with personal hygiene and grooming activities. Client showed good cooperation and maintained dignity throughout the session.",
                "Provided support with dressing and mobility assistance. Participant demonstrated improved confidence in daily activities.",
                "Completed personal care routine including bathing assistance. Client expressed satisfaction with the level of support provided."
            ],
            ServiceType.HOUSEHOLD_TASKS: [
                "Completed household cleaning tasks including kitchen and bathroom areas. Participant observed and learned organizational techniques.",
                "Assisted with laundry and basic meal preparation. Client showed interest in developing independent living skills.",
                "Performed general housekeeping duties and organized living spaces for improved accessibility."
            ],
            ServiceType.COMMUNITY_ACCESS: [
                "Accompanied participant to local library for reading program. Client engaged well with community activities.",
                "Supported community shopping trip with focus on budgeting skills. Participant demonstrated increased independence.",
                "Facilitated attendance at community center activities. Client showed improved social interaction skills."
            ],
            ServiceType.TRANSPORT: [
                "Provided transport assistance to medical appointment. Participant arrived on time and felt supported throughout journey.",
                "Accompanied client to social activities via public transport. Provided guidance on navigation and safety.",
                "Assisted with transport coordination for employment-related activities. Client gained confidence in travel planning."
            ],
            ServiceType.SOCIAL_SUPPORT: [
                "Engaged in conversation and social activities to reduce isolation. Participant showed improved mood and engagement.",
                "Facilitated social interaction with peers at community group. Client demonstrated better communication skills.",
                "Provided emotional support and active listening during challenging period. Participant expressed appreciation for the support."
            ]
        }
        
        for i in range(num_records):
            # Generate realistic data
            service_date = start_date + timedelta(days=random.randint(0, 30))
            service_type = random.choice(list(ServiceType))
            duration = round(random.uniform(0.5, 8.0), 2)
            
            # Weight outcomes realistically
            outcome_weights = [0.6, 0.25, 0.1, 0.05]  # positive, neutral, negative, incomplete
            service_outcome = random.choices(list(ServiceOutcome), weights=outcome_weights)[0]
            
            # Select appropriate narrative
            narratives = narrative_templates.get(service_type, ["Standard service provided with good outcomes."])
            narrative = random.choice(narratives)
            
            # Add some variation to narrative length
            if random.random() < 0.3:  # 30% chance of longer narrative
                narrative += f" Additional notes: Session lasted {duration} hours with positive participant engagement throughout."
            
            record_data = {
                "record_id": f"SR{random.randint(10000000, 99999999)}",
                "carer_id": f"CR{random.randint(100000, 999999)}",
                "participant_id": f"PT{random.randint(100000, 999999)}",
                "service_date": service_date.isoformat(),
                "service_type": service_type.value,
                "duration_hours": duration,
                "narrative_notes": narrative,
                "location_type": random.choice(list(LocationType)).value,
                "service_outcome": service_outcome.value,
                "support_techniques_used": random.sample([
                    "Verbal guidance", "Visual prompts", "Physical assistance", 
                    "Sensory support", "Communication aids", "Behavioral strategies"
                ], k=random.randint(1, 3)),
                "follow_up_required": random.choice([True, False]),
                "created_timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
            }
            
            records.append(record_data)
        
        return records
    
    def generate_validation_report(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate realistic validation report"""
        total_records = len(records)
        
        # Simulate validation results
        passed_records = int(total_records * random.uniform(0.85, 0.95))
        failed_records = total_records - passed_records
        
        # Generate realistic scores
        overall_score = random.uniform(82, 95)
        privacy_score = random.uniform(75, 90)
        realism_score = random.uniform(80, 95)
        
        # Generate quality gate results
        quality_gates = {
            "schema_validation": {
                "passed": passed_records,
                "failed": failed_records,
                "failure_rate": (failed_records / total_records) * 100
            },
            "privacy_risks": {
                "score": privacy_score,
                "risks_detected": random.randint(0, 5),
                "failure_rate": 100 - privacy_score
            },
            "utility_quality": {
                "score": realism_score,
                "failure_rate": 100 - realism_score
            }
        }
        
        # Generate recommendations
        recommendations = []
        if privacy_score < 85:
            recommendations.append("Consider improving template diversity to reduce potential PII patterns")
        if realism_score < 85:
            recommendations.append("Enhance narrative quality by expanding template variety")
        if failed_records > total_records * 0.1:
            recommendations.append("Review validation rules for potential over-strictness")
        
        validation_report = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_records": total_records,
            "overall_score": overall_score,
            "schema_validation": {
                "passed_records": passed_records,
                "failed_records": failed_records,
                "compliance_rate": (passed_records / total_records) * 100
            },
            "privacy_analysis": {
                "anonymization_score": privacy_score,
                "potential_pii": [f"Potential identifier {i}" for i in range(random.randint(0, 3))]
            },
            "utility_analysis": {
                "realism_score": realism_score,
                "narrative_quality": random.uniform(75, 90)
            },
            "quality_gates": quality_gates,
            "validation_errors": [
                f"Record SR{random.randint(10000000, 99999999)}: Narrative too short"
                for _ in range(failed_records)
            ],
            "recommendations": recommendations
        }
        
        return validation_report
    
    def save_demo_files(self, records: List[Dict[str, Any]], validation_report: Dict[str, Any]):
        """Save demo files to output directory"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"demo_dashboard_data_{timestamp}_{len(records)}records"
        
        # Save JSON
        json_file = OUTPUT_DIR / f"{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        # Save JSONL
        jsonl_file = OUTPUT_DIR / f"{base_name}.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # Save validation report
        report_file = OUTPUT_DIR / f"demo_validation_report_{len(records)}records.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Demo files saved:")
        logger.info(f"  Data: {json_file}")
        logger.info(f"  JSONL: {jsonl_file}")
        logger.info(f"  Validation: {report_file}")
        
        return {
            "json": str(json_file),
            "jsonl": str(jsonl_file),
            "validation": str(report_file)
        }
    
    def run_demo(self, num_records: int = 200):
        """Run complete dashboard demo"""
        logger.info("🎭 Starting Dashboard Demo")
        
        # Generate demo data
        records = self.generate_demo_data(num_records)
        logger.info(f"Generated {len(records)} demo records")
        
        # Generate validation report
        validation_report = self.generate_validation_report(records)
        logger.info(f"Generated validation report with {validation_report['overall_score']:.1f}/100 score")
        
        # Save files
        saved_files = self.save_demo_files(records, validation_report)
        
        # Run data aggregation
        logger.info("Running data aggregation...")
        result = self.aggregator.aggregate_all_data()
        
        if result.get("status") == "success":
            logger.info("✅ Data aggregation completed successfully")
            
            # Display summary
            output_metrics = result.get("output_metrics", {})
            validation_metrics = result.get("validation_metrics", {})
            derived_metrics = result.get("derived_metrics", {})
            
            logger.info("\n📊 Demo Data Summary:")
            logger.info(f"  Total Records: {output_metrics.get('total_records', 0):,}")
            logger.info(f"  Unique Carers: {output_metrics.get('unique_carers', 0):,}")
            logger.info(f"  Unique Participants: {output_metrics.get('unique_participants', 0):,}")
            logger.info(f"  Overall Quality Score: {validation_metrics.get('overall_score', 0):.1f}/100")
            logger.info(f"  Pass Rate: {derived_metrics.get('pass_rate', 0):.1f}%")
            logger.info(f"  Avg Narrative Length: {derived_metrics.get('avg_narrative_length', 0):.0f} chars")
            
            # Service type distribution
            service_dist = output_metrics.get("service_type_distribution", {})
            if service_dist:
                logger.info("\n📈 Service Type Distribution:")
                for service_type, count in service_dist.items():
                    percentage = (count / output_metrics.get('total_records', 1)) * 100
                    logger.info(f"  {service_type}: {count} ({percentage:.1f}%)")
            
            logger.info(f"\n🎉 Demo completed successfully!")
            logger.info(f"Dashboard data is ready. Run 'python run_dashboard.py' to view the dashboard.")
            
        else:
            logger.error(f"❌ Data aggregation failed: {result.get('error', 'Unknown error')}")
    
    def generate_time_series_data(self, days: int = 7):
        """Generate time series data for trend analysis"""
        logger.info(f"Generating time series data for {days} days")
        
        for day in range(days):
            date_offset = timedelta(days=day)
            
            # Generate records for this day
            daily_records = self.generate_demo_data(random.randint(20, 50))
            
            # Adjust timestamps
            target_date = datetime.now() - timedelta(days=days-day)
            for record in daily_records:
                record["created_timestamp"] = target_date.isoformat()
            
            # Generate validation report
            validation_report = self.generate_validation_report(daily_records)
            validation_report["validation_timestamp"] = target_date.isoformat()
            
            # Save files with date suffix
            timestamp = target_date.strftime("%Y%m%d_%H%M%S")
            base_name = f"demo_timeseries_{timestamp}_{len(daily_records)}records"
            
            # Save files
            json_file = OUTPUT_DIR / f"{base_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(daily_records, f, indent=2, ensure_ascii=False)
            
            report_file = OUTPUT_DIR / f"demo_validation_{timestamp}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated data for day {day+1}/{days}: {len(daily_records)} records")
        
        # Run aggregation to update database
        self.aggregator.aggregate_all_data()
        logger.info("✅ Time series data generation completed")

def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dashboard Demo Data Generator")
    parser.add_argument("--records", type=int, default=200, help="Number of records to generate")
    parser.add_argument("--time-series", action="store_true", help="Generate time series data")
    parser.add_argument("--days", type=int, default=7, help="Number of days for time series")
    
    args = parser.parse_args()
    
    demo = DashboardDemo()
    
    if args.time_series:
        demo.generate_time_series_data(args.days)
    else:
        demo.run_demo(args.records)

if __name__ == "__main__":
    main()
