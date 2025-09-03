"""
English Free Data Generator
Generates NDIS Carer service records in English using free/local methods
"""

import json
import random
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from faker import Faker

from english_data_schema import (
    CarerServiceRecord, ServiceType, ServiceOutcome, LocationType,
    CarerProfile, ParticipantProfile, EnglishDataValidator
)
from config import get_config
from data_validator import ComprehensiveValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker for English/Australian context
fake = Faker(['en_AU'])


class EnglishFreeGenerator:
    """English Free Data Generator for NDIS Carer Records"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the generator"""
        self.config = config or get_config()
        self.gen_config = self.config["data_generation"]
        
        # Set random seeds for reproducibility
        random.seed(self.gen_config["random_seed"])
        fake.seed_instance(self.gen_config["random_seed"])
        
        # Initialize data validator
        self.validator = EnglishDataValidator()
        
        # Pre-generated profiles
        self.carers: List[CarerProfile] = []
        self.participants: List[ParticipantProfile] = []
        
        logger.info("English Free Data Generator initialized successfully")
    
    def generate_carer_profiles(self, count: int = 50) -> List[CarerProfile]:
        """Generate carer profiles"""
        profiles = []
        config = self.config["carer_profile"]
        
        # English specializations
        english_specializations = [
            "Personal Care", "Behavioral Support", "Cognitive Support", 
            "Physical Disability Support", "Mental Health Support", 
            "Aged Care", "Developmental Support", "Community Access"
        ]
        
        for i in range(count):
            carer_id = f"CR{random.randint(100000, 999999):06d}"
            
            profile = CarerProfile(
                carer_id=carer_id,
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                certification_level=random.choice(config["certification_levels"]),
                years_experience=random.randint(*config["experience_range"]),
                specializations=random.sample(
                    english_specializations, 
                    random.randint(1, 3)
                ),
                available_hours_per_week=random.randint(*config["hours_range"]),
                languages=random.sample(
                    config["languages"],
                    random.randint(1, 2)
                )
            )
            profiles.append(profile)
        
        return profiles
    
    def generate_participant_profiles(self, count: int = 100) -> List[ParticipantProfile]:
        """Generate participant profiles"""
        profiles = []
        config = self.config["participant_profile"]
        
        # English disability types
        english_disability_types = [
            "Intellectual Disability", "Autism Spectrum Disorder", 
            "Physical Disability", "Sensory Disability", 
            "Psychosocial Disability", "Neurological Disability", 
            "Multiple Disabilities", "Acquired Brain Injury"
        ]
        
        # English communication preferences
        english_communication = [
            "Verbal Communication", "Sign Language", "Picture Exchange", 
            "Written Communication", "Assistive Technology", "Simple Language"
        ]
        
        for i in range(count):
            participant_id = f"PT{random.randint(100000, 999999):06d}"
            
            profile = ParticipantProfile(
                participant_id=participant_id,
                age_group=random.choice(config["age_groups"]),
                disability_type=random.choice(english_disability_types),
                support_level=random.choice(config["support_levels"]),
                communication_preferences=random.sample(
                    english_communication,
                    random.randint(1, 2)
                ),
                mobility_requirements=random.choice([
                    [], ["wheelchair"], ["walking aid"], ["transfer assistance"]
                ])
            )
            profiles.append(profile)
        
        return profiles
    
    def generate_english_narrative(self, 
                                 service_type: ServiceType, 
                                 outcome: ServiceOutcome,
                                 participant_name: str = None) -> str:
        """Generate English care narrative"""
        
        participant_name = participant_name or fake.first_name()
        
        # Care techniques and methods
        techniques = [
            "progressive guidance", "positive reinforcement", "structured support", 
            "sensory regulation", "cognitive restructuring", "behavioral shaping", 
            "environmental adaptation", "communication assistance", "individualized support", 
            "team collaboration", "multi-sensory stimulation", "behavioral intervention",
            "person-centered approach", "trauma-informed care", "strengths-based approach"
        ]
        
        # Service locations
        locations = [
            "participant's home", "community center activity room", "rehabilitation facility",
            "outdoor garden area", "quiet therapy room", "dedicated treatment space",
            "familiar environment", "day care center", "community pool", "local library",
            "shopping center", "medical clinic", "support group venue"
        ]
        
        # Service type mapping to English
        service_mapping = {
            "个人护理": "Personal Care",
            "家务支持": "Household Tasks", 
            "社区参与": "Community Access",
            "交通协助": "Transport Assistance",
            "社交支持": "Social Support",
            "物理治疗": "Physiotherapy",
            "用药支持": "Medication Support",
            "技能发展": "Skill Development",
            "临时护理": "Respite Care",
            "餐食准备": "Meal Preparation"
        }
        
        service_type_en = service_mapping.get(service_type.value, service_type.value)
        technique = random.choice(techniques)
        location = random.choice(locations)
        
        if outcome == ServiceOutcome.POSITIVE:
            narratives = [
                f"Provided {service_type_en.lower()} support for {participant_name}. Participant demonstrated excellent cooperation and active engagement throughout the session. Carer utilized {technique} methods in the {location}, achieving significant positive outcomes. The service proceeded smoothly and met all planned objectives with high participant satisfaction.",
                
                f"Delivered professional {service_type_en.lower()} services to {participant_name} today. Participant showed strong motivation and willingness to participate in all activities. Through effective implementation of {technique} strategies at the {location}, we successfully accomplished the established care goals.",
                
                f"Assisted {participant_name} with {service_type_en.lower()} activities using {technique} approach. Participant responded positively to guidance and demonstrated improved skills during the session. The intervention at the {location} was highly effective and contributed to meaningful progress.",
                
                f"Facilitated {service_type_en.lower()} support for {participant_name} with outstanding results. Participant actively participated and showed enthusiasm for the activities. The {technique} methodology proved successful in the {location} environment, leading to achievement of therapeutic goals."
            ]
        elif outcome == ServiceOutcome.NEUTRAL:
            narratives = [
                f"Provided routine {service_type_en.lower()} support for {participant_name}. Participant maintained stable engagement and completed basic activities as planned. Carer applied {technique} methods at the {location}. The session proceeded normally without significant incidents or concerns.",
                
                f"Delivered standard {service_type_en.lower()} services to {participant_name}. Participant demonstrated consistent cooperation and followed established routines. Using {technique} approach in the {location}, we maintained steady progress according to the care plan.",
                
                f"Assisted {participant_name} with {service_type_en.lower()} activities in a structured manner. Participant showed average engagement and completed most planned tasks. The {technique} intervention at the {location} supported continued stability.",
                
                f"Conducted {service_type_en.lower()} session with {participant_name} following standard protocols. Participant remained calm and cooperative throughout. Applied {technique} strategies at the {location} to maintain consistent care quality."
            ]
        else:  # NEGATIVE or INCOMPLETE
            narratives = [
                f"Attempted to provide {service_type_en.lower()} support for {participant_name} but encountered significant challenges. Participant experienced emotional dysregulation and showed resistance to some activities. Carer implemented {technique} interventions at the {location} with limited effectiveness. Requires strategy adjustment and follow-up planning.",
                
                f"Provided {service_type_en.lower()} services to {participant_name} with mixed outcomes. Participant demonstrated difficulty with attention and cooperation during the session. Despite applying {technique} approaches at the {location}, progress was slower than anticipated. Additional support strategies needed.",
                
                f"Supported {participant_name} with {service_type_en.lower()} activities but faced behavioral challenges. Participant required extra patience and modified interventions. The {technique} method was partially effective at the {location}, but comprehensive review of care approach is recommended.",
                
                f"Delivered {service_type_en.lower()} support to {participant_name} under challenging circumstances. Participant showed signs of distress and limited engagement. While {technique} strategies were employed at the {location}, outcomes were below expectations and warrant multidisciplinary consultation."
            ]
        
        return random.choice(narratives)
    
    async def generate_service_record(self,
                                    carer: CarerProfile,
                                    participant: ParticipantProfile,
                                    service_date: date,
                                    service_type: ServiceType) -> Optional[CarerServiceRecord]:
        """Generate a single English service record"""
        
        try:
            # Generate basic record data
            record_id = f"SR{random.randint(10000000, 99999999):08d}"
            
            # Determine service outcome (based on weights)
            outcome_weights = self.config["service"]["outcome_weights"]
            outcomes = list(ServiceOutcome)
            weights = [outcome_weights.get(oc.value, 0.1) for oc in outcomes]
            service_outcome = random.choices(outcomes, weights=weights)[0]
            
            # Determine service duration
            duration_ranges = self.config["service"]["duration_ranges"]
            duration_range = duration_ranges.get(service_type.value, (1.0, 4.0))
            duration = round(random.uniform(*duration_range), 2)
            
            # Determine location
            location_weights = self.config["location"]["location_weights"]
            location_types = list(LocationType)
            loc_weights = [location_weights.get(lt.value, 0.01) for lt in location_types]
            location_type = random.choices(location_types, weights=loc_weights)[0]
            
            # Generate English narrative
            participant_name = fake.first_name()
            narrative = self.generate_english_narrative(service_type, service_outcome, participant_name)
            
            # Generate support techniques (English)
            english_support_techniques = random.sample([
                "Visual Prompts", "Verbal Guidance", "Physical Assistance", "Environmental Modification",
                "Behavioral Reinforcement", "Sensory Support", "Time Management", "Social Skills Training",
                "Personalized Communication", "Emotional Regulation", "Cognitive Training", "Functional Activities",
                "Adaptive Equipment", "Routine Establishment", "Crisis Intervention", "Peer Support"
            ], random.randint(2, 4))
            
            # Generate challenges (English)
            challenges = []
            if service_outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]:
                challenges = random.sample([
                    "Emotional regulation difficulties", "Environmental adaptation challenges", "Communication barriers",
                    "Attention span limitations", "Physical fatigue", "Cognitive load management",
                    "Sensory sensitivities", "Behavioral expression", "Social interaction difficulties",
                    "Medication side effects", "Family dynamics", "Transportation issues"
                ], random.randint(1, 3))
            
            # Participant responses (English)
            participant_responses = {
                ServiceOutcome.POSITIVE: ["Highly cooperative", "Actively engaged", "Excellent progress", "Very satisfied"],
                ServiceOutcome.NEUTRAL: ["Generally cooperative", "Stable engagement", "Consistent participation", "Calm demeanor"],
                ServiceOutcome.NEGATIVE: ["Required encouragement", "Emotional fluctuations", "Needed additional support", "Expressed frustration"],
                ServiceOutcome.INCOMPLETE: ["Needed frequent breaks", "Attention difficulties", "Fatigue observed", "Session shortened"]
            }
            
            # Location details mapping to English
            location_mapping = {
                "参与者家中": "Participant's Home",
                "社区中心": "Community Centre", 
                "医疗机构": "Healthcare Facility",
                "购物中心": "Shopping Centre",
                "图书馆": "Library",
                "游泳馆": "Swimming Pool",
                "药房": "Pharmacy",
                "公园": "Park",
                "其他": "Other Location"
            }
            
            location_en = location_mapping.get(location_type.value, location_type.value)
            
            # Create service record
            record = CarerServiceRecord(
                record_id=record_id,
                carer_id=carer.carer_id,
                participant_id=participant.participant_id,
                service_date=service_date,
                service_type=service_type,
                duration_hours=duration,
                narrative_notes=narrative,
                location_type=location_type,
                location_details=f"{location_en} - Designated support area",
                service_outcome=service_outcome,
                support_techniques_used=english_support_techniques,
                challenges_encountered=challenges,
                participant_response=random.choice(participant_responses.get(service_outcome, ["Normal response"])),
                follow_up_required=service_outcome in [ServiceOutcome.NEGATIVE, ServiceOutcome.INCOMPLETE]
            )
            
            # Validate record
            errors = self.validator.validate_service_record(record)
            if errors:
                logger.warning(f"Record validation failed: {errors}")
                return None
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to generate service record: {e}")
            return None
    
    async def generate_dataset(self, size: int = 1000) -> List[CarerServiceRecord]:
        """Generate complete English dataset"""
        logger.info(f"Starting generation of {size} English service records")
        
        # Generate profiles
        self.carers = self.generate_carer_profiles(max(10, size // 20))
        self.participants = self.generate_participant_profiles(max(20, size // 10))
        
        records = []
        
        # Service type weights
        service_weights = self.config["service"]["service_types_weights"]
        service_types = list(ServiceType)
        weights = [service_weights.get(st.value, 0.1) for st in service_types]
        
        # Generate records
        for i in range(size):
            # Randomly select carer and participant
            carer = random.choice(self.carers)
            participant = random.choice(self.participants)
            
            # Generate service date (within past 90 days)
            days_ago = random.randint(1, 90)
            service_date = date.today() - timedelta(days=days_ago)
            
            # Select service type
            service_type = random.choices(service_types, weights=weights)[0]
            
            try:
                record = await self.generate_service_record(carer, participant, service_date, service_type)
                if record:
                    records.append(record)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Generated {i + 1} records, successful: {len(records)}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate record {i+1}: {e}")
        
        logger.info(f"Dataset generation completed. Total valid records: {len(records)}")
        return records
    
    def save_dataset(self, 
                    records: List[CarerServiceRecord], 
                    filename_prefix: str = "english_carers_data") -> Dict[str, str]:
        """Save dataset in multiple formats"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        data_dicts = [record.to_dict() for record in records]
        
        # Save JSON
        json_file = output_dir / f"{filename_prefix}_{timestamp}_{len(records)}records.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data_dicts, f, ensure_ascii=False, indent=2, default=str)
        saved_files["json"] = str(json_file)
        
        # Save JSONL
        jsonl_file = output_dir / f"{filename_prefix}_{timestamp}_{len(records)}records.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for record_dict in data_dicts:
                f.write(json.dumps(record_dict, ensure_ascii=False, default=str) + '\n')
        saved_files["jsonl"] = str(jsonl_file)
        
        # Save CSV
        try:
            import pandas as pd
            df = pd.DataFrame(data_dicts)
            csv_file = output_dir / f"{filename_prefix}_{timestamp}_{len(records)}records.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            saved_files["csv"] = str(csv_file)
        except ImportError:
            logger.warning("pandas not installed, skipping CSV export")
        
        logger.info(f"Dataset saved to: {saved_files}")
        return saved_files


async def main():
    """Main function - demonstrate English free data generation"""
    generator = EnglishFreeGenerator()
    
    # Generate test data
    test_size = 100
    logger.info(f"Generating English test dataset ({test_size} records)")
    
    records = await generator.generate_dataset(test_size)
    
    if records:
        # Save data
        saved_files = generator.save_dataset(records)
        
        # Perform validation
        logger.info("Performing data validation...")
        validator = ComprehensiveValidator()
        validation_results = validator.comprehensive_validation(records)
        
        # Save validation report
        report_file = validator.save_validation_report(
            validation_results, 
            f"english_validation_report_{test_size}records.json"
        )
        
        # Output results
        print(f"\n✅ English data generation completed successfully!")
        print(f"📊 Generated records: {len(records)}")
        print(f"🎯 Quality score: {validation_results['overall_score']}/100")
        print(f"🔒 Privacy score: {validation_results['privacy_analysis']['anonymization_score']}/100")
        print(f"📁 Saved files:")
        for format_type, filepath in saved_files.items():
            print(f"   {format_type}: {filepath}")
        print(f"📋 Validation report: {report_file}")
        
        # Display sample record
        print(f"\n📋 Sample record:")
        print(records[0].to_json())
        
    else:
        logger.error("Failed to generate any valid records")


if __name__ == "__main__":
    asyncio.run(main())
