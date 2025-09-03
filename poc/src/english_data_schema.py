"""
English Data Schema for NDIS Carer Service Records
Defines data structures and validation rules in English
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import re


class ServiceType(Enum):
    """Service type enumeration in English"""
    PERSONAL_CARE = "Personal Care"
    HOUSEHOLD_TASKS = "Household Tasks"
    COMMUNITY_ACCESS = "Community Access"
    TRANSPORT = "Transport Assistance"
    SOCIAL_SUPPORT = "Social Support"
    PHYSIOTHERAPY = "Physiotherapy"
    MEDICATION_SUPPORT = "Medication Support"
    SKILL_DEVELOPMENT = "Skill Development"
    RESPITE_CARE = "Respite Care"
    MEAL_PREPARATION = "Meal Preparation"


class ServiceOutcome(Enum):
    """Service outcome enumeration"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    INCOMPLETE = "incomplete"


class LocationType(Enum):
    """Service location type enumeration in English"""
    HOME = "Participant's Home"
    COMMUNITY_CENTRE = "Community Centre"
    HEALTHCARE_FACILITY = "Healthcare Facility"
    SHOPPING_CENTRE = "Shopping Centre"
    LIBRARY = "Library"
    POOL = "Swimming Pool"
    PHARMACY = "Pharmacy"
    PARK = "Park"
    OTHER = "Other Location"


@dataclass
class CarerProfile:
    """English Carer Profile"""
    carer_id: str
    first_name: str
    last_name: str
    certification_level: str  # Level 1-4 or Certificate III/IV, Diploma, Degree
    years_experience: int
    specializations: List[str]
    available_hours_per_week: int
    languages: List[str] = field(default_factory=lambda: ["English"])
    
    def __post_init__(self):
        """Validate carer profile data"""
        if not re.match(r'^CR\d{6}$', self.carer_id):
            raise ValueError(f"Invalid carer ID format: {self.carer_id}")
        if self.years_experience < 0 or self.years_experience > 50:
            raise ValueError(f"Invalid years of experience: {self.years_experience}")


@dataclass
class ParticipantProfile:
    """English Participant Profile (de-identified)"""
    participant_id: str
    age_group: str  # "18-25", "26-35", "36-50", "51-65", "65+"
    disability_type: str
    support_level: str  # "Low", "Medium", "High", "Complex"
    communication_preferences: List[str]
    mobility_requirements: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate participant profile data"""
        if not re.match(r'^PT\d{6}$', self.participant_id):
            raise ValueError(f"Invalid participant ID format: {self.participant_id}")


@dataclass
class CarerServiceRecord:
    """English Carer Service Record - Core Data Structure"""
    # Required fields
    record_id: str
    carer_id: str
    carer_name: str
    participant_id: str
    service_date: date
    service_type: ServiceType
    duration_hours: float
    narrative_notes: str
    
    # Auto-generated fields
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    # Optional fields
    location_type: Optional[LocationType] = None
    location_details: Optional[str] = None
    service_outcome: Optional[ServiceOutcome] = None
    support_techniques_used: List[str] = field(default_factory=list)
    challenges_encountered: List[str] = field(default_factory=list)
    participant_response: Optional[str] = None
    follow_up_required: bool = False
    billing_code: Optional[str] = None
    supervision_notes: Optional[str] = None
    
    def __post_init__(self):
        """Validate service record data"""
        # Validate ID formats
        if not re.match(r'^SR\d{8}$', self.record_id):
            raise ValueError(f"Invalid record ID format: {self.record_id}")
        if not re.match(r'^CR\d{6}$', self.carer_id):
            raise ValueError(f"Invalid carer ID format: {self.carer_id}")
        if not re.match(r'^PT\d{6}$', self.participant_id):
            raise ValueError(f"Invalid participant ID format: {self.participant_id}")
        
        # Validate duration
        if self.duration_hours <= 0 or self.duration_hours > 24:
            raise ValueError(f"Invalid service duration: {self.duration_hours}")
        
        # Validate narrative length
        if len(self.narrative_notes) < 50 or len(self.narrative_notes) > 1000:
            raise ValueError(f"Narrative length should be 50-1000 characters, current: {len(self.narrative_notes)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, (date, datetime)):
                result[key] = value.isoformat()
            elif isinstance(value, list):
                result[key] = [v.value if isinstance(v, Enum) else v for v in value]
            else:
                result[key] = value
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class EnglishDataValidator:
    """English Data Validator"""
    
    @staticmethod
    def validate_service_record(record: CarerServiceRecord) -> List[str]:
        """Validate service record, return list of errors"""
        errors = []
        
        # Check required fields
        if not record.narrative_notes.strip():
            errors.append("Narrative content cannot be empty")
        
        # Check narrative content quality - support English keywords
        narrative = record.narrative_notes.lower()
        english_keywords = [
            'assist', 'support', 'care', 'service', 'help', 'participant', 
            'activity', 'therapy', 'training', 'guidance', 'intervention',
            'carer', 'client', 'individual', 'person', 'patient', 'user',
            'provide', 'deliver', 'facilitate', 'enable', 'encourage'
        ]
        
        has_keywords = any(word in narrative for word in english_keywords)
        
        if not has_keywords:
            errors.append("Narrative content should contain care-related keywords")
        
        # Check date reasonableness
        if record.service_date > date.today():
            errors.append("Service date cannot be in the future")
        
        # Check duration reasonableness
        if record.service_type == ServiceType.TRANSPORT and record.duration_hours > 8:
            errors.append("Transport service duration should not exceed 8 hours")
        
        return errors
    
    @staticmethod
    def validate_data_quality(records: List[CarerServiceRecord]) -> Dict[str, Any]:
        """Validate dataset quality"""
        if not records:
            return {"valid": False, "errors": ["Dataset is empty"]}
        
        total_records = len(records)
        unique_carers = len(set(r.carer_id for r in records))
        unique_participants = len(set(r.participant_id for r in records))
        
        # Check data distribution
        service_types = [r.service_type.value for r in records]
        service_type_counts = {st: service_types.count(st) for st in set(service_types)}
        
        outcomes = [r.service_outcome.value for r in records if r.service_outcome]
        outcome_counts = {oc: outcomes.count(oc) for oc in set(outcomes)}
        
        return {
            "valid": True,
            "total_records": total_records,
            "unique_carers": unique_carers,
            "unique_participants": unique_participants,
            "service_type_distribution": service_type_counts,
            "outcome_distribution": outcome_counts,
            "avg_duration": sum(r.duration_hours for r in records) / total_records,
            "avg_narrative_length": sum(len(r.narrative_notes) for r in records) / total_records
        }


# English Data Dictionary and Schema Information
ENGLISH_DATA_DICTIONARY = {
    "CarerServiceRecord": {
        "description": "Core data structure for carer service records",
        "fields": {
            "record_id": {
                "type": "string",
                "format": "SR########",
                "description": "Unique service record identifier",
                "required": True
            },
            "carer_id": {
                "type": "string", 
                "format": "CR######",
                "description": "Unique carer identifier",
                "required": True
            },
            "participant_id": {
                "type": "string",
                "format": "PT######", 
                "description": "Unique participant identifier (de-identified)",
                "required": True
            },
            "service_date": {
                "type": "date",
                "description": "Date service was provided",
                "required": True
            },
            "service_type": {
                "type": "enum",
                "values": [st.value for st in ServiceType],
                "description": "Type of service provided",
                "required": True
            },
            "duration_hours": {
                "type": "float",
                "range": "0.25-24.0",
                "description": "Service duration in hours",
                "required": True
            },
            "narrative_notes": {
                "type": "string",
                "length": "50-1000 characters",
                "description": "Detailed service record narrative",
                "required": True
            },
            "location_type": {
                "type": "enum",
                "values": [lt.value for lt in LocationType],
                "description": "Type of service location",
                "required": False
            },
            "service_outcome": {
                "type": "enum", 
                "values": [so.value for so in ServiceOutcome],
                "description": "Service outcome assessment",
                "required": False
            },
            "support_techniques_used": {
                "type": "array",
                "description": "List of support techniques employed",
                "required": False
            },
            "challenges_encountered": {
                "type": "array",
                "description": "List of challenges faced during service",
                "required": False
            }
        }
    }
}

# English Configuration Constants
ENGLISH_SERVICE_TYPES = {
    "Personal Care": "Assistance with daily living activities",
    "Household Tasks": "Support with domestic activities", 
    "Community Access": "Participation in community activities",
    "Transport Assistance": "Support with transportation needs",
    "Social Support": "Social and emotional support",
    "Physiotherapy": "Physical therapy and rehabilitation",
    "Medication Support": "Assistance with medication management",
    "Skill Development": "Development of life skills",
    "Respite Care": "Temporary care relief",
    "Meal Preparation": "Assistance with food preparation"
}

ENGLISH_SUPPORT_TECHNIQUES = [
    "Visual Prompts", "Verbal Guidance", "Physical Assistance", 
    "Environmental Modification", "Behavioral Reinforcement", "Sensory Support",
    "Time Management", "Social Skills Training", "Personalized Communication",
    "Emotional Regulation", "Cognitive Training", "Functional Activities",
    "Adaptive Equipment", "Routine Establishment", "Crisis Intervention"
]

ENGLISH_CHALLENGES = [
    "Communication barriers", "Behavioral challenges", "Environmental factors",
    "Physical limitations", "Cognitive difficulties", "Sensory sensitivities",
    "Emotional regulation", "Social interaction", "Equipment issues",
    "Time constraints", "Family dynamics", "Medical complications"
]

if __name__ == "__main__":
    # Test English data schema
    from datetime import date
    
    test_record = CarerServiceRecord(
        record_id="SR12345678",
        carer_id="CR123456",
        participant_id="PT654321",
        service_date=date.today(),
        service_type=ServiceType.PERSONAL_CARE,
        duration_hours=2.5,
        narrative_notes="Provided personal care support for the participant, assisting with daily hygiene activities. Participant demonstrated good cooperation and completed established goals successfully."
    )
    
    print("Test English Record:")
    print(test_record.to_json())
    
    validator = EnglishDataValidator()
    errors = validator.validate_service_record(test_record)
    print(f"\nValidation Results: {len(errors)} errors")
    for error in errors:
        print(f"- {error}")
