"""
数据验证和质量检查模块
提供全面的数据质量评估、隐私检查和效用性分析
"""

import json
import re
import statistics
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter, defaultdict
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from carer_data_schema import CarerServiceRecord, ServiceType, ServiceOutcome
from config import get_config

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """数据质量验证器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.validation_config = self.config["validation"]
        
    def validate_schema_compliance(self, records: List[CarerServiceRecord]) -> Dict[str, Any]:
        """验证数据模式合规性"""
        results = {
            "total_records": len(records),
            "valid_records": 0,
            "invalid_records": 0,
            "validation_errors": [],
            "field_completeness": {},
            "data_type_errors": []
        }
        
        field_counts = defaultdict(int)
        
        for i, record in enumerate(records):
            try:
                # 检查必填字段
                record_dict = record.to_dict()
                for field, value in record_dict.items():
                    if value is not None and value != "":
                        field_counts[field] += 1
                
                # 验证数据类型和格式
                self._validate_record_format(record, i, results)
                results["valid_records"] += 1
                
            except Exception as e:
                results["invalid_records"] += 1
                results["validation_errors"].append({
                    "record_index": i,
                    "error": str(e)
                })
        
        # 计算字段完整性
        for field, count in field_counts.items():
            results["field_completeness"][field] = {
                "count": count,
                "percentage": (count / len(records)) * 100
            }
        
        return results
    
    def _validate_record_format(self, record: CarerServiceRecord, index: int, results: Dict):
        """验证单条记录格式"""
        # ID格式验证
        if not re.match(r'^SR\d{8}$', record.record_id):
            results["data_type_errors"].append({
                "record_index": index,
                "field": "record_id",
                "error": "ID格式不正确"
            })
        
        # 日期验证
        if record.service_date > date.today():
            results["data_type_errors"].append({
                "record_index": index,
                "field": "service_date",
                "error": "服务日期不能为未来"
            })
        
        # 时长验证
        if record.duration_hours <= 0 or record.duration_hours > 24:
            results["data_type_errors"].append({
                "record_index": index,
                "field": "duration_hours",
                "error": "服务时长超出合理范围"
            })
        
        # 叙述长度验证
        if len(record.narrative_notes) < 50 or len(record.narrative_notes) > 500:
            results["data_type_errors"].append({
                "record_index": index,
                "field": "narrative_notes",
                "error": "叙述长度不在要求范围内"
            })
    
    def check_data_consistency(self, records: List[CarerServiceRecord]) -> Dict[str, Any]:
        """检查数据一致性"""
        results = {
            "duplicate_records": [],
            "inconsistent_patterns": [],
            "outliers": [],
            "temporal_consistency": {}
        }
        
        # 检查重复记录
        seen_combinations = set()
        for i, record in enumerate(records):
            combo = (record.carer_id, record.participant_id, record.service_date, record.service_type.value)
            if combo in seen_combinations:
                results["duplicate_records"].append({
                    "record_index": i,
                    "combination": combo
                })
            seen_combinations.add(combo)
        
        # 检查时长异常值
        durations = [r.duration_hours for r in records]
        if durations:
            q1, q3 = np.percentile(durations, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, record in enumerate(records):
                if record.duration_hours < lower_bound or record.duration_hours > upper_bound:
                    results["outliers"].append({
                        "record_index": i,
                        "field": "duration_hours",
                        "value": record.duration_hours,
                        "expected_range": f"{lower_bound:.2f}-{upper_bound:.2f}"
                    })
        
        # 时间一致性检查
        dates = [r.service_date for r in records]
        if dates:
            date_range = max(dates) - min(dates)
            results["temporal_consistency"] = {
                "date_range_days": date_range.days,
                "earliest_date": min(dates).isoformat(),
                "latest_date": max(dates).isoformat(),
                "future_dates": sum(1 for d in dates if d > date.today())
            }
        
        return results


class PrivacyAnalyzer:
    """隐私分析器"""
    
    def __init__(self):
        self.sensitive_patterns = {
            "phone": r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "address": r'\b\d+\s+[A-Za-z\s]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b',
            "name_pattern": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            "medicare": r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{1}\b',
            "abn": r'\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b'
        }
    
    def analyze_privacy_risks(self, records: List[CarerServiceRecord]) -> Dict[str, Any]:
        """分析隐私风险"""
        results = {
            "total_records": len(records),
            "privacy_violations": [],
            "sensitive_data_found": {},
            "anonymization_score": 0,
            "risk_level": "LOW"
        }
        
        all_text = []
        violation_count = 0
        
        for i, record in enumerate(records):
            # 检查叙述内容
            text_to_check = record.narrative_notes
            if record.location_details:
                text_to_check += " " + record.location_details
            if record.participant_response:
                text_to_check += " " + record.participant_response
            
            all_text.append(text_to_check)
            
            # 检查敏感信息
            for pattern_name, pattern in self.sensitive_patterns.items():
                matches = re.findall(pattern, text_to_check)
                if matches:
                    violation_count += 1
                    results["privacy_violations"].append({
                        "record_index": i,
                        "pattern_type": pattern_name,
                        "matches": matches
                    })
                    
                    if pattern_name not in results["sensitive_data_found"]:
                        results["sensitive_data_found"][pattern_name] = 0
                    results["sensitive_data_found"][pattern_name] += len(matches)
        
        # 计算匿名化得分
        if records:
            anonymization_score = max(0, 100 - (violation_count / len(records)) * 100)
            results["anonymization_score"] = round(anonymization_score, 2)
            
            # 确定风险级别
            if anonymization_score >= 95:
                results["risk_level"] = "LOW"
            elif anonymization_score >= 80:
                results["risk_level"] = "MEDIUM"
            else:
                results["risk_level"] = "HIGH"
        
        # 文本唯一性分析
        results["text_uniqueness"] = self._analyze_text_uniqueness(all_text)
        
        return results
    
    def _analyze_text_uniqueness(self, texts: List[str]) -> Dict[str, Any]:
        """分析文本唯一性"""
        if not texts:
            return {"unique_ratio": 0, "similarity_analysis": {}}
        
        # 计算唯一文本比例
        unique_texts = set(texts)
        unique_ratio = len(unique_texts) / len(texts)
        
        # 相似度分析（样本）
        similarity_analysis = {}
        if len(texts) > 1:
            # 随机抽样进行相似度分析
            sample_size = min(100, len(texts))
            sample_texts = texts[:sample_size]
            
            try:
                vectorizer = TfidfVectorizer(max_features=1000)
                tfidf_matrix = vectorizer.fit_transform(sample_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # 计算平均相似度（排除对角线）
                mask = np.ones(similarity_matrix.shape, dtype=bool)
                np.fill_diagonal(mask, False)
                avg_similarity = similarity_matrix[mask].mean()
                
                similarity_analysis = {
                    "average_similarity": round(avg_similarity, 4),
                    "max_similarity": round(similarity_matrix[mask].max(), 4),
                    "sample_size": sample_size
                }
            except Exception as e:
                logger.warning(f"相似度分析失败: {e}")
        
        return {
            "unique_ratio": round(unique_ratio, 4),
            "total_texts": len(texts),
            "unique_texts": len(unique_texts),
            "similarity_analysis": similarity_analysis
        }


class UtilityAnalyzer:
    """效用性分析器"""
    
    def analyze_data_utility(self, records: List[CarerServiceRecord]) -> Dict[str, Any]:
        """分析数据效用性"""
        results = {
            "statistical_analysis": {},
            "distribution_analysis": {},
            "coverage_analysis": {},
            "realism_score": 0
        }
        
        if not records:
            return results
        
        # 统计分析
        results["statistical_analysis"] = self._statistical_analysis(records)
        
        # 分布分析
        results["distribution_analysis"] = self._distribution_analysis(records)
        
        # 覆盖范围分析
        results["coverage_analysis"] = self._coverage_analysis(records)
        
        # 真实性评分
        results["realism_score"] = self._calculate_realism_score(records)
        
        return results
    
    def _statistical_analysis(self, records: List[CarerServiceRecord]) -> Dict[str, Any]:
        """统计分析"""
        durations = [r.duration_hours for r in records]
        narrative_lengths = [len(r.narrative_notes) for r in records]
        
        return {
            "duration_stats": {
                "mean": round(statistics.mean(durations), 2),
                "median": round(statistics.median(durations), 2),
                "std": round(statistics.stdev(durations) if len(durations) > 1 else 0, 2),
                "min": min(durations),
                "max": max(durations)
            },
            "narrative_length_stats": {
                "mean": round(statistics.mean(narrative_lengths), 2),
                "median": round(statistics.median(narrative_lengths), 2),
                "std": round(statistics.stdev(narrative_lengths) if len(narrative_lengths) > 1 else 0, 2),
                "min": min(narrative_lengths),
                "max": max(narrative_lengths)
            }
        }
    
    def _distribution_analysis(self, records: List[CarerServiceRecord]) -> Dict[str, Any]:
        """分布分析"""
        # 服务类型分布
        service_types = [r.service_type.value for r in records]
        service_dist = dict(Counter(service_types))
        
        # 结果分布
        outcomes = [r.service_outcome.value for r in records if r.service_outcome]
        outcome_dist = dict(Counter(outcomes))
        
        # 地点分布
        locations = [r.location_type.value for r in records if r.location_type]
        location_dist = dict(Counter(locations))
        
        # 时间分布
        dates = [r.service_date for r in records]
        weekdays = [d.weekday() for d in dates]
        weekday_dist = dict(Counter(weekdays))
        
        return {
            "service_types": service_dist,
            "outcomes": outcome_dist,
            "locations": location_dist,
            "weekdays": {str(k): v for k, v in weekday_dist.items()}
        }
    
    def _coverage_analysis(self, records: List[CarerServiceRecord]) -> Dict[str, Any]:
        """覆盖范围分析"""
        unique_carers = len(set(r.carer_id for r in records))
        unique_participants = len(set(r.participant_id for r in records))
        unique_service_types = len(set(r.service_type for r in records))
        unique_locations = len(set(r.location_type for r in records if r.location_type))
        
        # 时间覆盖
        dates = [r.service_date for r in records]
        time_span = (max(dates) - min(dates)).days if dates else 0
        
        return {
            "unique_carers": unique_carers,
            "unique_participants": unique_participants,
            "unique_service_types": unique_service_types,
            "unique_locations": unique_locations,
            "time_span_days": time_span,
            "records_per_carer": round(len(records) / unique_carers, 2) if unique_carers > 0 else 0,
            "records_per_participant": round(len(records) / unique_participants, 2) if unique_participants > 0 else 0
        }
    
    def _calculate_realism_score(self, records: List[CarerServiceRecord]) -> float:
        """计算真实性评分"""
        score = 100.0
        
        # 检查数据多样性
        service_types = [r.service_type.value for r in records]
        type_diversity = len(set(service_types)) / len(ServiceType)
        if type_diversity < 0.5:
            score -= 20
        
        # 检查时长合理性
        durations = [r.duration_hours for r in records]
        avg_duration = statistics.mean(durations)
        if avg_duration < 0.5 or avg_duration > 8:
            score -= 15
        
        # 检查叙述质量
        narratives = [r.narrative_notes for r in records]
        avg_length = statistics.mean([len(n) for n in narratives])
        if avg_length < 100 or avg_length > 400:
            score -= 10
        
        # 检查结果分布合理性
        outcomes = [r.service_outcome.value for r in records if r.service_outcome]
        if outcomes:
            positive_ratio = outcomes.count("positive") / len(outcomes)
            if positive_ratio < 0.4 or positive_ratio > 0.8:
                score -= 10
        
        return max(0, round(score, 2))


class ComprehensiveValidator:
    """综合验证器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.quality_validator = DataQualityValidator(config)
        self.privacy_analyzer = PrivacyAnalyzer()
        self.utility_analyzer = UtilityAnalyzer()
    
    def comprehensive_validation(self, records: List[CarerServiceRecord]) -> Dict[str, Any]:
        """进行综合验证"""
        logger.info(f"开始综合验证 {len(records)} 条记录")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(records),
            "quality_validation": {},
            "privacy_analysis": {},
            "utility_analysis": {},
            "overall_score": 0,
            "recommendations": []
        }
        
        try:
            # 质量验证
            logger.info("执行质量验证...")
            results["quality_validation"] = {
                "schema_compliance": self.quality_validator.validate_schema_compliance(records),
                "data_consistency": self.quality_validator.check_data_consistency(records)
            }
            
            # 隐私分析
            logger.info("执行隐私分析...")
            results["privacy_analysis"] = self.privacy_analyzer.analyze_privacy_risks(records)
            
            # 效用性分析
            logger.info("执行效用性分析...")
            results["utility_analysis"] = self.utility_analyzer.analyze_data_utility(records)
            
            # 计算总体评分
            results["overall_score"] = self._calculate_overall_score(results)
            
            # 生成建议
            results["recommendations"] = self._generate_recommendations(results)
            
            logger.info("综合验证完成")
            
        except Exception as e:
            logger.error(f"验证过程出错: {e}")
            results["error"] = str(e)
        
        return results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """计算总体评分"""
        scores = []
        
        # 质量得分 (40%)
        quality_score = 100
        schema_errors = len(results["quality_validation"]["schema_compliance"].get("validation_errors", []))
        consistency_issues = len(results["quality_validation"]["data_consistency"].get("duplicate_records", []))
        quality_score -= (schema_errors + consistency_issues) * 2
        scores.append(("quality", max(0, quality_score), 0.4))
        
        # 隐私得分 (30%)
        privacy_score = results["privacy_analysis"].get("anonymization_score", 0)
        scores.append(("privacy", privacy_score, 0.3))
        
        # 效用性得分 (30%)
        utility_score = results["utility_analysis"].get("realism_score", 0)
        scores.append(("utility", utility_score, 0.3))
        
        # 加权平均
        weighted_sum = sum(score * weight for _, score, weight in scores)
        return round(weighted_sum, 2)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 质量建议
        schema_errors = len(results["quality_validation"]["schema_compliance"].get("validation_errors", []))
        if schema_errors > 0:
            recommendations.append(f"修复 {schema_errors} 个模式验证错误")
        
        # 隐私建议
        privacy_score = results["privacy_analysis"].get("anonymization_score", 100)
        if privacy_score < 95:
            recommendations.append("提高数据匿名化程度，移除或模糊敏感信息")
        
        # 效用性建议
        utility_score = results["utility_analysis"].get("realism_score", 100)
        if utility_score < 80:
            recommendations.append("改进数据真实性，确保服务时长和结果分布合理")
        
        # 覆盖范围建议
        coverage = results["utility_analysis"].get("coverage_analysis", {})
        if coverage.get("unique_service_types", 0) < len(ServiceType) * 0.7:
            recommendations.append("增加服务类型的多样性")
        
        return recommendations
    
    def save_validation_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """保存验证报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"
        
        output_dir = Path(self.config["output"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / filename
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"验证报告已保存: {report_file}")
        return str(report_file)


if __name__ == "__main__":
    # 测试验证器
    from carer_data_schema import CarerServiceRecord, ServiceType
    from datetime import date
    
    # 创建测试数据
    test_records = [
        CarerServiceRecord(
            record_id="SR12345678",
            carer_id="CR123456",
            participant_id="PT654321",
            service_date=date.today() - timedelta(days=1),
            service_type=ServiceType.PERSONAL_CARE,
            duration_hours=2.5,
            narrative_notes="为参与者提供个人护理支持，协助洗漱和穿衣。参与者配合度良好，完成了既定目标。护工使用了耐心引导的方法。"
        )
    ]
    
    validator = ComprehensiveValidator()
    results = validator.comprehensive_validation(test_records)
    
    print("验证结果:")
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

