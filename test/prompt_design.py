from langchain_core.prompts import PromptTemplate

# 定义通用 JSON 输出模板
template = """
请生成一条 Carer 数据记录，包含以下字段：
- CarerID: 一个唯一的字符串
- ParticipantID: 服务对象的 ID
- ServiceDate: 格式 YYYY-MM-DD
- ServiceType: 枚举 ["Home Care","Respite","Day Program"]
- Duration: 服务时长（小时，可带小数）
- NarrativeNotes: 简短文本说明

输出仅为合法的 JSON 对象格式。
"""
prompt = PromptTemplate.from_template(template) #PROMPT_TEMPLATE  # 参考示例及 Analytics Vidhya 报导 :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}