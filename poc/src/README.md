# NDIS护工数据生成器 (NDIS Carers Data Generator)

## 📋 项目概述

这是一个专为澳大利亚国家残疾保险计划(NDIS)设计的护工服务数据合成生成器。该项目使用本地LLM技术生成高质量、符合隐私要求的英文护工服务记录，用于安全的数据分析、系统测试和业务分析，无需使用真实参与者的敏感数据。

## 🎯 核心功能

### ✅ 已实现功能
- **🤖 纯LLM数据生成**: 使用Ollama本地LLM完全生成护工服务记录
- **🌍 英文输出**: 所有生成内容均为英文，符合国际标准
- **👤 完整护工信息**: 包含护工姓名、ID和详细服务记录
- **📊 多格式导出**: 支持JSON、JSONL、CSV格式
- **✅ 数据验证**: 内置数据质量验证和统计分析
- **🔒 隐私保护**: 生成的数据完全合成，不包含真实个人信息

### 📈 生成的数据包含
- 护工基本信息（姓名、ID）
- 参与者信息（去标识化）
- 服务类型和时长
- 详细的服务叙述
- 支持技术和挑战记录
- 参与者反应和后续需求
- NDIS计费代码和监督记录

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd D:\651\poc\new

# 安装依赖
pip install -r requirements.txt

# 设置Ollama（如果还未安装）
# 访问 https://ollama.ai 下载并安装Ollama
# 运行一个适合的模型，例如：
ollama pull gpt-oss:20b
```

### 2. 生成数据

```bash
# 生成20条英文护工记录
python main_english.py --size 20

# 生成100条记录并跳过验证（更快）
python main_english.py --size 100 --no-validate

# 验证现有数据文件
python main_english.py --validate-file output/your_data_file.json
```

### 3. 查看结果

生成的数据将保存在 `output/` 目录中，包含三种格式：
- `*.json` - 标准JSON格式
- `*.jsonl` - 每行一个JSON记录
- `*.csv` - CSV表格格式

## 📁 项目结构

```
D:\651\poc\new\
├── README.md                          # 项目文档
├── requirements.txt                   # 依赖包列表
├── main_english.py                   # 主程序入口
├── pure_llm_english_generator.py     # 纯LLM英文生成器
├── english_data_schema.py            # 英文数据结构定义
├── config.py                         # 项目配置
├── output/                           # 输出目录
├── logs/                             # 日志目录
├── dashboard/                        # 数据看板（可选）
└── *.py                              # 其他支持模块
```

## 🔧 核心组件

### 数据生成器
- **PureLLMEnglishGenerator**: 纯LLM驱动的英文数据生成器
- **EnglishTemplateGenerator**: 模板驱动的英文数据生成器（备选）

### 数据验证
- **EnglishDataValidator**: 英文数据质量验证器
- 支持统计分析、分布检查、完整性验证

### 数据结构
- **CarerServiceRecord**: 护工服务记录主结构
- **CarerProfile**: 护工档案信息
- **ParticipantProfile**: 参与者档案信息

## 📊 生成示例

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

## ⚙️ 配置选项

### 命令行参数
- `--size`: 生成记录数量 (默认: 100)
- `--no-validate`: 跳过数据验证以提升速度
- `--validate-file`: 验证现有数据文件
- `--output-formats`: 指定输出格式 (json, jsonl, csv)

### 环境配置
- 确保Ollama服务在 `localhost:11434` 运行
- 项目会自动检测可用的LLM模型

## 📈 性能指标

### 典型生成性能
- **生成速度**: 约2-5秒/记录（取决于LLM性能）
- **成功率**: >95%（网络稳定情况下）
- **数据质量**: 平均叙述长度730字符，内容丰富真实

### 推荐配置
- **小批量测试**: 10-20条记录
- **常规使用**: 50-100条记录  
- **大批量生成**: 500+条记录（建议分批进行）

## 🔍 数据验证

生成的数据会自动进行以下验证：
- ✅ 字段完整性检查
- ✅ 数据类型验证
- ✅ 叙述内容长度验证
- ✅ 服务类型分布分析
- ✅ 时间范围合理性检查
- ✅ 护工-参与者分配统计

## 🎨 可选功能

### 数据看板
项目包含一个可选的Streamlit数据看板：

```bash
cd dashboard
python run_dashboard.py --mode simple
```

### 数据聚合分析
```bash
python dashboard/data_aggregator.py
```

## 🛠️ 故障排除

### 常见问题

1. **Ollama连接失败**
   ```bash
   # 检查Ollama服务状态
   curl http://localhost:11434/api/tags
   
   # 重启Ollama服务
   ollama serve
   ```

2. **生成超时**
   - 增加超时时间设置
   - 检查网络连接
   - 使用更小的批量大小

3. **内存不足**
   - 减少生成批量大小
   - 关闭不必要的应用程序

## 📝 开发指南

### 添加新的服务类型
1. 编辑 `english_data_schema.py` 中的 `ServiceType` 枚举
2. 更新 `pure_llm_english_generator.py` 中的LLM提示模板
3. 测试新服务类型的生成效果

### 自定义验证规则
1. 修改 `EnglishDataValidator` 类
2. 添加新的验证方法
3. 更新输出格式

## 📄 许可证

本项目仅用于教育和研究目的。

## 🤝 贡献

欢迎提交问题报告和功能建议！

## 📞 支持

如需技术支持，请查看：
1. 项目文档和示例
2. 常见问题解答
3. 错误日志分析

---

**⚠️ 重要提醒**: 
- 生成的数据仅为合成数据，不应用于生产环境
- 请确保遵守当地数据保护法规
- 定期更新LLM模型以保持数据质量