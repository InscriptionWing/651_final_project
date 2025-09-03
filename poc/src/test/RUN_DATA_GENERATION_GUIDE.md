# 🤖 NDIS 护工数据生成系统 - 运行指南

## 🎯 概述

现在您的仪表板已经运行，接下来需要生成真实的护工服务记录数据来填充仪表板。

## 📁 数据生成系统位置

数据生成系统位于项目根目录：
```
D:\651\poc\new\
├── main.py                    # 主数据生成程序
├── main_english.py           # 英文版数据生成程序  
├── demo_generator.py         # 演示数据生成器
├── llm_data_generator.py     # LLM驱动的数据生成器
├── free_llm_generator.py     # 免费LLM数据生成器
└── carer_data_schema.py      # 数据模式定义
```

## 🚀 快速开始 - 推荐方式

### 方式1：演示数据生成（最简单）

```bash
# 切换到项目根目录
cd D:\651\poc\new

# 生成100条演示数据
python demo_generator.py

# 或指定数量
python demo_generator.py --size 200
```

### 方式2：使用主程序（标准方式）

```bash
# 切换到项目根目录  
cd D:\651\poc\new

# 生成100条记录（演示模式，无需API）
python main.py --size 100

# 生成更多记录
python main.py --size 500
```

### 方式3：英文版本（推荐用于仪表板）

```bash
# 切换到项目根目录
cd D:\651\poc\new

# 生成英文版本数据
python main_english.py --size 100
```

## 🔧 高级数据生成选项

### 使用LLM生成高质量数据

如果您有OpenAI API密钥：

```bash
# 设置API密钥（Windows）
set OPENAI_API_KEY=your_api_key_here

# 或在代码中配置，然后运行
python main.py --size 200 --validate
```

### 使用免费LLM服务

```bash
# 使用免费LLM服务
python free_llm_generator.py --size 100

# 检查可用的免费服务
python main.py --check-free-services
```

## 📊 数据生成后自动更新仪表板

### 自动方式：后台ETL调度器

```bash
# 在新的命令行窗口中运行
cd D:\651\poc\new\dashboard

# 启动后台ETL调度器（每5分钟自动更新）
python scheduler.py --daemon --interval 5
```

### 手动方式：手动刷新数据

```bash
# 在仪表板目录运行
cd D:\651\poc\new\dashboard

# 手动触发数据聚合
python data_aggregator.py

# 或使用API刷新
python -c "
from data_aggregator import DataAggregator
aggregator = DataAggregator()
result = aggregator.aggregate_all_data()
print(f'刷新完成: {result.get(\"status\", \"unknown\")}')
"
```

## 🔄 完整工作流程示例

### 1. 生成新数据
```bash
# 终端1：生成数据
cd D:\651\poc\new
python main_english.py --size 200
```

### 2. 更新仪表板数据
```bash  
# 终端2：更新仪表板
cd D:\651\poc\new\dashboard
python data_aggregator.py
```

### 3. 查看仪表板
- 打开浏览器访问：http://localhost:8501
- 点击仪表板中的 "🔄 Refresh Data" 按钮

## 📈 数据生成参数说明

### 主要参数

| 参数 | 描述 | 示例 |
|------|------|------|
| `--size` | 生成记录数量 | `--size 500` |
| `--validate` | 启用数据验证 | `--validate` |
| `--no-validate` | 跳过数据验证 | `--no-validate` |
| `--output-formats` | 输出格式 | `--output-formats json csv` |
| `--free-mode` | 使用免费LLM | `--free-mode` |

### 完整命令示例

```bash
# 生成500条记录，启用验证，输出JSON和CSV格式
python main.py --size 500 --validate --output-formats json csv

# 使用免费模式生成200条记录
python main.py --size 200 --free-mode --validate

# 英文版本，生成100条记录
python main_english.py --size 100 --output-formats json jsonl csv
```

## 📁 生成的文件位置

数据将保存在 `output/` 目录中：

```
D:\651\poc\new\output\
├── carers_synthetic_data_YYYYMMDD_HHMMSS_XXXrecords.json
├── carers_synthetic_data_YYYYMMDD_HHMMSS_XXXrecords.jsonl  
├── carers_synthetic_data_YYYYMMDD_HHMMSS_XXXrecords.csv
└── validation_report_XXXrecords.json
```

## 🎯 推荐的完整流程

### 第一次设置

```bash
# 1. 生成初始数据集
cd D:\651\poc\new
python main_english.py --size 200

# 2. 启动后台数据更新
cd dashboard  
python scheduler.py --daemon --interval 10 &

# 3. 仪表板应该已经在运行
# 访问：http://localhost:8501
```

### 日常使用

```bash
# 定期生成新数据（比如每天）
cd D:\651\poc\new
python main_english.py --size 100

# 仪表板会自动检测新数据并更新
# 或手动点击仪表板中的刷新按钮
```

## 🔍 验证数据生成是否成功

### 检查生成的文件
```bash
# 查看最新生成的文件
cd D:\651\poc\new
dir output\*latest*

# 或查看所有输出文件
dir output\
```

### 检查仪表板数据更新
1. 访问仪表板：http://localhost:8501
2. 查看 "Total Records" 数量是否增加
3. 检查 "Last Updated" 时间戳
4. 点击 "🔄 Refresh Data" 强制刷新

## ⚡ 快速测试脚本

创建一个测试脚本来验证完整流程：

```bash
# 创建测试脚本
cd D:\651\poc\new

# 运行快速测试
python -c "
import subprocess
import os

print('🧪 测试完整数据生成流程')

# 1. 生成测试数据
print('📊 生成测试数据...')
result = subprocess.run(['python', 'demo_generator.py', '--size', '50'], capture_output=True, text=True)
if result.returncode == 0:
    print('✅ 数据生成成功')
else:
    print('❌ 数据生成失败')
    print(result.stderr)

# 2. 更新仪表板数据  
print('🔄 更新仪表板数据...')
os.chdir('dashboard')
result = subprocess.run(['python', 'data_aggregator.py'], capture_output=True, text=True)
if result.returncode == 0:
    print('✅ 仪表板数据更新成功')
    print('🎉 完整流程测试通过！')
    print('📱 请访问 http://localhost:8501 查看更新后的仪表板')
else:
    print('❌ 仪表板数据更新失败')
    print(result.stderr)
"
```

## 🎊 总结

现在您有了完整的数据生成和监控解决方案：

1. **🤖 数据生成** - 使用 `main.py` 或 `main_english.py` 生成护工服务记录
2. **📊 数据监控** - 仪表板实时监控数据质量和流水线健康
3. **🔄 自动更新** - ETL调度器自动聚合新数据到仪表板
4. **📈 可视化分析** - 交互式图表和KPI监控

**立即开始生成数据，让您的仪表板展示真实的护工服务记录分析！** 🚀



