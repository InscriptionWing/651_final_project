# 免费LLM设置指南

本指南将帮助您设置完全免费的LLM数据生成环境，无需任何付费API。

## 🎯 可用的免费选项

1. **Ollama本地模型** ⭐ **推荐** - 完全免费，无限制
2. **Hugging Face免费API** - 有使用限制但质量好
3. **基于模板的生成** - 本地运行，速度快
4. **基于规则的生成** - 最简单，质量稳定

## 🚀 方法一：Ollama本地模型（推荐）

### 优势
- ✅ 完全免费，无使用限制
- ✅ 本地运行，隐私安全
- ✅ 支持中英文模型
- ✅ 一次安装，永久使用

### 安装步骤

#### 1. 下载并安装Ollama

**Windows:**
```bash
# 访问 https://ollama.ai/download
# 下载Windows安装包并安装
```

**macOS:**
```bash
# 使用Homebrew安装
brew install ollama

# 或下载安装包
# https://ollama.ai/download
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### 2. 启动Ollama服务

```bash
# 启动Ollama服务（Windows会自动启动）
ollama serve
```

#### 3. 安装推荐模型

```bash
# 安装Llama 2（英文，3.8GB）
ollama pull llama2

# 可选：安装中文模型通义千问（4.0GB）
ollama pull qwen:7b

# 可选：安装Mistral（代码能力强，4.1GB）
ollama pull mistral
```

#### 4. 验证安装

```bash
# 检查已安装的模型
ollama list

# 测试模型
ollama run llama2
```

### 配置项目使用Ollama

编辑 `free_config.py`：

```python
FREE_LLM_CONFIG = {
    "ollama": {
        "enabled": True,
        "base_url": "http://localhost:11434",
        "models": {
            "primary": "llama2",        # 或您安装的其他模型
            "chinese": "qwen:7b"        # 如果需要中文支持
        }
    }
}
```

## 🤗 方法二：Hugging Face免费API

### 优势
- ✅ 免费获取API密钥
- ✅ 多种预训练模型
- ✅ 无需本地存储空间

### 限制
- ⚠️ 每月有API调用限制
- ⚠️ 需要网络连接

### 设置步骤

#### 1. 注册Hugging Face账号

访问 [https://huggingface.co/](https://huggingface.co/) 注册免费账号

#### 2. 获取API Token

1. 登录后进入 **Settings** → **Access Tokens**
2. 点击 **New token**
3. 选择 **Read** 权限
4. 复制生成的token

#### 3. 配置项目

在 `free_config.py` 中替换token：

```python
"huggingface": {
    "enabled": True,
    "token": "hf_xxxxxxxxxxxxxxxxxxxx",  # 替换为您的token
}
```

## 📝 方法三：基于模板的生成（无需LLM）

### 优势
- ✅ 完全本地运行
- ✅ 速度极快
- ✅ 质量稳定
- ✅ 无任何依赖

### 配置

这个方法默认启用，无需额外配置。您可以在 `templates_enhanced.txt` 中自定义模板。

## ⚡ 快速开始

### 检查可用服务

```bash
cd D:\651\poc\new
python free_config.py
```

### 运行免费数据生成器

```bash
# 使用免费LLM生成数据
python free_llm_generator.py

# 或使用主程序（会自动选择最佳免费方法）
python main.py --size 100 --free-mode
```

## 🔧 故障排除

### Ollama相关问题

**问题：无法连接到Ollama服务**
```bash
# 检查服务是否运行
curl http://localhost:11434/api/tags

# 重启服务
ollama serve
```

**问题：模型下载缓慢**
```bash
# 使用国内镜像（如果可用）
export OLLAMA_HOST=0.0.0.0:11434
```

**问题：内存不足**
```bash
# 使用更小的模型
ollama pull llama2:7b-chat-q4_0  # 量化版本，更小
```

### Hugging Face问题

**问题：API调用失败**
- 检查token是否正确
- 确认网络连接
- 检查是否超出免费配额

**问题：响应速度慢**
- 减少batch_size
- 使用更轻量的模型

## 📊 性能比较

| 方法 | 速度 | 质量 | 成本 | 设置难度 |
|------|------|------|------|----------|
| Ollama | ⭐⭐⭐ | ⭐⭐⭐⭐ | 免费 | ⭐⭐ |
| Hugging Face | ⭐⭐ | ⭐⭐⭐⭐ | 免费* | ⭐ |
| 基于模板 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 免费 | ⭐ |
| 基于规则 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 免费 | ⭐ |

*有使用限制

## 🎯 推荐配置

### 个人使用（推荐配置）

```python
# 优先级顺序
AUTO_SELECTION_STRATEGY = {
    "priority_order": [
        "ollama",           # 首选：本地Ollama
        "template_based",   # 次选：基于模板
        "rule_based",       # 保底：基于规则
    ]
}
```

### 轻量级使用

```python
# 如果只需要快速生成数据
AUTO_SELECTION_STRATEGY = {
    "priority_order": [
        "template_based",   # 基于模板
        "rule_based",       # 基于规则
    ]
}
```

### 高质量使用

```python
# 如果追求最高质量
AUTO_SELECTION_STRATEGY = {
    "priority_order": [
        "ollama",           # Ollama + 大模型
        "huggingface",      # HF API
        "template_based",   # 模板兜底
    ]
}
```

## 📋 使用示例

### 生成100条记录（Ollama）

```bash
python free_llm_generator.py
```

### 生成1000条记录（混合方法）

```python
from free_llm_generator import FreeLLMDataGenerator
import asyncio

async def generate_large_dataset():
    generator = FreeLLMDataGenerator()
    records = await generator.generate_dataset(1000)
    saved_files = generator.save_dataset(records)
    print(f"生成了 {len(records)} 条记录")
    return records

records = asyncio.run(generate_large_dataset())
```

## 🔄 升级路径

1. **开始**: 使用基于模板/规则的方法快速体验
2. **提升**: 安装Ollama本地模型获得更好质量
3. **优化**: 配置Hugging Face API作为备选
4. **扩展**: 根据需要安装更多Ollama模型

## 💡 小贴士

1. **存储空间**: Ollama模型需要3-5GB存储空间
2. **内存要求**: 运行7B模型建议8GB+内存
3. **网络**: 首次下载模型需要稳定网络
4. **备份**: 定期备份生成的数据
5. **模板**: 可以自定义模板以提高数据质量

## 🆘 获取帮助

如果遇到问题：

1. 查看项目logs目录下的日志文件
2. 运行 `python free_config.py` 检查配置
3. 确认所有依赖已正确安装
4. 检查网络连接（如使用在线API）

---

通过以上任一方法，您都可以完全免费地生成高质量的护工数据！推荐从Ollama开始，它提供了最佳的质量与成本平衡。

