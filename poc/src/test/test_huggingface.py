"""
Hugging Face API 测试脚本
使用免费的Hugging Face API生成护工数据
"""

import asyncio
import json
import logging
import requests
from typing import Dict, Any
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从free_config获取token
from free_config import FREE_LLM_CONFIG

async def test_huggingface_api():
    """测试Hugging Face API连接"""
    
    hf_config = FREE_LLM_CONFIG["huggingface"]
    token = hf_config["token"]
    
    if not token or token == "your_huggingface_token_here":
        print("❌ 错误：Hugging Face token未配置")
        print("请在 free_config.py 中设置正确的token")
        return False
    
    # 测试API连接
    api_url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    test_data = {
        "inputs": "The carer provided support to",
        "parameters": {
            "max_length": 100,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        print("🔄 测试Hugging Face API连接...")
        response = requests.post(api_url, headers=headers, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Hugging Face API连接成功!")
            print(f"📝 示例生成: {result}")
            return True
        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 连接错误: {e}")
        return False

async def generate_with_huggingface(prompt: str) -> str:
    """使用Hugging Face生成文本"""
    
    hf_config = FREE_LLM_CONFIG["huggingface"]
    token = hf_config["token"]
    
    # 使用GPT-2进行文本生成
    api_url = "https://api-inference.huggingface.co/models/gpt2"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "inputs": prompt,
        "parameters": {
            "max_length": 150,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False,
            "pad_token_id": 50256  # GPT-2的pad token
        }
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                return generated_text.strip()
            return ""
        else:
            logger.error(f"Hugging Face API错误: {response.status_code} - {response.text}")
            return ""
            
    except Exception as e:
        logger.error(f"Hugging Face调用失败: {e}")
        return ""

async def generate_carer_narrative():
    """生成护工服务叙述"""
    
    prompts = [
        "The carer provided personal care support to the participant, assisting with daily hygiene activities.",
        "Today the support worker helped the client with community access activities.",
        "The disability support worker delivered household assistance services.",
        "Professional care was provided to assist the participant with transport needs.",
        "The carer facilitated social support activities for the participant."
    ]
    
    print("\n🎯 生成护工服务叙述示例:")
    print("=" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. 输入提示: {prompt}")
        
        generated = await generate_with_huggingface(prompt)
        
        if generated:
            # 组合完整叙述
            full_narrative = f"{prompt} {generated}"
            print(f"   生成结果: {full_narrative}")
        else:
            print("   ❌ 生成失败")

async def test_structured_generation():
    """测试结构化数据生成"""
    
    print("\n🏗️ 测试结构化护工记录生成:")
    print("=" * 50)
    
    # 护工服务记录模板
    service_types = [
        "Personal Care", "Community Access", "Household Tasks", 
        "Transport Assistance", "Social Support"
    ]
    
    outcomes = ["positive", "neutral", "negative"]
    
    for service_type in service_types[:2]:  # 测试前两个
        for outcome in outcomes[:2]:  # 测试前两个结果
            
            prompt = f"The carer provided {service_type.lower()} support with a {outcome} outcome. The participant"
            
            print(f"\n🔸 服务类型: {service_type}")
            print(f"🔸 预期结果: {outcome}")
            print(f"🔸 提示: {prompt}")
            
            generated = await generate_with_huggingface(prompt)
            
            if generated:
                full_text = f"{prompt} {generated}"
                print(f"✅ 生成: {full_text[:100]}...")
                
                # 模拟创建JSON记录
                record = {
                    "service_type": service_type,
                    "service_outcome": outcome,
                    "narrative_notes": full_text[:200],  # 限制长度
                    "generated_at": datetime.now().isoformat()
                }
                
                print(f"📋 记录: {json.dumps(record, indent=2)}")
            else:
                print("❌ 生成失败")

async def main():
    """主函数"""
    print("🤗 Hugging Face API 测试程序")
    print("=" * 50)
    
    # 1. 测试API连接
    if not await test_huggingface_api():
        print("\n💡 解决方案:")
        print("1. 访问 https://huggingface.co/settings/tokens")
        print("2. 创建新的 Read 权限 token")
        print("3. 在 free_config.py 中更新 token")
        return
    
    # 2. 生成叙述示例
    await generate_carer_narrative()
    
    # 3. 测试结构化生成
    await test_structured_generation()
    
    print("\n🎉 测试完成!")
    print("您现在可以使用Hugging Face API进行数据生成了。")

if __name__ == "__main__":
    asyncio.run(main())

