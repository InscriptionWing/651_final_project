"""
Ollama API测试脚本
用于验证Ollama服务是否正常运行，以及检测可用模型
"""

import requests
import json
import time

def test_ollama_connection():
    """测试Ollama连接"""
    print("🔄 测试Ollama API连接...")
    
    try:
        # 测试基本连接
        response = requests.get("http://localhost:11434", timeout=5)
        print(f"✅ Ollama服务响应: {response.status_code}")
        
        # 获取可用模型列表
        models_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if models_response.status_code == 200:
            models_data = models_response.json()
            models = models_data.get("models", [])
            
            if models:
                print(f"📦 检测到 {len(models)} 个模型:")
                for model in models:
                    print(f"   - {model['name']} (大小: {model.get('size', 'unknown')})")
                
                # 测试第一个模型的生成功能
                test_model = models[0]['name']
                print(f"\n🧪 测试模型 '{test_model}' 的生成功能...")
                
                test_prompt = "请用中文简单介绍一下NDIS护工服务。"
                
                generation_data = {
                    "model": test_model,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 100
                    }
                }
                
                start_time = time.time()
                gen_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=generation_data,
                    timeout=60
                )
                end_time = time.time()
                
                if gen_response.status_code == 200:
                    result = gen_response.json()
                    generated_text = result.get("response", "").strip()
                    
                    print(f"✅ 生成成功 (耗时: {end_time - start_time:.2f}秒)")
                    print(f"📝 生成内容: {generated_text[:200]}...")
                    
                    return {
                        "status": "success",
                        "models": [m['name'] for m in models],
                        "test_model": test_model,
                        "generation_time": end_time - start_time,
                        "generated_text": generated_text
                    }
                else:
                    print(f"❌ 生成失败: HTTP {gen_response.status_code}")
                    print(f"错误信息: {gen_response.text}")
                    return {"status": "generation_failed", "error": gen_response.text}
            else:
                print("❌ 没有检测到任何模型")
                print("💡 请先安装模型，例如: ollama pull llama2")
                return {"status": "no_models"}
        else:
            print(f"❌ 获取模型列表失败: {models_response.status_code}")
            return {"status": "models_list_failed", "error": models_response.text}
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Ollama服务")
        print("💡 请确保Ollama已安装并运行:")
        print("   1. 下载Ollama: https://ollama.ai/download")
        print("   2. 启动服务: ollama serve")
        print("   3. 安装模型: ollama pull llama2")
        return {"status": "connection_failed"}
    
    except requests.exceptions.Timeout:
        print("❌ 连接超时")
        return {"status": "timeout"}
    
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return {"status": "error", "error": str(e)}

def recommend_setup_steps():
    """推荐设置步骤"""
    print("\n📋 Ollama设置建议:")
    print("1. 确保Ollama服务正在运行: ollama serve")
    print("2. 推荐安装以下模型之一:")
    print("   - ollama pull llama2         # 通用模型，3.8GB")
    print("   - ollama pull mistral        # 高质量模型，4.1GB")
    print("   - ollama pull qwen:7b        # 中文支持更好，4.0GB")
    print("3. 验证安装: ollama list")
    print("4. 测试生成: ollama run llama2")

if __name__ == "__main__":
    print("🦙 Ollama API 测试程序")
    print("=" * 50)
    
    result = test_ollama_connection()
    
    if result["status"] == "success":
        print(f"\n✅ Ollama API测试通过!")
        print(f"🎯 推荐使用模型: {result['test_model']}")
        print(f"⚡ 生成速度: {result['generation_time']:.2f}秒")
    else:
        print(f"\n❌ Ollama API测试失败: {result['status']}")
        recommend_setup_steps()
    
    print("\n" + "=" * 50)

