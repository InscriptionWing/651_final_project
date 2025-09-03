"""
调试Ollama生成API调用
专门用于诊断404错误
"""

import requests
import json
import time

def debug_ollama_generation():
    """调试Ollama生成API"""
    print("🔍 调试Ollama生成API调用...")
    
    # 1. 首先获取模型列表
    try:
        models_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if models_response.status_code == 200:
            models_data = models_response.json()
            models = models_data.get("models", [])
            
            print(f"✅ 检测到模型: {len(models)}")
            for model in models:
                print(f"   模型名称: '{model['name']}'")
                print(f"   模型大小: {model.get('size', 'unknown')}")
                print(f"   完整信息: {model}")
                print("-" * 40)
            
            if not models:
                print("❌ 没有检测到模型")
                return
            
            # 2. 尝试不同的API端点和参数组合
            test_model = models[0]['name']
            print(f"\n🧪 测试模型: '{test_model}'")
            
            # 测试不同的API调用方式
            test_cases = [
                {
                    "name": "标准generate API",
                    "url": "http://localhost:11434/api/generate",
                    "data": {
                        "model": test_model,
                        "prompt": "Hello, please respond in English.",
                        "stream": False
                    }
                },
                {
                    "name": "简化参数",
                    "url": "http://localhost:11434/api/generate", 
                    "data": {
                        "model": test_model,
                        "prompt": "Hello"
                    }
                },
                {
                    "name": "chat API",
                    "url": "http://localhost:11434/api/chat",
                    "data": {
                        "model": test_model,
                        "messages": [
                            {"role": "user", "content": "Hello, please respond in English."}
                        ],
                        "stream": False
                    }
                }
            ]
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n{i}. 测试: {test_case['name']}")
                print(f"   URL: {test_case['url']}")
                print(f"   Data: {json.dumps(test_case['data'], indent=2)}")
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        test_case['url'],
                        json=test_case['data'],
                        timeout=60
                    )
                    end_time = time.time()
                    
                    print(f"   状态码: {response.status_code}")
                    print(f"   响应时间: {end_time - start_time:.2f}秒")
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"   ✅ 成功!")
                        
                        # 根据API类型提取响应
                        if 'response' in result:
                            content = result['response']
                        elif 'message' in result:
                            content = result['message'].get('content', '')
                        else:
                            content = str(result)
                        
                        print(f"   响应内容: {content[:100]}...")
                        print(f"   完整响应: {json.dumps(result, indent=2)[:200]}...")
                        
                        return {
                            "success": True,
                            "working_config": test_case,
                            "response": result
                        }
                    else:
                        print(f"   ❌ 失败: {response.status_code}")
                        print(f"   错误内容: {response.text[:200]}")
                        
                except requests.exceptions.Timeout:
                    print(f"   ❌ 超时")
                except Exception as e:
                    print(f"   ❌ 异常: {e}")
            
            print("\n❌ 所有测试用例都失败了")
            return {"success": False}
            
        else:
            print(f"❌ 无法获取模型列表: {models_response.status_code}")
            return {"success": False}
            
    except Exception as e:
        print(f"❌ 调试过程中出现异常: {e}")
        return {"success": False}

def test_ollama_basic_endpoints():
    """测试Ollama基础端点"""
    print("\n🔧 测试Ollama基础端点...")
    
    endpoints = [
        "http://localhost:11434",
        "http://localhost:11434/api/version",
        "http://localhost:11434/api/tags",
        "http://localhost:11434/api/ps"  # 显示运行中的模型
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            print(f"✅ {endpoint}: {response.status_code}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   数据: {json.dumps(data, indent=2)[:200]}...")
                except:
                    print(f"   文本: {response.text[:200]}...")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")

if __name__ == "__main__":
    print("🦙 Ollama API 调试工具")
    print("=" * 50)
    
    # 测试基础端点
    test_ollama_basic_endpoints()
    
    # 调试生成API
    result = debug_ollama_generation()
    
    if result and result.get("success"):
        print(f"\n✅ 找到工作的API配置!")
        working_config = result["working_config"]
        print(f"📋 推荐使用配置:")
        print(f"   URL: {working_config['url']}")
        print(f"   Data format: {json.dumps(working_config['data'], indent=2)}")
    else:
        print(f"\n❌ 无法找到工作的API配置")
        print("💡 建议:")
        print("1. 确认Ollama服务正在运行: ollama serve")
        print("2. 检查模型是否正确加载: ollama list")
        print("3. 尝试手动测试: ollama run <model_name>")
    
    print("\n" + "=" * 50)

