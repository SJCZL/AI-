#!/usr/bin/env python3
"""
DeepSeek API调试测试脚本
用于验证API连接、密钥有效性和响应格式
"""

import os
import sys
import requests
import json
from typing import Dict, Any

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_api_connection(api_key: str, base_url: str = "https://api.deepseek.com/v1") -> Dict[str, Any]:
    """测试DeepSeek API连接"""
    print("🔍 开始测试DeepSeek API连接...")
    
    results = {
        "api_key_provided": bool(api_key),
        "base_url": base_url,
        "connection_success": False,
        "model_test": False,
        "error": None,
        "raw_response": None
    }
    
    if not api_key:
        results["error"] = "未提供API密钥"
        return results
    
    try:
        # 测试1: 直接HTTP请求测试
        print("📡 测试1: 直接HTTP连接...")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 测试基础连接
        test_url = base_url.rstrip('/') + "/models"
        response = requests.get(test_url, headers=headers, timeout=10)
        
        print(f"🔗 请求URL: {test_url}")
        print(f"📊 HTTP状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ HTTP连接成功")
            models = response.json()
            print(f"📋 可用模型: {models}")
            results["connection_success"] = True
        else:
            print(f"❌ HTTP连接失败: {response.text}")
            results["error"] = f"HTTP {response.status_code}: {response.text}"
            return results
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 网络请求失败: {e}")
        results["error"] = f"网络错误: {str(e)}"
        return results
    
    try:
        # 测试2: LangChain集成测试
        print("\n🤖 测试2: LangChain集成测试...")
        
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7,
            max_tokens=1000
        )
        
        # 简单测试消息
        messages = [HumanMessage(content="你好，请回复一个简单的JSON格式：{'status': 'ok'}")]
        
        response = llm.invoke(messages)
        
        print("✅ LangChain调用成功")
        print(f"📄 响应内容: {response.content}")
        
        results["model_test"] = True
        results["raw_response"] = response.content
        
    except Exception as e:
        print(f"❌ LangChain测试失败: {e}")
        results["error"] = f"LangChain错误: {str(e)}"
    
    return results

def test_prompt_file() -> Dict[str, Any]:
    """测试prompt.txt文件是否存在"""
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.txt')
    
    result = {
        "file_exists": False,
        "file_size": 0,
        "content_preview": None,
        "error": None
    }
    
    try:
        if os.path.exists(prompt_path):
            result["file_exists"] = True
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                result["file_size"] = len(content)
                result["content_preview"] = content[:200] + "..." if len(content) > 200 else content
                print(f"✅ prompt.txt文件存在，大小: {result['file_size']} 字符")
        else:
            result["error"] = "prompt.txt文件不存在"
            print("❌ prompt.txt文件不存在")
            
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ 读取prompt.txt失败: {e}")
    
    return result

def test_json_extraction() -> Dict[str, Any]:
    """测试JSON提取功能"""
    print("\n📋 测试JSON提取功能...")
    
    # 模拟API响应
    test_responses = [
        '{"对话质量问题": [], "销售违禁词问题": [], "优秀销售话术": [], "整体评估": {"对话总轮次": "5", "问题轮次占比": "20%", "优秀话术轮次占比": "40%", "总体质量等级": "良好"}}',
        '```json\n{"对话质量问题": [{"轮次": 1, "类型": "测试"}], "销售违禁词问题": [], "优秀销售话术": [], "整体评估": {"对话总轮次": "3"}}\n```',
        '一些前置文字\n{"对话质量问题": [], "销售违禁词问题": [], "优秀销售话术": [], "整体评估": {"对话总轮次": "7"}}\n一些后置文字'
    ]
    
    results = []
    
    for i, response in enumerate(test_responses, 1):
        print(f"🧪 测试案例 {i}...")
        try:
            # 使用extract_json_from_text函数
            import re
            json_pattern = r'```json\s*({.*?})\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                extracted = json.loads(match.group(1))
            else:
                json_pattern = r'\{.*\}'
                match = re.search(json_pattern, response, re.DOTALL)
                if match:
                    extracted = json.loads(match.group())
                else:
                    extracted = json.loads(response)
            
            results.append({
                "test_case": i,
                "success": True,
                "extracted_keys": list(extracted.keys()),
                "sample": extracted
            })
            print(f"✅ 测试案例 {i} 成功，提取到键: {list(extracted.keys())}")
            
        except Exception as e:
            results.append({
                "test_case": i,
                "success": False,
                "error": str(e)
            })
            print(f"❌ 测试案例 {i} 失败: {e}")
    
    return results

def main():
    """主测试函数"""
    print("=" * 60)
    print("🚀 DeepSeek API调试测试工具")
    print("=" * 60)
    
    # 获取API密钥
    api_key = input("请输入DeepSeek API密钥 (或按Enter使用环境变量): ").strip()
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    
    base_url = input("请输入API基础URL (默认: https://api.deepseek.com/v1): ").strip()
    if not base_url:
        base_url = "https://api.deepseek.com/v1"
    
    print(f"\n🔑 使用API密钥: {'*' * 8}{api_key[-4:] if api_key else '未提供'}")
    print(f"🔗 使用基础URL: {base_url}")
    
    # 运行测试
    print("\n" + "=" * 60)
    api_results = test_api_connection(api_key, base_url)
    
    print("\n" + "=" * 60)
    prompt_results = test_prompt_file()
    
    print("\n" + "=" * 60)
    json_results = test_json_extraction()
    
    # 总结报告
    print("\n" + "=" * 60)
    print("📊 测试总结报告")
    print("=" * 60)
    
    print("\n🔍 API连接测试结果:")
    for key, value in api_results.items():
        print(f"  {key}: {value}")
    
    print("\n📄 Prompt文件测试结果:")
    for key, value in prompt_results.items():
        print(f"  {key}: {value}")
    
    print("\n📋 JSON提取测试结果:")
    for result in json_results:
        print(f"  测试案例 {result['test_case']}: {'✅ 成功' if result['success'] else '❌ 失败'}")
    
    # 建议
    print("\n💡 建议:")
    if not api_results["connection_success"]:
        print("  - 检查网络连接")
        print("  - 验证API密钥是否正确")
        print("  - 尝试使用VPN或代理")
        print("  - 检查base_url是否正确")
    
    if not prompt_results["file_exists"]:
        print("  - 确保prompt.txt文件存在于langchain目录中")
    
    if any(not r["success"] for r in json_results):
        print("  - 检查JSON提取逻辑")

if __name__ == "__main__":
    main()
