#!/usr/bin/env python3
"""
调试脚本：验证LangChain质检功能的核心流程
"""

import os
import sys
import json
import re
from typing import List, Dict, Any

# 测试基础功能
def test_basic_functionality():
    print("=== 基础功能测试 ===")
    
    # 测试1: 环境变量
    print("\n1. 环境变量检查:")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    print(f"   API密钥: {'✅ 已设置' if api_key else '❌ 未设置'}")
    
    # 测试2: 文件存在性
    print("\n2. 文件存在性检查:")
    required_files = ['prompt.txt', 'streamlit_langchain_app.py', 'requirements.txt']
    for file in required_files:
        exists = os.path.exists(file)
        print(f"   {file}: {'✅ 存在' if exists else '❌ 不存在'}")
    
    # 测试3: prompt.txt内容
    print("\n3. prompt.txt内容检查:")
    try:
        with open('prompt.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"   文件大小: {len(content)} 字符")
        print(f"   包含JSON格式要求: {'✅' if 'JSON' in content else '❌'}")
        print(f"   包含三个维度: {'✅' if '对话质量问题' in content else '❌'}")
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")

def test_langchain_import():
    print("\n=== LangChain导入测试 ===")
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema import HumanMessage, SystemMessage
        print("✅ 所有LangChain模块导入成功")
        return True
    except Exception as e:
        print(f"❌ LangChain导入失败: {e}")
        return False

def test_json_extraction():
    print("\n=== JSON提取功能测试 ===")
    
    # 模拟LLM返回的文本
    test_text = """
    这里是一些分析文本...
    ```json
    {
        "对话质量问题": [
            {
                "轮次": 1,
                "类型": "测试问题",
                "描述": "这是一个测试",
                "严重程度": "低"
            }
        ],
        "销售违禁词问题": [],
        "优秀销售话术": [],
        "整体评估": {
            "对话总轮次": "5",
            "问题轮次占比": "20%",
            "优秀话术轮次占比": "0%",
            "总体质量等级": "良好"
        }
    }
    ```
    其他文本...
    """
    
    def extract_json_from_text(text: str) -> Dict[str, Any]:
        """从文本中提取JSON数据"""
        try:
            return json.loads(text)
        except:
            json_pattern = r'```json\s*({.*?})\s*```'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
            
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        
        return {}
    
    result = extract_json_from_text(test_text)
    print(f"JSON提取结果: {'✅ 成功' if result else '❌ 失败'}")
    if result:
        print(f"   包含的键: {list(result.keys())}")
    return bool(result)

def test_llm_connection():
    print("\n=== LLM连接测试 ===")
    
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  未设置API密钥，跳过LLM连接测试")
        return None
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.7,
            max_tokens=1000
        )
        
        # 测试简单调用
        response = llm.invoke(["Hello, this is a test"])
        print("✅ LLM连接成功")
        print(f"   响应长度: {len(str(response))} 字符")
        return True
        
    except Exception as e:
        print(f"❌ LLM连接失败: {e}")
        return False

def test_complete_flow():
    print("\n=== 完整流程测试 ===")
    
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  未设置API密钥，跳过完整流程测试")
        return
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        
        # 初始化LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.7,
            max_tokens=4000
        )
        
        # 读取提示词
        with open('prompt.txt', 'r', encoding='utf-8') as f:
            system_template = f.read()
        
        human_template = "请分析以下销售对话：{conversation}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
        
        # 测试对话
        test_conversation = """
        销售：您好，欢迎光临，请问有什么可以帮您的？
        顾客：我想看看手表。
        销售：这款是我们最新款的机械表，采用瑞士进口机芯，走时非常精准。
        顾客：价格怎么样？
        销售：这款绝对是最低价，今天买还有特别优惠，错过就没有了。
        """
        
        # 格式化消息
        messages = prompt.format_messages(conversation=test_conversation)
        print(f"✅ 提示格式化成功，消息数: {len(messages)}")
        
        # 调用LLM
        response = llm.invoke(messages)
        print(f"✅ LLM调用成功，响应长度: {len(str(response.content))} 字符")
        
        # 提取JSON
        result = extract_json_from_text(response.content)
        print(f"✅ JSON提取 {'成功' if result else '失败'}")
        
        if result:
            print(f"   包含键: {list(result.keys())}")
            for key, value in result.items():
                if isinstance(value, list):
                    print(f"   {key}: {len(value)} 个项目")
                else:
                    print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 LangChain质检功能调试测试")
    print("=" * 50)
    
    test_basic_functionality()
    
    if test_langchain_import():
        test_json_extraction()
        test_llm_connection()
        test_complete_flow()
    
    print("\n" + "=" * 50)
    print("🔍 调试测试完成")
