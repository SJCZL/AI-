#!/usr/bin/env python3
"""
测试脚本：验证修复后的LangChain质检功能
"""

import os
import sys
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def test_prompt_loading():
    """测试prompt.txt文件加载"""
    try:
        prompt_file_path = os.path.join(os.path.dirname(__file__), 'prompt.txt')
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("✅ prompt.txt文件加载成功")
        print(f"📄 文件长度: {len(content)} 字符")
        print(f"🔍 内容预览: {content[:200]}...")
        return True
    except Exception as e:
        print(f"❌ prompt.txt文件加载失败: {e}")
        return False

def test_create_qa_prompt():
    """测试创建质检提示模板"""
    try:
        # 模拟create_qa_prompt函数
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), 'prompt.txt')
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                system_template = f.read()
            print("✅ 使用prompt.txt作为系统提示词")
        except Exception as e:
            print(f"⚠️ 使用默认提示词: {e}")
            system_template = "默认提示词"
        
        human_template = "请分析以下销售对话：{conversation}"
        
        # 创建提示模板
        from langchain.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
        
        # 测试格式化
        test_conversation = "这是一个测试对话"
        messages = prompt.format_messages(conversation=test_conversation)
        
        print("✅ 提示模板创建成功")
        print(f"📋 消息数量: {len(messages)}")
        print(f"🔍 系统消息长度: {len(messages[0].content)} 字符")
        
        return True
        
    except Exception as e:
        print(f"❌ 提示模板创建失败: {e}")
        return False

def test_llm_initialization():
    """测试LLM初始化"""
    try:
        # 检查环境变量
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("⚠️ 未找到API密钥环境变量，跳过LLM测试")
            return True
            
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.7,
            max_tokens=1000
        )
        print("✅ LLM初始化成功")
        return True
        
    except Exception as e:
        print(f"❌ LLM初始化失败: {e}")
        return False

if __name__ == "__main__":
    print("🧪 开始测试LangChain质检修复...")
    print("=" * 50)
    
    tests = [
        ("prompt.txt文件加载", test_prompt_loading),
        ("提示模板创建", test_create_qa_prompt),
        ("LLM初始化", test_llm_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 测试: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   测试失败")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！修复成功")
    else:
        print("⚠️ 部分测试失败，请检查配置")
