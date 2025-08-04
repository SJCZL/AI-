#!/usr/bin/env python3
"""
测试脚本：直接使用API密钥测试LLM连接和质检功能
"""

import os
import sys
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# 直接设置API密钥
os.environ["DEEPSEEK_API_KEY"] = "sk-f18946020e0944bd860b03d02d459e79"

def test_llm_connection():
    """测试LLM连接"""
    print("=== 测试LLM连接 ===")
    
    # 检查API密钥
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 未找到API密钥")
        return False
    
    print(f"✅ 找到API密钥: {api_key[:10]}...")
    
    try:
        # 初始化LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.7,
            max_tokens=1000
        )
        
        # 测试简单调用
        response = llm.invoke([HumanMessage(content="请回复'连接成功'四个字")])
        print(f"✅ LLM连接成功")
        print(f"🤖 模型回复: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ LLM连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qa_functionality():
    """测试质检功能"""
    print("\n=== 测试质检功能 ===")
    
    # 检查API密钥
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 未找到API密钥")
        return False
    
    try:
        # 初始化LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.7,
            max_tokens=4000
        )
        
        # 读取提示词
        try:
            prompt_file_path = os.path.join(os.path.dirname(__file__), 'prompt.txt')
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                system_template = f.read()
            print("✅ 成功读取prompt.txt")
        except Exception as e:
            print(f"⚠️ 读取prompt.txt失败，使用默认提示词: {e}")
            system_template = """你是一个专业的销售对话质检专家。请分析以下销售对话，并从多个维度进行质量评估。"""
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "请分析以下销售对话：\n\n{conversation}\n\n请提供详细的质检分析结果。")
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
        print("✅ 提示格式化成功")
        
        # 调用LLM
        response = llm.invoke(messages)
        print(f"✅ LLM调用成功")
        print(f"📊 响应长度: {len(response.content)} 字符")
        print(f"🔍 响应预览: {response.content[:200]}...")
        
        # 尝试解析JSON
        def extract_json_from_text(text: str):
            """从文本中提取JSON数据"""
            try:
                return json.loads(text)
            except:
                import re
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
        
        result = extract_json_from_text(response.content)
        print(f"✅ JSON提取 {'成功' if result else '失败'}")
        
        if result:
            print(f"📋 包含的键: {list(result.keys())}")
            for key, value in result.items():
                if isinstance(value, list):
                    print(f"   {key}: {len(value)} 个项目")
                else:
                    print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 质检功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 LLM连接和质检功能测试")
    print("=" * 50)
    
    # 测试连接
    connection_ok = test_llm_connection()
    
    if connection_ok:
        # 测试质检功能
        qa_ok = test_qa_functionality()
        
        if qa_ok:
            print("\n🎉 所有测试通过！")
            print("✅ 系统可以正常调用大模型进行质检")
        else:
            print("\n⚠️ 质检功能测试失败")
    else:
        print("\n❌ LLM连接测试失败")
    
    print("\n" + "=" * 50)
