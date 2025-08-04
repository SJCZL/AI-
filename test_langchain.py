#!/usr/bin/env python3
"""
测试LangChain + DeepSeek V3集成的简单测试脚本
"""

import os
import sys
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

def test_deepseek_connection(api_key, base_url="https://api.deepseek.com/v1"):
    """测试DeepSeek API连接"""
    try:
        print("🔄 正在测试DeepSeek API连接...")
        
        # 初始化LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7,
            max_tokens=500
        )
        
        # 创建测试提示
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的助手，请用简短的中文回复。"),
            ("human", "请用一句话介绍你自己")
        ])
        
        # 发送测试消息
        messages = prompt.format_messages()
        response = llm.invoke(messages)
        
        print("✅ 连接成功！")
        print(f"🤖 模型回复: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {str(e)}")
        return False

def test_qa_functionality(api_key, base_url="https://api.deepseek.com/v1"):
    """测试质检功能"""
    try:
        print("\n🔄 正在测试质检功能...")
        
        # 初始化LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7,
            max_tokens=2000
        )
        
        # 测试销售对话
        test_conversation = """
        销售：你好，欢迎光临我们的手机店！
        客户：我想看看新手机。
        销售：好的，这款是最新的iPhone 15，价格5999元。
        客户：有点贵啊。
        销售：不贵不贵，这是最新款，功能很强大。
        """
        
        # 创建质检提示
        system_template = """你是一个专业的销售对话质检专家。请分析以下销售对话，并从多个维度进行质量评估。

请严格按照以下JSON格式返回结果：
{
    "总体评分": "1-10的整数",
    "质检结果": "合格/不合格",
    "主要问题": ["问题1", "问题2", ...],
    "改进建议": ["建议1", "建议2", ...],
    "优秀表现": ["表现1", "表现2", ...],
    "关键指标": {
        "服务态度": "1-10",
        "专业程度": "1-10",
        "沟通技巧": "1-10",
        "产品知识": "1-10"
    },
    "风险点": ["风险1", "风险2", ...]
}

请确保JSON格式正确，所有字段都要包含。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "请分析以下销售对话：\n\n{conversation}\n\n请提供详细的质检分析结果。")
        ])
        
        messages = prompt.format_messages(conversation=test_conversation)
        response = llm.invoke(messages)
        
        print("✅ 质检功能测试成功！")
        print(f"📊 分析结果:\n{response.content}")
        
        # 尝试解析JSON
        import json
        try:
            result = json.loads(response.content)
            print("✅ JSON解析成功")
            print(f"📈 总体评分: {result.get('总体评分', 'N/A')}")
            print(f"🎯 质检结果: {result.get('质检结果', 'N/A')}")
        except:
            print("⚠️ 注意：返回格式不是标准JSON")
        
        return True
        
    except Exception as e:
        print(f"❌ 质检功能测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 LangChain + DeepSeek V3 测试工具")
    print("=" * 50)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python test_langchain.py <your_deepseek_api_key> [base_url]")
        print("示例: python test_langchain.py sk-xxxxxxxxxxxxxxxx")
        sys.exit(1)
    
    api_key = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "https://api.deepseek.com/v1"
    
    print(f"🔑 使用API密钥: {api_key[:10]}...")
    print(f"🔗 API端点: {base_url}")
    
    # 测试连接
    if test_deepseek_connection(api_key, base_url):
        # 测试质检功能
        test_qa_functionality(api_key, base_url)
    else:
        print("\n❌ 请检查API密钥和网络连接")
        sys.exit(1)
    
    print("\n✅ 所有测试完成！")
    print("🎯 现在可以运行: streamlit run streamlit_langchain_app.py")
