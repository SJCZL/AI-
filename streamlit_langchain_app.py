import streamlit as st
import pandas as pd
import json
import time
from typing import List, Dict, Any
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="✨ LangChain智能质检助手",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 全局样式 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* 标题样式 */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* 卡片样式 */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* 数据表格样式 */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* 进度条样式 */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 选择器样式 */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        transition: border-color 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
    }
    
    /* 多选框样式 */
    .stMultiSelect > div > div {
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    
    /* 滑块样式 */
    .stSlider > div > div {
        color: #667eea;
    }
    
    /* 卡片容器 */
    .card-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 1rem 0;
    }
    
    /* 动画效果 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* 状态指示器 */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success {
        background: #28a745;
    }
    
    .status-warning {
        background: #ffc107;
    }
    
    .status-error {
        background: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'json_sections' not in st.session_state:
    st.session_state.json_sections = None
if 'selected_sections' not in st.session_state:
    st.session_state.selected_sections = []

# 标题区域
st.markdown('<h1 class="main-title">✨ LangChain智能质检助手</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">基于LangChain + DeepSeek V3的智能销售对话质检分析平台</p>', unsafe_allow_html=True)

# 装饰性分隔线
st.markdown("---")

def initialize_llm(api_key: str, base_url: str = None) -> ChatOpenAI:
    """初始化DeepSeek模型"""
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base=base_url or "https://api.deepseek.com/v1",
            temperature=0.7,
            max_tokens=10000
        )
        return llm
    except Exception as e:
        st.error(f"❌ 模型初始化失败: {str(e)}")
        return None

def create_qa_prompt() -> ChatPromptTemplate:
    """创建质检提示模板"""
    try:
        # 读取prompt.txt文件内容作为系统提示词
        prompt_file_path = os.path.join(os.path.dirname(__file__), 'prompt.txt')
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            system_template = f.read()
    except Exception as e:
        st.error(f"❌ 读取prompt.txt文件失败: {str(e)}")
        # 如果读取失败，使用默认提示词
        system_template = """你是一个专业的销售对话质检专家。请分析以下销售对话，并从多个维度进行质量评估。

请严格按照以下JSON格式返回结果：
{
    "对话质量问题": [],
    "销售违禁词问题": [],
    "优秀销售话术": [],
    "整体评估": {
        "对话总轮次": "总轮次数",
        "问题轮次占比": "X%",
        "优秀话术轮次占比": "X%",
        "总体质量等级": "优秀/良好/一般/待改进"
    }
}

请确保JSON格式正确，所有字段都要包含。"""
    
    human_template = """请分析以下销售对话：

{conversation}

请提供详细的质检分析结果。"""
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])

def analyze_conversation(llm: ChatOpenAI, conversation: str) -> str:
    try:
        prompt = create_qa_prompt()
        messages = prompt.format_messages(conversation=conversation)
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        # 不要访问 response，直接返回错误信息
        return f"❌ 分析失败: {str(e)}"

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

def get_json_sections(json_data: Dict[str, Any]) -> List[str]:
    """获取JSON中的所有顶级键名"""
    return list(json_data.keys())

def extract_section_content(json_data: Dict[str, Any], section: str) -> Any:
    """提取指定section的内容"""
    return json_data.get(section, None)

def process_single_row(row_data, llm):
    """处理单行数据"""
    idx, value = row_data
    if pd.isna(value) or str(value).strip() == "":
        return idx, "⏭️ 空值跳过"
    
    result = analyze_conversation(llm, str(value))
    return idx, result

def process_batch_parallel(data: pd.DataFrame, column_name: str, llm: ChatOpenAI,
                          progress_bar=None, status_text=None, max_workers: int = 5) -> pd.DataFrame:
    """并行批量处理数据"""
    total_rows = len(data)
    
    tasks = [(idx, value) for idx, value in enumerate(data[column_name])]
    results = [None] * total_rows
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_row, task, llm): task[0] 
            for task in tasks
        }
        
        for future in as_completed(future_to_idx):
            idx, result = future.result()
            results[idx] = result
            completed += 1
            
            if progress_bar:
                progress = completed / total_rows
                progress_bar.progress(progress)
            if status_text:
                status_text.text(f"🔄 处理进度: {completed}/{total_rows}")
    
    result_df = data.copy()
    result_df['质检结果'] = results
    
    return result_df

def process_batch(data: pd.DataFrame, column_name: str, llm: ChatOpenAI,
                 progress_bar=None, status_text=None) -> pd.DataFrame:
    """顺序批量处理数据"""
    results = []
    total_rows = len(data)
    
    for idx, value in enumerate(data[column_name]):
        if pd.isna(value) or str(value).strip() == "":
            results.append("⏭️ 空值跳过")
            continue
            
        result = analyze_conversation(llm, str(value))
        results.append(result)
        
        if progress_bar:
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
        if status_text:
            status_text.text(f"🔄 处理进度: {idx + 1}/{total_rows}")
        
        time.sleep(0.5)
    
    result_df = data.copy()
    result_df['质检结果'] = results
    
    return result_df

# 侧边栏配置
with st.sidebar:
    st.markdown("### 🔧 配置中心")
    
    with st.container():
        st.markdown("#### 🔑 DeepSeek API配置")
        
        api_key = st.text_input(
            "DeepSeek API密钥",
            type="password",
            help="从DeepSeek平台获取的API密钥",
            placeholder="输入您的DeepSeek API密钥..."
        )
        
        base_url = st.text_input(
            "API基础URL (可选)",
            value="https://api.deepseek.com/v1",
            help="DeepSeek API端点，通常不需要修改",
            placeholder="https://api.deepseek.com/v1"
        )
        
        # 测试连接按钮
        if st.button("🔄 测试连接"):
            if api_key:
                with st.spinner("测试连接中..."):
                    try:
                        test_llm = initialize_llm(api_key, base_url)
                        if test_llm:
                            response = test_llm.invoke([HumanMessage(content="你好")])
                            st.success("✅ 连接成功！")
                        else:
                            st.error("❌ 连接失败")
                    except Exception as e:
                        st.error(f"❌ 连接失败: {str(e)}")
            else:
                st.warning("⚠️ 请先输入API密钥")
    
    st.markdown("---")
    
    with st.container():
        st.markdown("#### 📁 文件管理")
        uploaded_file = st.file_uploader(
            "选择文件",
            type=['csv', 'xlsx', 'xls'],
            help="支持CSV、Excel文件",
            label_visibility="collapsed"
        )

# 主内容区域
if uploaded_file is not None:
    try:
        # 读取文件
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # 文件信息展示
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            st.success(f"✅ 成功加载文件: **{uploaded_file.name}**")
        with col2:
            file_size = len(uploaded_file.getvalue()) / 1024
            st.info(f"📊 文件大小: {file_size:.1f} KB")
        with col3:
            st.info(f"📋 数据行数: {len(df)}")
        
        # 数据预览卡片
        with st.container():
            st.markdown("### 📊 数据预览")
            st.dataframe(df.head(10), use_container_width=True)
        
        # 配置区域
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 处理配置")
            columns = df.columns.tolist()
            selected_column = st.selectbox(
                "选择质检列",
                columns,
                help="选择包含需要质检文本的列"
            )
            
            # 显示选中列的示例
            with st.expander("📋 查看示例数据"):
                sample_data = df[selected_column].head()
                for i, text in enumerate(sample_data, 1):
                    st.markdown(f"**第{i}行:** {text}")
        
        with col2:
            st.markdown("### ⚡ 性能配置")
            use_parallel = st.checkbox("🚀 启用并行处理", value=True)
            max_workers = st.slider(
                "并发线程数",
                min_value=1,
                max_value=10,
                value=3,
                help="同时处理的API调用数量，建议不超过5个"
            )
            
            # 性能提示
            with st.expander("ℹ️ 性能提示"):
                st.markdown("""
                - **顺序处理**: 每行0.5秒间隔
                - **并行处理**: 3线程约提升2-3倍速度
                - **建议**: 根据API限流调整线程数
                - **DeepSeek限制**: 每分钟最多20次调用
                """)
        
        # 处理按钮区域
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 开始智能质检", type="primary", use_container_width=True):
                if not api_key:
                    st.error("❌ 请先输入DeepSeek API密钥")
                else:
                    with st.spinner("🔄 初始化模型..."):
                        llm = initialize_llm(api_key, base_url)
                        if llm is None:
                            st.error("❌ 模型初始化失败")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            with st.spinner("🔄 正在处理数据..."):
                                if use_parallel:
                                    processed_df = process_batch_parallel(
                                        df, 
                                        selected_column, 
                                        llm, 
                                        progress_bar, 
                                        status_text,
                                        max_workers
                                    )
                                else:
                                    processed_df = process_batch(
                                        df, 
                                        selected_column, 
                                        llm, 
                                        progress_bar, 
                                        status_text
                                    )
                            
                            st.session_state.processed_data = processed_df
                            st.session_state.processing_complete = True
                            st.success("✅ 处理完成！")
                            st.balloons()
                
    except Exception as e:
        st.error(f"❌ 文件读取失败: {str(e)}")

# 结果展示区域
if st.session_state.processing_complete and st.session_state.processed_data is not None:
    st.markdown("---")
    st.markdown("### 📋 质检结果")
    
    # 结果统计卡片
    col1, col2, col3, col4 = st.columns(4)
    
    total_rows = len(st.session_state.processed_data)
    success_rows = len([r for r in st.session_state.processed_data['质检结果'] 
                       if not str(r).startswith(('❌', '⚠️', '⏭️'))])
    
    with col1:
        st.metric("📊 总处理行数", total_rows)
    with col2:
        st.metric("✅ 成功处理", success_rows)
    with col3:
        st.metric("❌ 失败行数", total_rows - success_rows)
    with col4:
        st.metric("📈 成功率", f"{(success_rows/total_rows)*100:.1f}%")
    
    # 结果表格
    st.markdown("#### 📊 详细结果")
    st.dataframe(st.session_state.processed_data, use_container_width=True)
    
    # JSON内容提取功能
    st.markdown("### 📊 智能内容提取")
    
    all_json_data = []
    valid_rows = []
    
    for idx, row in st.session_state.processed_data.iterrows():
        result_text = str(row.get('质检结果', ''))
        if result_text and not result_text.startswith(('❌', '⚠️', '⏭️')):
            json_data = extract_json_from_text(result_text)
            if json_data:
                all_json_data.append(json_data)
                valid_rows.append(idx)
    
    if all_json_data:
        all_sections = set()
        for json_data in all_json_data:
            all_sections.update(get_json_sections(json_data))
        
        if all_sections:
            st.markdown("#### 🎯 选择提取内容")
            
            selected_sections = st.multiselect(
                "选择要整理的JSON部分：",
                sorted(all_sections),
                help="选择要提取和整理的JSON部分"
            )
            
            if selected_sections:
                st.markdown("#### 📋 提取结果")
                
                tabs = st.tabs(selected_sections)
                
                for i, section in enumerate(selected_sections):
                    with tabs[i]:
                        section_data = []
                        for idx, json_data in enumerate(all_json_data):
                            content = extract_section_content(json_data, section)
                            if content is not None:
                                if isinstance(content, list):
                                    for item in content:
                                        section_data.append({
                                            '原始行号': valid_rows[idx] + 1,
                                            '内容': item
                                        })
                                else:
                                    section_data.append({
                                        '原始行号': valid_rows[idx] + 1,
                                            '内容': content
                                        })
                        
                        if section_data:
                            df_section = pd.DataFrame(section_data)
                            st.dataframe(df_section, use_container_width=True)
                            
                            csv_data = df_section.to_csv(index=False)
                            st.download_button(
                                label=f"📥 下载 {section}",
                                data=csv_data,
                                file_name=f"{section}.csv",
                                mime="text/csv",
                                key=f"download_{section}_{i}"
                            )
                        else:
                            st.info(f"📭 {section} 暂无数据")
    
    # 下载区域
    st.markdown("### 📥 下载选项")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = st.session_state.processed_data.to_csv(index=False)
        st.download_button(
            label="📄 下载完整CSV",
            data=csv,
            file_name="质检结果.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.processed_data.to_excel(writer, index=False, sheet_name='质检结果')
        excel_data = output.getvalue()
        
        st.download_button(
            label="📊 下载完整Excel",
            data=excel_data,
            file_name="质检结果.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        if st.button("🔄 重新处理", use_container_width=True):
            st.session_state.processed_data = None
            st.session_state.processing_complete = False
            st.session_state.json_sections = None
            st.session_state.selected_sections = []
            st.rerun()

# 底部信息
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🚀 <strong>LangChain智能质检助手</strong> - 让AI为您的销售对话质检赋能</p>
    <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
        基于Streamlit + LangChain + DeepSeek V3构建 | 支持并行处理 | 智能内容提取
    </p>
</div>
""", unsafe_allow_html=True)
