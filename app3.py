import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import streamlit as st
import logging
import random
import requests
from datetime import datetime
import schedule
import time
import threading
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from matplotlib import font_manager

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# Set page configuration
st.set_page_config(
    page_title="银巢 - 智慧伴老平台",
    page_icon="👴",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://yinchao.x.ai/help',
        'Report a bug': 'mailto:yinchao@cufe.edu.cn',
        'About': '银巢 - 智慧伴老平台，专为老年人设计。'
    }
)

# Enhanced custom CSS with cyan buttons and background
st.markdown("""
    <style>
    .stApp {
        font-family: 'Noto Sans', sans-serif;
        font-size: 18px;
        color: #2E2E2E;
        background: linear-gradient(to bottom, #E0F7FA, #F9FBFC); /* Soft cyan to white gradient */
    }
    .stSidebar {
        background-color: #E8F0F2;
        padding: 20px;
        border-right: 2px solid #D3E0E5;
    }
    .stButton>button {
        background-color: #26A69A; /* Cyan button color */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 18px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #00897B; /* Darker cyan on hover */
    }
    .stTextInput>input, .stTimeInput input, .stNumberInput input {
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #B0BEC5;
    }
    .stExpander {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 10px;
    }
    .stTitle {
        text-align: center;
        color: #00695C;
        font-size: 36px;
        font-weight: bold;
    }
    .stCaption {
        text-align: center;
        color: #607D8B;
        font-size: 18px;
    }
    .stSubheader {
        color: #00796B;
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="stTitle">👴 银巢 - 智慧伴老平台</h1>', unsafe_allow_html=True)
st.markdown('<p class="stCaption">您的智能伴侣，随时为您提供温暖陪伴与实用帮助</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### 👴 银巢")
    st.markdown("欢迎体验银巢，我们为您带来贴心服务！")
    st.markdown("---")
    st.markdown("#### 关于我们")
    st.markdown("银巢团队致力于通过AI技术提升老年生活品质。")
    st.image("图片1.jpg", caption="银巢 Logo", use_container_width=True)
    st.markdown("---")
    st.markdown("**联系我们**")
    st.markdown("📧 [yinchao@cufe.edu.cn](mailto:yinchao@cufe.edu.cn)")
    st.markdown("🌐 [银巢官网](https://yinchao.x.ai)")

# Weather API function
@st.cache_data
def get_weather(city="Beijing"):
    api_key = "your_openweather_api_key"  # Replace with your OpenWeather API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data["cod"] == 200:
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"{city} 今天天气: {weather}, 温度: {temp}°C"
    return "暂无法获取天气信息"

# Joke API function
@st.cache_data
def get_joke():
    url = "https://official-joke-api.appspot.com/random_joke"
    response = requests.get(url)
    data = response.json()
    return f"{data['setup']} - {data['punchline']}"

# News API function
@st.cache_data
def get_news():
    api_key = "your_news_api_key"  # Replace with your News API key
    url = f"https://newsapi.org/v2/top-headlines?country=cn&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data["status"] == "ok":
        headlines = [article["title"] for article in data["articles"][:3]]
        return "\n".join(headlines)
    return "暂无法获取新闻信息"

# Load model function
@st.cache_resource
def load_model():
    try:
        adapter_path = "qwen_finance_model"
        config = PeftConfig.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

# Load model
with st.spinner("正在加载模型..."):
    model, tokenizer = load_model()

# Create pipeline
try:
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    st.error(f"Pipeline创建失败: {str(e)}")
    st.stop()

# Reminder functionality
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

def send_reminder(message):
    st.session_state.reminder_triggered = True
    st.session_state.reminder_message = message

if "reminder_triggered" not in st.session_state:
    st.session_state.reminder_triggered = False
if "reminder_message" not in st.session_state:
    st.session_state.reminder_message = ""

# Start scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if st.button("开始聊天"):
    prompt = "您好，我是银巢，您的虚拟伴侣！有什么可以帮助您的吗？"
    st.session_state.messages.append({"role": "assistant", "content": prompt})
    st.chat_message("assistant").write(prompt)

if prompt := st.chat_input("有什么可以帮助您的？"):
    system_prompt = "system\n你是银巢，一个基于 Qwen2-1.5B 微调的智慧伴老助手，专为老年人提供陪伴聊天、情感关怀和智能助手服务。你由中央财经大学银巢团队开发，旨在模拟虚拟子女或伴侣的角色，用温馨、亲切的语气与用户交流，并支持阿尔兹海默症预防和监测功能。\n"
    full_prompt = system_prompt + f"user\n{prompt}\nassistant\n"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("正在生成回答..."):
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            generate_kwargs = {
                "inputs": inputs.input_ids,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            }
            outputs = model.generate(**generate_kwargs)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("assistant\n")[-1].strip() if "assistant\n" in response else response.strip()
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Dynamic features section
st.markdown('<h2 class="stSubheader">每日动态</h2>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.write("🌤️ **今日天气**")
    city = st.selectbox("选择城市", ["Beijing", "Shanghai", "Guangzhou", "Shenzhen"], label_visibility="collapsed")
    st.write(get_weather(city))
with col2:
    st.write("😄 **每日笑话**")
    st.write(get_joke())
with col3:
    st.write("📰 **今日新闻**")
    st.write(get_news())

# Health monitoring with data collection and visualization
st.markdown('<h2 class="stSubheader">健康监测</h2>', unsafe_allow_html=True)
with st.expander("记录您的健康数据"):
    bp_systolic = st.number_input("收缩压 (mmHg)", min_value=0, max_value=300, value=120)
    bp_diastolic = st.number_input("舒张压 (mmHg)", min_value=0, max_value=200, value=80)
    heart_rate = st.number_input("心率 (次/分钟)", min_value=0, max_value=200, value=70)
    if st.button("保存健康数据"):
        # Initialize health data storage
        if "health_data" not in st.session_state:
            st.session_state.health_data = []
        
        # Store new data with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.health_data.append({
            "timestamp": timestamp,
            "bp_systolic": bp_systolic,
            "bp_diastolic": bp_diastolic,
            "heart_rate": heart_rate
        })
        
        # Health feedback
        if bp_systolic > 140 or bp_diastolic > 90:
            st.warning("您的血压偏高，建议关注饮食和适量运动，或咨询医生。")
        elif heart_rate > 100 or heart_rate < 60:
            st.warning("您的心率可能异常，请注意休息并咨询专业意见。")
        else:
            st.success("您的健康数据正常，继续保持！")

# Plot health data
if "health_data" in st.session_state and st.session_state.health_data:
    st.markdown("### 健康数据趋势")
    # Convert health data to DataFrame
    df = pd.DataFrame(st.session_state.health_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Set Chinese font for Matplotlib
    plt.rcParams['font.family'] = 'Noto Sans CJK SC'
    plt.rcParams['axes.unicode_minus'] = False եկ

    # Create line plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp"], df["bp_systolic"], label="收缩压 (mmHg)", color="red", marker="o")
    plt.plot(df["timestamp"], df["bp_diastolic"], label="舒张压 (mmHg)", color="blue", marker="o")
    plt.plot(df["timestamp"], df["heart_rate"], label="心率 (次/分钟)", color="green", marker="o")
    
    # Customize plot for elderly users
    plt.title("健康数据趋势", fontsize=20, pad=15)
    plt.xlabel("时间", fontsize=16)
    plt.ylabel("数值", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    
    # Display the plot
    st.image(buffer, use_column_width=True)
    
    # Close the plot to free memory
    plt.close()

# Task manager with completion
st.markdown('<h2 class="stSubheader">今日任务</h2>', unsafe_allow_html=True)
task = st.text_input("添加新任务")
if st.button("添加任务"):
    if "tasks" not in st.session_state:
        st.session_state.tasks = []
    st.session_state.tasks.append({"text": task, "completed": False})
    st.success("任务已添加！")
if "tasks" in st.session_state:
    for i, task in enumerate(st.session_state.tasks):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{i+1}. {task['text']}{' ✅' if task['completed'] else ''}")
        with col2:
            if st.button("完成", key=f"complete_{i}"):
                task["completed"] = True
                st.experimental_rerun()

# Family photos
st.markdown('<h2 class="stSubheader">家庭相册</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("上传家庭照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="家庭照片", use_container_width=True)

# Reminder settings
st.sidebar.markdown("### 提醒设置")
reminder_time = st.sidebar.time_input("设置提醒时间")
reminder_message = st.sidebar.text_input("提醒内容", "该喝水了！")
if st.sidebar.button("设置提醒"):
    formatted_time = reminder_time.strftime("%H:%M")
    schedule.every().day.at(formatted_time).do(send_reminder, reminder_message)
    st.sidebar.success(f"将在 {formatted_time} 提醒您：{reminder_message}")

# Display reminder when triggered
if st.session_state.reminder_triggered:
    st.sidebar.success(st.session_state.reminder_message)
    st.session_state.reminder_triggered = False  # Reset after display
