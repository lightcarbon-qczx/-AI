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
from google.cloud import speech_v1p1beta1 as speech
import os
import json

# 配置日志
logging.basicConfig(filename="app.log", level=logging.INFO)

# 初始化 session_state
if "reminder_triggered" not in st.session_state:
    st.session_state.reminder_triggered = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "health_data" not in st.session_state:
    st.session_state.health_data = []
if "tasks" not in st.session_state:
    st.session_state.tasks = []

# 设置 Google Cloud 认证
try:
    credentials_json = st.secrets["google_cloud"]["credentials"]
    with open("gcp_credentials.json", "w") as f:
        f.write(credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_credentials.json"
except KeyError:
    st.error("Google Cloud 认证密钥未配置，请在 secrets.toml 中添加 google_cloud.credentials")
    st.stop()

# 设置页面配置
st.set_page_config(
    page_title="银巢 - 智慧伴老平台",
    page_icon="👴",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://yinchao.x.ai/help',
        'Report a bug': 'mailto:yinchao@cufe.edu.cn',
        'About': '银巢 - 智慧伴老平台，专为老年用户设计。'
    }
)

# 自定义 CSS 样式
st.markdown("""
    <style>
    .stApp {
        font-family: 'Noto Sans', sans-serif;
        font-size: 20px;
        color: #333333;
        background: linear-gradient(to bottom, #E0F7FA, #F9FBFC);
    }
    .stButton>button {
        background-color: #26A69A;
        color: white;
        font-size: 22px;
        padding: 16px 28px;
        border-radius: 12px;
    }
    .stButton>button:hover {
        background-color: #00897B;
    }
    .stTitle {
        font-size: 40px;
        color: #00695C;
        text-align: center;
    }
    .stSubheader {
        color: #00796B;
        font-size: 28px;
        font-weight: 600;
        margin-top: 25px;
    }
    .stTextInput>input, .stTimeInput input, .stNumberInput input {
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #B0BEC5;
    }
    </style>
""", unsafe_allow_html=True)

# 标题和描述
st.markdown('<h1 class="stTitle">👴 银巢 - 智慧伴老平台</h1>', unsafe_allow_html=True)
st.markdown('<p class="stCaption">您的智能伴侣，随时为您提供温暖陪伴与实用帮助</p>', unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.markdown("### 👴 银巢")
    st.markdown("欢迎体验银巢，我们为您带来贴心服务！")
    st.markdown("---")
    st.markdown("#### 关于我们")
    st.markdown("银巢团队致力于通过AI技术提升老年生活品质。")
    try:
        st.image("图片1.jpg", caption="银巢 Logo", use_container_width=True)
    except FileNotFoundError:
        st.warning("图片1.jpg 未找到，请上传正确的图片文件")
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
# 笑话 API 函数
@st.cache_data
def get_joke():
    try:
        url = "https://official-joke-api.appspot.com/random_joke"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return f"{data['setup']} - {data['punchline']}"
    except requests.RequestException:
        return "暂无法获取笑话"

# 新闻 API 函数
@st.cache_data
def get_news():
    try:
        api_key = st.secrets["NEWS_API_KEY"]
        url = f"https://newsapi.org/v2/top-headlines?country=cn&apiKey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["status"] == "ok":
            headlines = [article["title"] for article in data["articles"][:3]]
            return "\n".join(headlines)
        return "暂无法获取新闻信息"
    except (requests.RequestException, KeyError):
        return "新闻服务不可用，请检查 API 密钥配置"

# 加载模型函数
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

# 加载模型
with st.spinner("正在加载模型..."):
    model, tokenizer = load_model()

# 创建 pipeline
try:
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    st.error(f"Pipeline创建失败: {str(e)}")
    st.stop()

# 上传音频文件
def upload_audio():
    uploaded_file = st.file_uploader("上传音频文件（WAV 或 MP3）", type=["wav", "mp3"])
    if uploaded_file is not None:
        return uploaded_file
    return None

# 语音转文本函数（使用 Google Cloud Speech-to-Text）
def audio_to_text(audio_file):
    try:
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_file.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="zh-CN",
        )
        response = client.recognize(config=config, audio=audio)
        for result in response.results:
            return result.alternatives[0].transcript
        return "未识别到语音内容"
    except Exception as e:
        return f"语音识别错误: {e}"

# 聊天界面
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

# 语音输入
if st.button("上传音频文件"):
    audio_file = upload_audio()
    if audio_file:
        with st.spinner("正在识别语音..."):
            text = audio_to_text(audio_file)
            st.write(f"识别结果: {text}")
            if text and not text.startswith("语音识别错误"):
                st.session_state.messages.append({"role": "user", "content": text})
                st.chat_message("user").write(text)
                # 自动将识别结果作为输入处理
                system_prompt = "system\n你是银巢，一个基于 Qwen2-1.5B 微调的智慧伴老助手，专为老年人提供陪伴聊天、情感关怀和智能助手服务。你由中央财经大学银巢团队开发，旨在模拟虚拟子女或伴侣的角色，用温馨、亲切的语气与用户交流，并支持阿尔兹海默症预防和监测功能。\n"
                full_prompt = system_prompt + f"user\n{text}\nassistant\n"
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

# 动态功能区域
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

# 健康监测
st.markdown('<h2 class="stSubheader">健康监测</h2>', unsafe_allow_html=True)
with st.expander("记录您的健康数据"):
    bp_systolic = st.number_input("收缩压 (mmHg)", min_value=0, max_value=300, value=120)
    bp_diastolic = st.number_input("舒张压 (mmHg)", min_value=0, max_value=200, value=80)
    heart_rate = st.number_input("心率 (次/分钟)", min_value=0, max_value=200, value=70)
    if st.button("保存健康数据"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.health_data.append({
            "timestamp": timestamp,
            "bp_systolic": bp_systolic,
            "bp_diastolic": bp_diastolic,
            "heart_rate": heart_rate
        })
        if bp_systolic > 140 or bp_diastolic > 90:
            st.warning("您的血压偏高，建议关注饮食和适量运动，或咨询医生。")
        elif heart_rate > 100 or heart_rate < 60:
            st.warning("您的心率可能异常，请注意休息并咨询专业意见。")
        else:
            st.success("您的健康数据正常，继续保持！")

# 绘制健康数据趋势
if st.session_state.health_data:
    st.markdown("### 健康数据趋势")
    df = pd.DataFrame(st.session_state.health_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    plt.rcParams['font.family'] = 'Noto Sans CJK SC'
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp"], df["bp_systolic"], label="收缩压 (mmHg)", color="red", marker="o")
    plt.plot(df["timestamp"], df["bp_diastolic"], label="舒张压 (mmHg)", color="blue", marker="o")
    plt.plot(df["timestamp"], df["heart_rate"], label="心率 (次/分钟)", color="green", marker="o")
    plt.title("健康数据趋势", fontsize=20, pad=15)
    plt.xlabel("时间", fontsize=16)
    plt.ylabel("数值", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    st.image(buffer, use_column_width=True)
    plt.close()

# 任务管理
st.markdown('<h2 class="stSubheader">今日任务</h2>', unsafe_allow_html=True)
task = st.text_input("添加新任务")
if st.button("添加任务"):
    st.session_state.tasks.append({"text": task, "completed": False})
    st.success("任务已添加！")
if st.session_state.tasks:
    for i, task in enumerate(st.session_state.tasks):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{i+1}. {task['text']}{' ✅' if task['completed'] else ''}")
        with col2:
            if st.button("完成", key=f"complete_{i}"):
                task["completed"] = True
                st.experimental_rerun()

# 家庭相册
st.markdown('<h2 class="stSubheader">家庭相册</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("上传家庭照片", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="家庭照片", use_container_width=True)

# 提醒设置
st.sidebar.markdown("### 提醒设置")
reminder_time = st.sidebar.time_input("设置提醒时间")
reminder_message = st.sidebar.text_input("提醒内容", "该喝水了！")
if st.sidebar.button("设置提醒"):
    formatted_time = reminder_time.strftime("%H:%M")
    schedule.every().day.at(formatted_time).do(lambda: st.session_state.update({
        "reminder_triggered": True,
        "reminder_message": reminder_message
    }))
    st.sidebar.success(f"将在 {formatted_time} 提醒您：{reminder_message}")

# 显示提醒
if "reminder_triggered" in st.session_state and st.session_state.reminder_triggered:
    st.sidebar.success(st.session_state.reminder_message)
    st.session_state.reminder_triggered = False
