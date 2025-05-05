import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import streamlit as st
import logging
import random
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# Set page configuration
st.set_page_config(
    page_title="银巢 - 智慧伴老平台",
    page_icon="👴",
    layout="wide"
)

# Custom CSS for refined design
st.markdown("""
    <style>
    .stApp {
        font-size: 18px;
        color: #333333;
        background-color: #ffffff;
    }
    .stSidebar {
        background-color: #f0f0f0;
    }
    .stButton>button {
        background-color: #e0f7fa;
        color: #00695c;
        border: 2px solid #00695c;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #b2ebf2;
    }
    .stExpander {
        background-color: #fafafa;
        border-radius: 5px;
    }
    .stTitle {
        text-align: center;
        color: #00695c;
        font-size: 28px;
    }
    .stCaption {
        text-align: center;
        color: #555555;
        font-size: 16px;
    }
    .stSubheader {
        color: #00796b;
        font-size: 22px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description with custom styling
st.markdown('<h1 class="stTitle">👴 银巢 - 智慧伴老平台</h1>', unsafe_allow_html=True)
st.markdown('<p class="stCaption">您的虚拟伴侣，随时为您提供陪伴和帮助</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("👴 银巢")
    st.markdown("欢迎使用银巢，我们随时为您服务！")
    st.markdown("银巢是一个专为老年人设计的智慧伴老平台，提供陪伴聊天、情感关怀和智能助手服务。")
    st.markdown("---")
    st.header("关于我们")
    st.markdown("我们是银巢团队，致力于通过人工智能技术提升老年人的生活质量。")
    st.image("图片1.jpg", caption="银巢 Logo", use_container_width=True)
    st.markdown("---")
    st.markdown("**联系我们**")
    st.markdown("📧 邮箱: [yinchao@cufe.edu.cn](mailto:yinchao@cufe.edu.cn)")
    st.markdown("🌐 官网: [银巢官网](https://yinchao.x.ai)")

# Weather API function (caching data)
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
    else:
        return "无法获取天气信息"

# Joke API function (caching data)
@st.cache_data
def get_joke():
    url = "https://official-joke-api.appspot.com/random_joke"
    response = requests.get(url)
    data = response.json()
    return f"{data['setup']} - {data['punchline']}"

# News API function (caching data)
@st.cache_data
def get_news():
    api_key = "your_news_api_key"  # Replace with your News API key
    url = f"https://newsapi.org/v2/top-headlines?country=cn&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data["status"] == "ok":
        headlines = [article["title"] for article in data["articles"][:3]]
        return "\n".join(headlines)
    else:
        return "无法获取新闻信息"

# Load model function (caching resource)
@st.cache_resource
def load_model():
    try:
        adapter_path = "qwen_finance_model"
        config = PeftConfig.from_pretrained(adapter_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Load and merge adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        
        # Verify model type
        if "Peft" in str(type(model)):
            raise ValueError("适配器未正确合并")
            
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
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
except Exception as e:
    st.error(f"Pipeline创建失败: {str(e)}")
    st.stop()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if st.button("开始聊天"):
    prompt = "您好，我是银巢，您的虚拟伴侣！有什么可以帮助您的？"
    st.session_state.messages.append({"role": "assistant", "content": prompt})
    st.chat_message("assistant").write(prompt)
elif st.button("帮助"):
    st.write("使用指南：您可以直接在下方输入框中输入您的问题或想说的话，银巢会尽力为您提供帮助。")

if prompt := st.chat_input("您好，我是银巢，您的虚拟伴侣！有什么可以帮助您的？"):
    # System prompt for 银巢
    system_prompt = "system\n你是银巢，一个基于 Qwen2-1.5B 微调的智慧伴老助手，专为老年人提供陪伴聊天、情感关怀和智能助手服务。你由中央财经大学银巢团队开发，旨在模拟虚拟子女或伴侣的角色，用温馨、亲切的语气与用户交流，并支持阿尔兹海默症预防和监测功能。\n"
    
    # Build input prompt
    full_prompt = system_prompt + f"user\n{prompt}\nassistant\n"
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("正在生成回答..."):
            # Encode input
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            
            # Generation parameters
            generate_kwargs = {
                "inputs": inputs.input_ids,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            }
            
            # Generate response
            outputs = model.generate(**generate_kwargs)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "assistant\n" in response:
                answer = response.split("assistant\n")[-1].strip()
            else:
                answer = response.strip()

        # Display response
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Dynamic features section
st.markdown("---")
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

# Health tips
st.markdown("---")
st.markdown('<h2 class="stSubheader">子女留言</h2>', unsafe_allow_html=True)
tips = ["每天喝八杯水，保持身体水分。", "适量运动，保持身心健康。", "多吃蔬菜水果，补充维生素。"]
st.write(random.choice(tips))

# Task manager
st.markdown("---")
st.markdown('<h2 class="stSubheader">今日任务</h2>', unsafe_allow_html=True)
task = st.text_input("添加新任务")
if st.button("添加"):
    if "tasks" not in st.session_state:
        st.session_state.tasks = []
    st.session_state.tasks.append(task)
    st.success("任务已添加！")
if "tasks" in st.session_state:
    for i, t in enumerate(st.session_state.tasks):
        st.write(f"{i+1}. {t}")

# Family photos
st.markdown("---")
st.markdown('<h2 class="stSubheader">家庭共享相册</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("上传家庭照片", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="家庭照片", use_container_width=True)

# More services (paid features)
with st.expander("更多服务"):
    st.markdown("以下是银巢的付费功能：")
    st.markdown("- **个性化对话服务**：定制更真实、贴心的对话体验。（首月仅需19.9元）")
    st.markdown("- **阿尔兹海默症监测与预防**：预防和早期监测阿尔兹海默症。（每月29.9元）")
    st.markdown("- **子女端关怀功能**：实时了解父母情绪和健康状况。（每月15元）")
    st.markdown("[了解更多](https://yinchao.x.ai/pay)")

# Reminder settings
st.sidebar.header("提醒设置")
reminder_time = st.sidebar.time_input("设置提醒时间")
reminder_message = st.sidebar.text_input("提醒内容", "该喝水了！")
if st.sidebar.button("设置提醒"):
    st.sidebar.success(f"将在 {reminder_time} 提醒您：{reminder_message}")
