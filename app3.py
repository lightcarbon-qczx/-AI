import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import streamlit as st
import logging
import random

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# Set page configuration
st.set_page_config(
    page_title="银巢 - 智慧伴老平台",
    page_icon="👴",
    layout="wide"
)

# Custom CSS for better readability
st.markdown("""
    <style>
    .stApp {
        font-size: 18px;
        color: #333333;
        background-color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("👴 银巢 - 智慧伴老平台")
st.caption("您的虚拟伴侣，随时为您提供陪伴和帮助")

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

# Load model function
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
            
            # Generation parameters (fixed values)
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

# Health tips
tips = ["每天喝八杯水，保持身体水分。", "适量运动，保持身心健康。", "多吃蔬菜水果，补充维生素。"]
st.write("健康小贴士：", random.choice(tips))

# More services (paid features)
with st.expander("更多服务"):
    st.markdown("以下是银巢的付费功能：")
    st.markdown("- **个性化对话服务**：定制更真实、贴心的对话体验。（首月仅需19.9元）")
    st.markdown("- **阿尔兹海默症监测与预防**：预防和早期监测阿尔兹海默症。（每月29.9元）")
    st.markdown("- **子女端关怀功能**：实时了解父母情绪和健康状况。（每月150元）")
    st.markdown("[了解更多](https://yinchao.x.ai/pay)")

# Reminder settings
st.sidebar.header("提醒设置")
reminder_time = st.sidebar.time_input("设置提醒时间")
reminder_message = st.sidebar.text_input("提醒内容", "该喝水了！")
if st.sidebar.button("设置提醒"):
    st.sidebar.success(f"将在 {reminder_time} 提醒您：{reminder_message}")
