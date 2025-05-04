import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import streamlit as st
import logging

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# Set page configuration
st.set_page_config(
    page_title="银巢 - 智慧伴老平台",
    page_icon="👴",
    layout="wide"
)

# Title and description
st.title("👴 银巢 - 智慧伴老平台")
st.caption("基于 Qwen2-1.5B 微调的智慧伴老平台，为老年人提供陪伴、情感关怀和智能助手服务")

# Sidebar configuration
with st.sidebar:
    # Title and introduction
    st.title("👴 银巢")
    st.markdown("""
        **银巢团队**  
        银巢是一个专为老年人设计的智慧伴老平台，基于 Qwen2-1.5B 微调技术，扮演虚拟子女或伴侣，提供陪伴聊天、情感关怀和智能助手服务，同时支持阿尔兹海默症预防和监测功能。
    """)
    
    # Divider
    st.markdown("---")
    
    # Generation parameters
    st.header("生成参数")
    max_new_tokens = st.slider("最大生成长度", 50, 512, 256, help="控制生成文本的最大长度。")
    temperature = st.slider("随机性", 0.1, 1.0, 0.7, help="控制生成文本的随机性，值越高越随机。")
    top_p = st.slider("Top-p 采样", 0.1, 1.0, 0.9, help="控制生成文本的多样性，值越高越多样。")
    repetition_penalty = st.slider("重复惩罚", 1.0, 2.0, 1.2)
    
    # Divider
    st.markdown("---")
    
    # Team introduction
    st.header("关于我们")
    st.markdown("""
        我们是银巢团队，由中央财经大学、电子科技大学、南昌大学等多所高校的学生组成，致力于通过人工智能技术提升老年人的生活质量。  
        我们的使命是为老年人提供贴心的陪伴和智能服务，缓解孤独感，促进心理健康，并助力智慧养老产业发展。
    """)
    
    # Placeholder image
    st.image("https://via.placeholder.com/150", caption="银巢 Logo", use_column_width=True)
    
    # Contact information
    st.markdown("---")
    st.markdown("**联系我们**")
    st.markdown("📧 邮箱: [yinchao@cufe.edu.cn](mailto:yinchao@cufe.edu.cn)")
    st.markdown("🌐 官网: [银巢官网](https://yinchao.x.ai)")

# Load model function
@st.cache_resource
def load_model():
    try:
        adapter_path = "qwen_yinchao_model"
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
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
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

# Paid features section
st.markdown("---")
st.markdown("**付费功能**")
st.markdown("以下是银巢的付费功能：")
st.markdown("- **个性化对话服务**：基于子女提供的聊天记录和兴趣爱好，定制更真实、贴心的对话体验。（首月仅需19.9元）")
st.markdown("- **阿尔兹海默症监测与预防**：通过认知训练和行为分析，预防和早期监测阿尔兹海默症。（每月29.9元）")
st.markdown("- **子女端关怀功能**：实时了解父母情绪和健康状况，发送个性化问候，防范诈骗风险。（每月15元）")

st.markdown("---")
st.markdown("**立即付费**")
st.markdown("[前往付费页面](https://yinchao.x.ai/pay)")
st.markdown("如果您已经是付费用户，请输入您对应付费功能的凭证：")
paid_code = st.text_input("付费凭证")
if st.button("验证"):
    if paid_code == "yinchao_paid_code":  # Replace with actual verification logic
        st.success("验证成功！您已成功解锁付费功能。")
    else:
        st.error("验证失败，请检查您的付费凭证。")
```
