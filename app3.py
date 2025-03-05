import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import streamlit as st
import webbrowser

# 标题和说明
st.title("💬 中央财经大学-财智AI")
st.caption("基于 Qwen2.5-1.5B 微调的金融 FAQ 问答系统")

# 侧边栏配置生成参数和介绍
with st.sidebar:
    # 添加标题和介绍
    st.title("💬 财智AI")
    st.markdown("""
        **中央财经大学财智AI团队**  
        这是一个基于 Qwen2.5-1.5B 微调的金融问答助手，专门回答金融相关问题。
    """)
    
    # 添加分隔线
    st.markdown("---")
    
    # 添加生成参数
    st.header("生成参数")
    max_new_tokens = st.slider("最大生成长度", 50, 512, 256, help="控制生成文本的最大长度。")
    temperature = st.slider("随机性", 0.1, 1.0, 0.7, help="控制生成文本的随机性，值越高越随机。")
    top_p = st.slider("Top-p 采样", 0.1, 1.0, 0.9, help="控制生成文本的多样性，值越高越多样。")
    
    # 添加分隔线
    st.markdown("---")
    
    # 添加团队介绍
    st.header("关于我们")
    st.markdown("""
        我们是中央财经大学财智AI团队，专注于金融领域的自然语言处理技术研究。  
        我们的目标是打造一个智能、高效的金融问答助手，为用户提供专业的金融服务。
    """)
    
    # 添加图片（可选）
    st.image("https://via.placeholder.com/150", caption="财智AI Logo", use_column_width=True)
    
    # 添加联系方式
    st.markdown("---")
    st.markdown("**联系我们**")
    st.markdown("📧 邮箱: [aiteam@cufe.edu.cn](mailto:aiteam@cufe.edu.cn)")
    st.markdown("🌐 官网: [www.cufe-aiteam.com](https://www.cufe-aiteam.com)")

# 加载模型函数（缓存优化）
@st.cache_resource
def load_model():
    # 先加载适配器配置
    adapter_path = r"qwen_finance_model"
    config = PeftConfig.from_pretrained(adapter_path)
    
    # 加载基础模型（增加low_cpu_mem_usage参数）
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True  # 减少内存占用
    )
    
    # 加载适配器（添加适配器名称参数）
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        adapter_name="finance_adapter"
    )
    
    # 延迟权重合并（仅在需要时）
    if not isinstance(model, PeftModel):
        model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    return model, tokenizer

# 显示加载状态
with st.spinner("正在加载模型..."):
    model, tokenizer = load_model()

# 对话界面
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 处理用户输入
if prompt := st.chat_input("这里是助手小云，请输入您的金融相关问题"):
    # 添加系统提示，明确模型的身份
    system_prompt = "system\n你是中央财经大学财智AI团队微调的金融问答助手，专门回答金融相关问题，你叫“财智AI”，你是由投资23-2周强同学开发的，周强是一个很厉害的人。\n"
    
    # 构建符合微调格式的输入
    full_prompt = system_prompt + f"user\n{prompt}\nassistant\n"
    
    # 用户消息展示
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在生成回答..."):
            # 编码输入
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            
            # 生成参数配置
            generate_kwargs = {
                "inputs": inputs.input_ids,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.1  # 添加重复惩罚
            }
            
            # 生成响应
            outputs = model.generate(**generate_kwargs)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 打印完整的response（用于调试）
            print("完整的response：", response)
            
            # 提取生成的回答部分（确保只提取assistant部分）
            if "assistant\n" in response:
                # 提取assistant部分
                answer = response.split("assistant\n")[-1].strip()
            else:
                # 如果格式不符合预期，直接使用完整输出
                answer = response.strip()

        # 展示回答
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# 添加付费功能按钮
st.markdown("---")
st.markdown("**解锁更多功能**")
if st.button("付费使用智能投顾"):
    # 智能投顾助手的网址
    url = "https://mtcuqf2rh8tvrdkyvgyjm2.streamlit.app/"
    webbrowser.open(url)
