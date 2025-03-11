import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import streamlit as st
import logging
import pandas as pd
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import time
import requests
import hashlib

# 设置页面配置（必须在第一个命令中调用）
st.set_page_config(
    page_title="财智AI - 金融问答助手",
    page_icon="💬",
    layout="wide"
)

# 用户数据文件路径
USER_DATA_FILE = "users.yaml"

# 加载用户数据
def load_users():
    try:
        with open(USER_DATA_FILE, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}

# 保存用户数据
def save_users(users):
    with open(USER_DATA_FILE, "w") as file:
        yaml.dump(users, file)

# 初始化会话状态
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# 初始化用户数据
USERS = load_users()

# 登录注册页面
def login_register():
    st.title("用户登录/注册")
    st.divider()

    # 切换登录/注册
    login_type = st.radio("选择操作", ("登录", "注册"))

    if login_type == "登录":
        login_method = st.radio("选择登录方式", ("用户名", "手机号"))
        if login_method == "用户名":
            username = st.text_input("用户名")
        else:
            phone = st.text_input("手机号")
        password = st.text_input("密码", type="password")

        if st.button("登录"):
            if login_method == "用户名":
                if username in USERS and USERS[username]["password"] == password:
                    st.session_state.logged_in = True
                    st.success("登录成功!")
                    time.sleep(0.5)
                    st.rerun()  # 重新运行应用以加载主界面
                else:
                    st.error("用户名或密码错误")
            else:
                found = False
                for user in USERS.values():
                    if "phone" in user and user["phone"] == phone and user["password"] == password:
                        st.session_state.logged_in = True
                        st.success("登录成功!")
                        time.sleep(0.5)
                        st.rerun()  # 重新运行应用以加载主界面
                        found = True
                        break
                if not found:
                    st.error("手机号或密码错误")

    else:
        new_username = st.text_input("新用户名")
        new_password = st.text_input("新密码", type="password")
        new_phone = st.text_input("手机号（可选，用于登录和找回密码）")

        if st.button("注册"):
            username_exists = False
            phone_exists = False
            for user in USERS.values():
                if user["name"] == new_username:
                    username_exists = True
                if "phone" in user and user["phone"] == new_phone:
                    phone_exists = True
            if username_exists:
                st.error("用户名已存在")
            elif phone_exists:
                st.error("手机号已存在")
            else:
                USERS[new_username] = {
                    "name": new_username,
                    "password": new_password,
                    "subscription": "免费",
                    "phone": new_phone
                }
                save_users(USERS)
                st.success("注册成功! 请登录.")
                time.sleep(0.5)
                st.rerun()  # 重新运行应用以加载登录界面

# 主应用逻辑
def main():
    # 配置日志
    logging.basicConfig(filename="app.log", level=logging.INFO)

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
        repetition_penalty = st.slider("重复惩罚", 1.0, 2.0, 1.2)
        
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
        st.markdown("📧 邮箱: [13292017003@163.com](mailto:aiteam@cufe.edu.cn)")
        st.markdown("🌐 官网: [www.cufe-aiteam.com](https://www.cufe-aiteam.com)")

    # 加载模型函数（关键修改点）
    @st.cache_resource
    def load_model():
        try:
            adapter_path = "qwen_finance_model"
            config = PeftConfig.from_pretrained(adapter_path)
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # 加载并强制合并适配器
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model = model.merge_and_unload()  # 确保合并适配器
            
            # 加载tokenizer（强制从基础模型路径加载）
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            
            # 验证模型类型
            if "Peft" in str(type(model)):
                raise ValueError("适配器未正确合并")
                
            model.eval()
            return model, tokenizer
            
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            st.stop()

    # 加载模型
    with st.spinner("正在加载模型..."):
        model, tokenizer = load_model()
        # 调试信息（可选）
        print(f"Model device: {model.device}")
        print(f"Model dtype: {model.dtype}")
        print(f"Tokenizer length: {len(tokenizer)}")

    # 创建pipeline（关键修改点）
    try:
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
    except Exception as e:
        st.error(f"Pipeline创建失败: {str(e)}")
        st.stop()

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

    # 付费功能的说明和按钮
    st.markdown("---")
    st.markdown("**风险声明：AI生成的投资建议需经持牌分析师复核并留痕，我公司AI生成的投资建议仅供参考，若需人工审核，请转至付费板块。——资料来源：《证券基金经营机构信息技术管理办法》与《生成式人工智能服务安全基本要求》**")
    st.markdown("---")
    st.markdown("**付费功能**")
    st.markdown("以下是我们的付费功能：")
    st.markdown("- **智能投顾助手**：为您提供专业的投资建议和资产配置方案。（首月仅需9.9元）")
    st.markdown("- **AI制作PPT**：根据您的需求自动生成高质量的PPT。（一次制作仅需5元）")
    st.markdown("- **论文查重**：为您提供快速准确的论文查重服务。（一万字10元，我们拥有远超其他平台的品质和极高的性价比）")
    st.markdown("- **人工审核**：我公司持牌分析师为您量化定制投资建议，最大化减少风险。")

    st.markdown("---")
    st.markdown("**立即付费**")
    st.markdown("[前往付费页面](https://mtcuqf2rh8tvrdkyvgyjm2.streamlit.app/)")

    # 付费凭证验证部分
    if "paid_code_verified" not in st.session_state:
        st.session_state.paid_code_verified = False

    if not st.session_state.paid_code_verified:
        st.markdown("如果您已经是付费用户，请输入您对应付费功能的凭证：")
        paid_code = st.text_input("付费凭证")
        if st.button("验证"):
            if paid_code == "your_paid_code":  # 替换为实际的付费凭证验证逻辑
                st.session_state.paid_code_verified = True
                st.success("验证成功！您已成功解锁付费功能。")
            else:
                st.error("验证失败，请检查您的付费凭证。")
    else:
         st.markdown(f'''
            <a href="https://mtcuqf2rh8tvrdkyvgyjm2.streamlit.app/" target="_blank" style="display: inline-block; padding: 0.5em 1em; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px;">
            进入付费功能页面
            </a>
          ''', unsafe_allow_html=True)

# 主程序
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_register()  # 显示登录/注册界面
    else:
        main()  # 显示主应用界面

        # 退出登录
        if st.sidebar.button("退出登录"):
            st.session_state.logged_in = False
            st.rerun()  # 重新运行应用以显示登录界面
