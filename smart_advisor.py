import streamlit as st
import pandas as pd


def load_excel(file_path):
    """加载股票Excel文件"""
    return pd.read_excel(file_path)


def filter_stocks(df, risk_preference, asset_size):
    """根据用户风险偏好和资产规模筛选股票"""
    # 定义风险偏好筛选条件
    if risk_preference == "厌恶风险":
        # 筛选低风险股票：K、D、J指标较低，RSI也不高
        filtered = df[(df['K'] < 30) & (df['D'] < 30) & (df['J'] < 30) & (df['RSI'] < 50)]
    elif risk_preference == "中立":
        # 筛选中等风险股票：K、D、J指标中等，RSI在合理区间
        filtered = df[(df['K'].between(30, 70)) & (df['D'].between(30, 70)) &
                      (df['J'].between(30, 70)) & (df['RSI'].between(30, 70))]
    elif risk_preference == "偏爱风险":
        # 筛选高风险股票：K、D、J指标较高，RSI也高
        filtered = df[(df['K'] > 70) & (df['D'] > 70) & (df['J'] > 70) & (df['RSI'] > 50)]
    else:
        raise ValueError("风险偏好输入错误，请输入：厌恶风险、中立、偏爱风险")

    # 根据资产规模进一步筛选
    if asset_size < 100000:  # 假设资产规模非常有限，只推荐低风险股票
        filtered = filtered[filtered['RSI'] < 30]
    elif 100000 <= asset_size < 1000000:  # 资产中等，推荐中等风险股票
        filtered = filtered[filtered['RSI'].between(30, 70)]
    else:  # 资产规模大，可承担高风险
        filtered = filtered[filtered['RSI'] > 70]

    return filtered[['股票代码', 'K', 'D', 'J', 'RSI']].drop_duplicates()
def main():
    # 页面配置
    st.set_page_config(
        page_title="智能股票推荐系统",
        page_icon="📈",
        layout="wide"
    )

    # 标题和说明
    st.title("📈 智能股票推荐系统")
    st.markdown("""
    ### 我们是由中央财经大学财智AI团队（团队成员：周强、王悦、谢佳益.....）研发的软件，根据您的风险偏好和资产规模，为您推荐合适股票
    填写以下参数获取个性化推荐 →
    """)

    # 创建输入表单
    with st.sidebar:
        st.header("用户参数设置")
        asset_size = st.number_input("资产总额（元）",
                                     min_value=1000,
                                     max_value=100000000,
                                     value=100000,
                                     step=10000)

        risk_preference = st.selectbox(
            "风险偏好",
            ("厌恶风险", "中立", "偏爱风险"),
            index=1
        )

        st.markdown("---")
        st.caption("参数说明：")
        st.caption("- 厌恶风险：优先选择低波动性股票")
        st.caption("- 中立：平衡收益与风险")
        st.caption("- 偏爱风险：追求高收益，接受高波动")

    # 加载数据
    try:
        df = load_excel('all_stocks_predictions.xlsx')  # 确保文件路径正确
    except FileNotFoundError:
        st.error("错误：股票数据文件未找到！")
        return

    # 显示加载状态
    with st.spinner('正在筛选优质股票...'):
        result = filter_stocks(df, risk_preference, asset_size)

    # 显示结果
    st.subheader("推荐股票列表")
    if not result.empty:
        # 添加样式
        st.dataframe(
            result.style
            .highlight_max(subset=['RSI'], color='#FF7043')
            .format({'K': "{:.2f}", 'D': "{:.2f}",
                     'J': "{:.2f}", 'RSI': "{:.2f}"}),
            use_container_width=True
        )

        # 显示统计信息
        col1, col2 = st.columns(2)
        with col1:
            st.metric("推荐股票数量", len(result))
        with col2:
            st.metric("平均RSI指标", f"{result['RSI'].mean():.2f}")
    else:
        st.warning("未找到符合条件的股票，请尝试调整筛选条件！")


if __name__ == "__main__":
    main()
