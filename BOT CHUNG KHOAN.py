import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(layout="wide", page_title="Stock Advisor PRO", page_icon="📈")

# --- CSS TÙY CHỈNH (GIAO DIỆN CAO CẤP) ---
st.markdown("""
<style>
    /* 1. IMPORT FONT HIỆN ĐẠI */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', 'Segoe UI', sans-serif;
    }
    
    /* 2. HEADER */
    .main-title {
        text-align: center;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* 3. RESULT CARD (THẺ KẾT LUẬN) - ĐẸP HƠN */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    /* Màu sắc động cho thẻ kết luận */
    .bg-green { background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); }
    .bg-red { background: linear-gradient(135deg, #b71c1c 0%, #c62828 100%); }
    .bg-orange { background: linear-gradient(135deg, #e65100 0%, #ef6c00 100%); }
    .bg-blue { background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%); }

    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .result-reason {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        margin-top: 10px;
        font-style: italic;
    }

    /* 4. REPORT BOX (PHÂN TÍCH CHI TIẾT) */
    .report-box {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 25px;
        margin-top: 10px;
    }
    .report-header {
        color: #FF914D;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 1px solid #333;
        padding-bottom: 10px;
    }
    .report-item {
        margin-bottom: 12px;
        font-size: 1rem;
        color: #E0E0E0;
        display: flex;
        align-items: center;
    }
    .icon-dot {
        margin-right: 10px;
        font-size: 1.2rem;
    }

    /* 5. METRIC CARDS */
    .metric-container {
        background-color: #262730;
        border: 1px solid #41424C;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        height: 100%;
        transition: transform 0.2s;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: #FF914D;
    }
    .metric-label { font-size: 0.9rem; color: #AAA; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 2rem; font-weight: 900; color: #FFF; margin-bottom: 5px; }
    .trend-badge { padding: 4px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: bold; color: white; display: inline-block; }
    
</style>
""", unsafe_allow_html=True)

# --- HÀM TÍNH TOÁN ---
def calculate_indicators(df):
    # 1. BB
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA20'] + (2 * df['StdDev'])
    df['Lower'] = df['SMA20'] - (2 * df['StdDev'])
    
    # 2. RSI (Wilder's)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. ADX/DI
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    
    df['TR14'] = df['TR'].ewm(alpha=1/14, adjust=False).mean()
    df['+DM14'] = df['+DM'].ewm(alpha=1/14, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=1/14, adjust=False).mean()
    
    df['+DI'] = 100 * (df['+DM14'] / df['TR14'])
    df['-DI'] = 100 * (df['-DM14'] / df['TR14'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].ewm(alpha=1/14, adjust=False).mean()
    
    return df

# --- LOGIC MUA BÁN ---
def analyze_strategy(df):
    if len(df) < 25: return "Không đủ dữ liệu", "NEUTRAL", "gray", "Chưa đủ dữ liệu."
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    # Values
    price = curr['Close']
    rsi = curr['RSI']
    adx = curr['ADX']
    di_plus = curr['+DI']
    di_minus = curr['-DI']
    lower_band = curr['Lower']
    upper_band = curr['Upper']

    # Triggers
    buy_trigger = (price <= lower_band * 1.01) and (rsi < 30)
    sell_trigger = (price >= upper_band * 0.99) and (rsi > 70)
    
    rec, reason, color_class = "QUAN SÁT (HOLD)", "Chưa có tín hiệu giao dịch đặc biệt.", "bg-blue"
    
    # --- LOGIC ---
    if buy_trigger:
        if adx < 25:
            if (di_minus > di_plus) and (di_minus < prev['-DI']):
                rec, reason, color_class = "MUA NGAY", "Giá chạm đáy BB, RSI thấp. Xu hướng giảm yếu và đang suy thoái.", "bg-green"
            else:
                rec, reason, color_class = "CHỜ MUA", "Giá rẻ nhưng lực bán vẫn còn. Chờ DI- giảm.", "bg-orange"
        elif adx > 50:
            cooling = (adx < prev['ADX'] < prev2['ADX']) and (di_minus < prev['-DI'] < prev2['-DI'])
            if cooling:
                rec, reason, color_class = "MUA NGAY", "Bắt đáy sau sập mạnh (ADX & DI- giảm 2 phiên).", "bg-green"
            else:
                rec, reason, color_class = "ĐỨNG NGOÀI", f"Đang sập mạnh (ADX={adx:.1f}). Đừng bắt dao rơi!", "bg-red"
        else:
             if (di_minus > di_plus) and (di_minus < prev['-DI']):
                rec, reason, color_class = "MUA THĂM DÒ", "Giá rẻ, xu hướng giảm trung bình.", "bg-green"

    elif sell_trigger:
        if adx < 25:
             if (di_plus > di_minus) and (di_plus < prev['+DI']):
                rec, reason, color_class = "BÁN NGAY", "Giá đỉnh BB, RSI cao. Lực tăng yếu.", "bg-red"
        elif adx > 50:
            cooling = (adx < prev['ADX'] < prev2['ADX']) and (di_plus < prev['+DI'] < prev2['+DI'])
            if cooling:
                rec, reason, color_class = "BÁN CHỐT LỜI", "Siêu sóng kết thúc (ADX & DI+ giảm 2 phiên).", "bg-red"
            else:
                rec, reason, color_class = "NẮM GIỮ", f"Trend tăng cực mạnh (ADX={adx:.1f}). Gồng lãi!", "bg-green"
        else:
             rec, reason, color_class = "CÂN NHẮC BÁN", "Vùng quá mua, cân nhắc chốt lời.", "bg-orange"

    # --- REPORT TEXT HTML (CUSTOM STYLE) ---
    trend_state = "TĂNG" if di_plus > di_minus else "GIẢM"
    trend_strength = "YẾU (Sideway)" if adx < 25 else ("CỰC MẠNH" if adx > 50 else "TRUNG BÌNH")
    
    price_pos = "trong biên độ an toàn"
    if price <= lower_band * 1.01: price_pos = "<span style='color:#4CAF50; font-weight:bold'>chạm dải dưới (Rẻ)</span>"
    elif price >= upper_band * 0.99: price_pos = "<span style='color:#FF5252; font-weight:bold'>chạm dải trên (Đắt)</span>"
    
    rsi_state = "Trung tính"
    if rsi < 30: rsi_state = "<span style='color:#4CAF50; font-weight:bold'>QUÁ BÁN (Cơ hội)</span>"
    elif rsi > 70: rsi_state = "<span style='color:#FF5252; font-weight:bold'>QUÁ MUA (Rủi ro)</span>"
    
    trend_color = "#4CAF50" if di_plus > di_minus else "#FF5252" # Xanh/Đỏ

    report = f"""
    <div class='report-box'>
        <div class='report-header'>📝 PHÂN TÍCH CHI TIẾT</div>
        <div class='report-item'><span class='icon-dot'>🌊</span> <span>Xu hướng: Thị trường đang <b style='color:{trend_color}'>{trend_state}</b> với cường độ <b>{trend_strength}</b> (ADX={adx:.1f}).</span></div>
        <div class='report-item'><span class='icon-dot'>📍</span> <span>Vị thế giá: Giá hiện tại đang {price_pos} của Bollinger Bands.</span></div>
        <div class='report-item'><span class='icon-dot'>🚀</span> <span>Động lượng: Chỉ số RSI đạt <b>{rsi:.1f}</b>, trạng thái {rsi_state}.</span></div>
        <div class='report-item'><span class='icon-dot'>⚖️</span> <span>Tín hiệu ADX/DI: { "Phe Mua đang kiểm soát (+DI > -DI)" if di_plus > di_minus else "Phe Bán đang kiểm soát (-DI > +DI)" }.</span></div>
    </div>
    """
             
    return rec, reason, color_class, report

# --- HÀM RENDER METRIC ---
def render_metric_card(label, value, delta=None, color=None):
    delta_html = ""
    if delta is not None:
        delta_color = "#4CAF50" if delta > 0 else ("#FF5252" if delta < 0 else "#888")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "")
        delta_html = f"<div style='font-size:0.9rem; margin-top:5px; color:{delta_color}'>{arrow} {abs(delta):.1f} vs phiên trước</div>"
    
    value_html = f"<div class='metric-value'>{value}</div>"
    if color: 
        value_html = f"<div class='trend-badge' style='background-color:{color}'>{value}</div>"

    st.markdown(f"""
    <div class='metric-container'>
        <div class='metric-label'>{label}</div>
        {value_html}
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# --- GIAO DIỆN CHÍNH ---

st.markdown("<h1 class='main-title'>STOCK ADVISOR PRO</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Công cụ săn tìm điểm đảo chiều thông minh</p>", unsafe_allow_html=True)

# FORM
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.form(key='search_form'):
        c_in, c_btn = st.columns([3, 1])
        with c_in:
            ticker_input = st.text_input("Mã cổ phiếu:", "HPG", placeholder="Ví dụ: VNM").upper()
        with c_btn:
            st.write("") 
            st.write("")
            submit_button = st.form_submit_button(label='🔍 PHÂN TÍCH')

if submit_button:
    try:
        ticker = ticker_input.strip()
        symbol = ticker if ".VN" in ticker else f"{ticker}.VN"
        
        with st.spinner(f'Đang phân tích dữ liệu {ticker}...'):
            data = yf.download(symbol, period="1y", interval="1d", progress=False)
            
            if data.empty:
                st.error(f"❌ Không tìm thấy mã **{ticker}**!")
            else:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                
                df = calculate_indicators(data)
                rec, reason, bg_class, report = analyze_strategy(df)
                curr = df.iloc[-1]
                prev = df.iloc[-2]
                
                # 1. KẾT QUẢ (CUSTOM CARD)
                st.markdown(f"""
                <div class='result-card {bg_class}'>
                    <div class='result-title'>{rec}</div>
                    <div class='result-reason'>💡 Lý do: {reason}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 2. PHÂN TÍCH CHI TIẾT
                st.markdown(report, unsafe_allow_html=True)
                
                # 3. CHỈ SỐ
                st.markdown("<br>", unsafe_allow_html=True)
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    render_metric_card("GIÁ ĐÓNG CỬA", f"{curr['Close']:,.0f}", curr['Close'] - prev['Close'])
                with col_m2:
                    render_metric_card("RSI (14)", f"{curr['RSI']:.1f}", curr['RSI'] - prev['RSI'])
                with col_m3:
                    render_metric_card("ADX (14)", f"{curr['ADX']:.1f}", curr['ADX'] - prev['ADX'])
                with col_m4:
                    trend_txt = "TĂNG" if curr['+DI'] > curr['-DI'] else "GIẢM"
                    trend_col = "#4CAF50" if trend_txt == "TĂNG" else "#FF5252"
                    render_metric_card("XU HƯỚNG", trend_txt, None, color=trend_col)

                # 4. BIỂU ĐỒ
                st.markdown("<br>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"### 📉 Biểu Đồ Kỹ Thuật: {ticker}")
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03,
                                   subplot_titles=("Giá & Bollinger Bands", "RSI (14)", "ADX & DI"))
                
                # Chart 1
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Giá"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'), name="Upper"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'), name="Lower"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#FF914D', width=1), name="SMA20"), row=1, col=1)

                # Chart 2
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#E040FB', width=2), name="RSI"), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="#FF5252")
                fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="#4CAF50")
                
                # Chart 3
                fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='white', width=2), name="ADX"), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], line=dict(color='#4CAF50', width=1), name="+DI"), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], line=dict(color='#FF5252', width=1), name="-DI"), row=3, col=1)
                fig.add_hline(y=25, line_dash="dot", row=3, col=1, line_color="gray")
                fig.add_hline(y=50, line_dash="dot", row=3, col=1, line_color="#FF5252")
                
                # Layout Tối ưu Dark Mode
                fig.update_layout(height=800, xaxis_rangeslider_visible=False, 
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='#FAFAFA'),
                                  margin=dict(l=10, r=10, t=30, b=10))
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
                
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi hệ thống: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.8em;'>⚠️ Dữ liệu từ Yahoo Finance (Trễ 15p). Công cụ chỉ mang tính tham khảo.</p>", unsafe_allow_html=True)
