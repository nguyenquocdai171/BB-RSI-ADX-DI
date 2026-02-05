import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(layout="wide", page_title="Stock Advisor PRO", page_icon="📈")

# --- CSS TÙY CHỈNH (LÀM ĐẸP GIAO DIỆN DARK MODE) ---
st.markdown("""
<style>
    /* Chỉnh Font chữ toàn bộ web */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Tiêu đề chính */
    .main-title {
        text-align: center;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        margin-bottom: 10px;
        padding-top: 20px;
    }
    
    /* Sub-title */
    .sub-title {
        text-align: center;
        color: #aaaaaa;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }

    /* Khung báo cáo phân tích */
    .report-box {
        background-color: #262730; /* Màu nền card tối */
        border: 1px solid #41424C;
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .report-box h4 {
        color: #FF4B4B;
        border-bottom: 1px solid #41424C;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .report-box ul {
        list-style-type: none;
        padding: 0;
    }
    .report-box li {
        margin-bottom: 10px;
        font-size: 1.05rem;
    }
    .highlight {
        color: #FF914D;
        font-weight: bold;
    }

    /* Style cho Metric Box tùy chỉnh */
    .metric-container {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        height: 100%;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FFF;
    }
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .trend-badge {
        padding: 5px 15px;
        border-radius: 15px;
        font-weight: bold;
        color: white;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- HÀM TÍNH TOÁN (Giữ nguyên logic chuẩn) ---
def calculate_indicators(df):
    # 1. BB
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA20'] + (2 * df['StdDev'])
    df['Lower'] = df['SMA20'] - (2 * df['StdDev'])
    
    # 2. RSI (Wilder's Smoothing)
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
    
    rec, reason, color = "QUAN SÁT (HOLD)", "Chưa có tín hiệu giao dịch đặc biệt.", "blue"
    
    # --- LOGIC ---
    if buy_trigger:
        if adx < 25:
            if (di_minus > di_plus) and (di_minus < prev['-DI']):
                rec, reason, color = "MUA NGAY", "Giá chạm đáy BB, RSI thấp. Xu hướng giảm yếu và đang suy thoái.", "green"
            else:
                rec, reason, color = "CHỜ MUA", "Giá rẻ nhưng lực bán vẫn còn. Chờ DI- giảm.", "orange"
        elif adx > 50:
            cooling = (adx < prev['ADX'] < prev2['ADX']) and (di_minus < prev['-DI'] < prev2['-DI'])
            if cooling:
                rec, reason, color = "MUA NGAY", "Bắt đáy sau sập mạnh (ADX & DI- giảm 2 phiên).", "green"
            else:
                rec, reason, color = "ĐỨNG NGOÀI", f"Đang sập mạnh (ADX={adx:.1f}). Đừng bắt dao rơi!", "red"
        else:
             if (di_minus > di_plus) and (di_minus < prev['-DI']):
                rec, reason, color = "MUA THĂM DÒ", "Giá rẻ, xu hướng giảm trung bình.", "green"

    elif sell_trigger:
        if adx < 25:
             if (di_plus > di_minus) and (di_plus < prev['+DI']):
                rec, reason, color = "BÁN NGAY", "Giá đỉnh BB, RSI cao. Lực tăng yếu.", "red"
        elif adx > 50:
            cooling = (adx < prev['ADX'] < prev2['ADX']) and (di_plus < prev['+DI'] < prev2['+DI'])
            if cooling:
                rec, reason, color = "BÁN CHỐT LỜI", "Siêu sóng kết thúc (ADX & DI+ giảm 2 phiên).", "red"
            else:
                rec, reason, color = "NẮM GIỮ", f"Trend tăng cực mạnh (ADX={adx:.1f}). Gồng lãi!", "green"
        else:
             rec, reason, color = "CÂN NHẮC BÁN", "Vùng quá mua, cân nhắc chốt lời.", "orange"

    # --- REPORT TEXT HTML ---
    trend_state = "TĂNG" if di_plus > di_minus else "GIẢM"
    trend_strength = "YẾU (Sideway)" if adx < 25 else ("CỰC MẠNH" if adx > 50 else "TRUNG BÌNH")
    
    price_pos = "trong biên độ an toàn"
    if price <= lower_band * 1.01: price_pos = "<span class='highlight'>chạm dải dưới (Rẻ)</span>"
    elif price >= upper_band * 0.99: price_pos = "<span class='highlight'>chạm dải trên (Đắt)</span>"
    
    rsi_state = "Trung tính"
    if rsi < 30: rsi_state = "<span class='highlight'>QUÁ BÁN (Cơ hội)</span>"
    elif rsi > 70: rsi_state = "<span class='highlight'>QUÁ MUA (Rủi ro)</span>"
    
    trend_color = "#4CAF50" if di_plus > di_minus else "#FF5252" # Xanh/Đỏ cho xu hướng

    report = f"""
    <div class='report-box'>
        <h4>📝 PHÂN TÍCH CHI TIẾT</h4>
        <ul>
            <li><b>Xu hướng:</b> Thị trường đang <b style='color:{trend_color}'>{trend_state}</b> với cường độ <b>{trend_strength}</b> (ADX={adx:.1f}).</li>
            <li><b>Vị thế giá:</b> Giá hiện tại đang {price_pos} của Bollinger Bands.</li>
            <li><b>Động lượng (RSI):</b> Chỉ số RSI đạt <b>{rsi:.1f}</b>, trạng thái <b>{rsi_state}</b>.</li>
            <li><b>Tín hiệu ADX/DI:</b> { "Phe Mua đang kiểm soát (+DI > -DI)" if di_plus > di_minus else "Phe Bán đang kiểm soát (-DI > +DI)" }.</li>
        </ul>
    </div>
    """
             
    return rec, reason, color, report

# --- HÀM VẼ GIAO DIỆN CHỈ SỐ (METRIC CARD) ---
def render_metric_card(label, value, delta=None, color=None):
    delta_html = ""
    if delta is not None:
        delta_color = "#4CAF50" if delta > 0 else ("#FF5252" if delta < 0 else "#888")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "")
        delta_html = f"<div class='metric-delta' style='color: {delta_color};'>{arrow} {abs(delta):.1f}</div>"
    
    # Custom Trend Badge
    value_html = f"<div class='metric-value'>{value}</div>"
    if color: # Nếu là badge Xu hướng
        value_html = f"<div class='trend-badge' style='background-color: {color};'>{value}</div>"

    st.markdown(f"""
    <div class='metric-container'>
        <div class='metric-label'>{label}</div>
        {value_html}
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# --- GIAO DIỆN CHÍNH ---

st.markdown("<h1 class='main-title'>STOCK ADVISOR PRO</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Hệ thống săn tìm điểm đảo chiều: BB + RSI + ADX + DI</p>", unsafe_allow_html=True)

# 1. FORM NHẬP LIỆU (CĂN GIỮA)
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

# LOGIC KHI SUBMIT
if submit_button:
    try:
        ticker = ticker_input.strip()
        symbol = ticker if ".VN" in ticker else f"{ticker}.VN"
        
        with st.spinner(f'Đang tải dữ liệu {ticker}...'):
            data = yf.download(symbol, period="1y", interval="1d", progress=False)
            
            if data.empty:
                st.error(f"❌ Không tìm thấy mã **{ticker}**!")
            else:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                
                df = calculate_indicators(data)
                rec, reason, color, report = analyze_strategy(df)
                curr = df.iloc[-1]
                prev = df.iloc[-2]
                
                # --- PHẦN 1: KẾT LUẬN (ALERT) ---
                st.write("") # Spacer
                if color == 'green': st.success(f"## {rec}")
                elif color == 'red': st.error(f"## {rec}")
                elif color == 'orange': st.warning(f"## {rec}")
                else: st.info(f"## {rec}")
                
                st.markdown(f"**💡 Lý do:** {reason}")
                
                # --- PHẦN 2: BÁO CÁO CHI TIẾT ---
                st.markdown(report, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True) # Khoảng cách lớn
                
                # --- PHẦN 3: CÁC CHỈ SỐ KỸ THUẬT (ĐÓNG KHUNG) ---
                st.markdown("### 🔢 Các Chỉ Số Kỹ Thuật (Phiên Hiện Tại)")
                
                # Sử dụng Container có viền (Streamlit mới hỗ trợ border)
                with st.container(border=True):
                    m1, m2, m3, m4 = st.columns(4)
                    
                    # 1. Giá
                    price_delta = curr['Close'] - prev['Close']
                    with m1:
                        render_metric_card("Giá Đóng Cửa", f"{curr['Close']:,.0f}", price_delta)
                    
                    # 2. RSI (Có so sánh)
                    rsi_delta = curr['RSI'] - prev['RSI']
                    with m2:
                        render_metric_card("RSI (14)", f"{curr['RSI']:.1f}", rsi_delta)
                    
                    # 3. ADX (Có so sánh)
                    adx_delta = curr['ADX'] - prev['ADX']
                    with m3:
                        render_metric_card("ADX (14)", f"{curr['ADX']:.1f}", adx_delta)
                    
                    # 4. Xu hướng (Màu sắc)
                    trend_txt = "TĂNG" if curr['+DI'] > curr['-DI'] else "GIẢM"
                    trend_bg = "#4CAF50" if trend_txt == "TĂNG" else "#FF5252"
                    with m4:
                        render_metric_card("Xu Hướng Chính", trend_txt, None, color=trend_bg)

                # --- PHẦN 4: BIỂU ĐỒ ---
                st.markdown("<br>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"### 📉 Biểu Đồ Kỹ Thuật: {ticker}")
                
                # Cấu hình biểu đồ tối ưu cho Dark Mode
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03,
                                   subplot_titles=("Giá & Bollinger Bands", "RSI (14)", "ADX & DI"))
                
                # Chart 1
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Giá"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='rgba(200,200,200,0.5)', width=1, dash='dash'), name="Upper"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='rgba(200,200,200,0.5)', width=1, dash='dash'), name="Lower"), row=1, col=1)
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
                
                # Layout Chart Dark Mode
                fig.update_layout(height=900, xaxis_rangeslider_visible=False, 
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', # Nền trong suốt để ăn theo theme
                                  font=dict(color='#FAFAFA'),
                                  margin=dict(l=20, r=20, t=40, b=20))
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
                
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi hệ thống: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.8em;'>⚠️ Công cụ hỗ trợ phân tích kỹ thuật. Dữ liệu từ Yahoo Finance.</p>", unsafe_allow_html=True)
