import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import random

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(layout="wide", page_title="Stock Advisor PRO", page_icon="📈")

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', 'Segoe UI', sans-serif; }
    
    /* HEADER */
    .main-title {
        text-align: center; font-weight: 900;
        background: -webkit-linear-gradient(45deg, #00E676, #69F0AE); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3.5rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 2px;
        text-shadow: 0px 0px 20px rgba(0, 230, 118, 0.3);
    }
    .sub-title {
        text-align: center; color: #E0E0E0 !important; font-size: 1.2rem;
        font-weight: 400; margin-bottom: 20px; letter-spacing: 0.5px;
    }

    /* DISCLAIMER BOX (ĐÃ TÁCH DÒNG RÕ RÀNG) */
    .disclaimer-box {
        background-color: #1E1E1E; border: 1px solid #444; border-radius: 8px;
        padding: 20px; margin: 0 auto 30px auto; text-align: center; max-width: 800px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .disclaimer-title { color: #FF5252; font-weight: bold; font-size: 1rem; text-transform: uppercase; margin-bottom: 15px; letter-spacing: 1px; }
    .disclaimer-line { 
        color: #AAA; font-size: 0.95rem; line-height: 1.8; margin-bottom: 8px; 
        display: block; /* Xuống dòng */
    }
    .highlight-text { color: #E0E0E0; font-weight: 700; }

    /* RESULT CARD */
    .result-card {
        padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .bg-green { background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); }
    .bg-red { background: linear-gradient(135deg, #b71c1c 0%, #c62828 100%); }
    .bg-orange { background: linear-gradient(135deg, #e65100 0%, #ef6c00 100%); }
    .bg-blue { background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%); }
    .result-title { font-size: 2.2rem; font-weight: 800; color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
    .result-reason { font-size: 1.1rem; color: #EEE; margin-top: 10px; font-style: italic; }

    /* REPORT BOX */
    .report-box { background-color: #1E1E1E; border: 1px solid #444; border-radius: 12px; padding: 25px; margin-top: 10px; }
    .report-header { color: #00E676; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px; border-bottom: 1px solid #444; padding-bottom: 10px; text-transform: uppercase; }
    .report-item { margin-bottom: 12px; font-size: 1rem; color: #FAFAFA; display: flex; align-items: center; }
    .icon-dot { margin-right: 12px; font-size: 1.2rem; }

    /* METRIC CARDS */
    .metric-container {
        background-color: #262730; border: 1px solid #41424C; border-radius: 12px;
        padding: 15px 10px; text-align: center; height: 160px;
        display: flex; flex-direction: column; justify-content: flex-start; align-items: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-label { font-size: 0.9rem; color: #FFF; font-weight: 700; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; height: 20px; display: flex; align-items: center; }
    .metric-value-box { flex-grow: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    .metric-value { font-size: 2.2rem; font-weight: 900; color: #FFF; line-height: 1; }
    .trend-badge { padding: 10px 30px; border-radius: 30px; font-size: 1.3rem; font-weight: 900; color: white; display: inline-block; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
    
    /* BUTTON */
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; height: 50px; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# --- HÀM TÍNH TOÁN ---
def calculate_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA20'] + (2 * df['StdDev'])
    df['Lower'] = df['SMA20'] - (2 * df['StdDev'])
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
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
    curr = df.iloc[-1]; prev = df.iloc[-2]; prev2 = df.iloc[-3]
    price = curr['Close']; rsi = curr['RSI']; adx = curr['ADX']
    di_plus = curr['+DI']; di_minus = curr['-DI']
    lower_band = curr['Lower']; upper_band = curr['Upper']

    buy_trigger = (price <= lower_band * 1.01) and (rsi < 30)
    sell_trigger = (price >= upper_band * 0.99) and (rsi > 70)
    rec, reason, color_class = "QUAN SÁT (HOLD)", "Chưa có tín hiệu giao dịch đặc biệt.", "bg-blue"
    
    if buy_trigger:
        if adx < 25:
            if (di_minus > di_plus) and (di_minus < prev['-DI']): rec, reason, color_class = "MUA NGAY", "Giá chạm đáy BB, RSI thấp. Xu hướng giảm yếu.", "bg-green"
            else: rec, reason, color_class = "CHỜ MUA", "Giá rẻ nhưng lực bán vẫn còn. Chờ DI- giảm.", "bg-orange"
        elif adx > 50:
            cooling = (adx < prev['ADX'] < prev2['ADX']) and (di_minus < prev['-DI'] < prev2['-DI'])
            if cooling: rec, reason, color_class = "MUA NGAY", "Bắt đáy sau sập mạnh (ADX & DI- giảm 2 phiên).", "bg-green"
            else: rec, reason, color_class = "ĐỨNG NGOÀI", f"Đang sập mạnh (ADX={adx:.1f}). Đừng bắt dao rơi!", "bg-red"
        else:
             if (di_minus > di_plus) and (di_minus < prev['-DI']): rec, reason, color_class = "MUA THĂM DÒ", "Giá rẻ, xu hướng giảm trung bình.", "bg-green"
    elif sell_trigger:
        if adx < 25:
             if (di_plus > di_minus) and (di_plus < prev['+DI']): rec, reason, color_class = "BÁN NGAY", "Giá đỉnh BB, RSI cao. Lực tăng yếu.", "bg-red"
        elif adx > 50:
            cooling = (adx < prev['ADX'] < prev2['ADX']) and (di_plus < prev['+DI'] < prev2['+DI'])
            if cooling: rec, reason, color_class = "BÁN CHỐT LỜI", "Siêu sóng kết thúc (ADX & DI+ giảm 2 phiên).", "bg-red"
            else: rec, reason, color_class = "NẮM GIỮ", f"Trend tăng cực mạnh (ADX={adx:.1f}). Gồng lãi!", "bg-green"
        else: rec, reason, color_class = "CÂN NHẮC BÁN", "Vùng quá mua, cân nhắc chốt lời.", "bg-orange"

    trend_state = "TĂNG" if di_plus > di_minus else "GIẢM"
    trend_strength = "YẾU (Sideway)" if adx < 25 else ("CỰC MẠNH" if adx > 50 else "TRUNG BÌNH")
    price_pos = "trong biên độ an toàn"
    if price <= lower_band * 1.01: price_pos = "<span style='color:#4CAF50; font-weight:bold'>chạm dải dưới (Rẻ)</span>"
    elif price >= upper_band * 0.99: price_pos = "<span style='color:#FF5252; font-weight:bold'>chạm dải trên (Đắt)</span>"
    rsi_state = "Trung tính"
    if rsi < 30: rsi_state = "<span style='color:#4CAF50; font-weight:bold'>QUÁ BÁN (Cơ hội)</span>"
    elif rsi > 70: rsi_state = "<span style='color:#FF5252; font-weight:bold'>QUÁ MUA (Rủi ro)</span>"
    trend_color = "#00E676" if di_plus > di_minus else "#FF5252" 

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

def render_metric_card(label, value, delta=None, color=None):
    delta_html = ""
    if delta is not None:
        delta_color = "#00E676" if delta > 0 else ("#FF5252" if delta < 0 else "#888")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "")
        delta_html = f"<div style='font-size:0.9rem; margin-top:5px; color:{delta_color}'>{arrow} {abs(delta):.1f} vs phiên trước</div>"
    value_html = f"<div class='metric-value'>{value}</div>"
    if color: value_html = f"<div class='trend-badge' style='background-color:{color}'>{value}</div>"
    st.markdown(f"<div class='metric-container'><div class='metric-label'>{label}</div><div class='metric-value-box'>{value_html}{delta_html}</div></div>", unsafe_allow_html=True)

# --- GIAO DIỆN CHÍNH ---
st.markdown("<h1 class='main-title'>STOCK ADVISOR PRO</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Hệ thống Hỗ trợ Phân tích & Quản trị Rủi ro Đầu tư</p>", unsafe_allow_html=True)

# DISCLAIMER (ĐÃ TÁCH DÒNG RIÊNG BIỆT)
st.markdown("""
<div class='disclaimer-box'>
    <div class='disclaimer-title'>⚠️ TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM</div>
    <div class='disclaimer-line'>
        Công cụ sử dụng thuật toán kỹ thuật (BB, RSI, ADX) để hỗ trợ tham khảo.
    </div>
    <div class='disclaimer-line'>
        <span class='highlight-text'>KHÔNG</span> phải lời khuyên đầu tư tài chính chính thức.
    </div>
    <div class='disclaimer-line'>
        Người dùng tự chịu trách nhiệm. Dữ liệu Yahoo Finance (Trễ 15p).
    </div>
</div>
""", unsafe_allow_html=True)

# FORM
col1, col2, col3 = st.columns([1, 1.5, 1])
with col2:
    with st.form(key='search_form'):
        ticker_input = st.text_input("Nhập mã cổ phiếu:", value="", placeholder="Ví dụ: HPG, VNM, FPT...").upper()
        submit_button = st.form_submit_button(label='🚀 PHÂN TÍCH NGAY', use_container_width=True)

if submit_button:
    # --- JS HACK V2.0: CHẠY LẠI MỖI LẦN BẤM NÚT ---
    # Sử dụng key ngẫu nhiên để ép component reload mỗi lần
    js_hack = f"""
    <script>
        function blurInput() {{
            const inputs = window.parent.document.querySelectorAll('input[type="text"]');
            inputs.forEach(input => {{
                input.blur();
            }});
            window.parent.document.body.focus();
        }}
        setTimeout(blurInput, 100);
        setTimeout(blurInput, 500); // Chạy lại lần nữa cho chắc
    </script>
    <div style="display:none;">{random.random()}</div> 
    """
    components.html(js_hack, height=0)

    ticker = ticker_input.strip()
    
    if not ticker:
        st.warning("⚠️ Vui lòng nhập mã cổ phiếu!")
    else:
        symbol = ticker if ".VN" in ticker else f"{ticker}.VN"
        with st.spinner(f'Đang tải dữ liệu {ticker} (Full lịch sử)...'):
            try:
                # DỮ LIỆU MAX: Lấy toàn bộ lịch sử để Zoom
                data = yf.download(symbol, period="max", interval="1d", progress=False)
                if data.empty:
                    st.error(f"❌ Không tìm thấy mã **{ticker}**!")
                else:
                    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                    df = calculate_indicators(data)
                    rec, reason, bg_class, report = analyze_strategy(df)
                    curr = df.iloc[-1]; prev = df.iloc[-2]
                    
                    st.markdown(f"<div class='result-card {bg_class}'><div class='result-title'>{rec}</div><div class='result-reason'>💡 Lý do: {reason}</div></div>", unsafe_allow_html=True)
                    st.markdown(report, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1: render_metric_card("GIÁ ĐÓNG CỬA", f"{curr['Close']:,.0f}", curr['Close'] - prev['Close'])
                    with col_m2: render_metric_card("RSI (14)", f"{curr['RSI']:.1f}", curr['RSI'] - prev['RSI'])
                    with col_m3: render_metric_card("ADX (14)", f"{curr['ADX']:.1f}", curr['ADX'] - prev['ADX'])
                    with col_m4:
                        trend_txt = "TĂNG" if curr['+DI'] > curr['-DI'] else "GIẢM"
                        render_metric_card("XU HƯỚNG", trend_txt, None, color="#00E676" if trend_txt == "TĂNG" else "#FF5252")

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.divider()
                    
                    # CẤU HÌNH NÚT ZOOM (DÙNG CHUNG CHO CẢ 3 BIỂU ĐỒ)
                    zoom_config = dict(
                        buttons=list([
                            dict(count=1, label="1T", step="month", stepmode="backward"),
                            dict(count=3, label="3T", step="month", stepmode="backward"),
                            dict(count=6, label="6T", step="month", stepmode="backward"),
                            dict(count=1, label="1N", step="year", stepmode="backward"),
                            dict(count=3, label="3N", step="year", stepmode="backward"),
                            dict(count=5, label="5N", step="year", stepmode="backward"),
                            dict(step="all", label="Tất cả")
                        ]),
                        bgcolor="#262730", activecolor="#00E676", font=dict(color="white")
                    )

                    # --- CHART 1: GIÁ & BB ---
                    st.markdown(f"### 📊 Biểu đồ Giá & Bollinger Bands ({ticker})")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='rgba(255,255,255,0.5)', width=1, dash='dash'), name="Upper Band"))
                    fig1.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='rgba(255,255,255,0.5)', width=1, dash='dash'), name="Lower Band"))
                    fig1.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#FF914D', width=1.5), name="SMA 20"))
                    fig1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Giá"))
                    
                    # Mặc định hiển thị 1 năm gần nhất để trục Y đỡ bị nén
                    default_start = df.index[-252] if len(df) > 252 else df.index[0]
                    default_end = df.index[-1]

                    fig1.update_layout(
                        height=550, 
                        xaxis_rangeslider_visible=False,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=40),
                        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                        xaxis=dict(
                            showgrid=True, gridwidth=1, gridcolor='#333',
                            rangeselector=zoom_config,
                            range=[default_start, default_end] # Set mặc định 1 năm
                        )
                    )
                    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
                    st.plotly_chart(fig1, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': True})

                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        st.markdown("### 🚀 Chỉ số RSI (14)")
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#E040FB', width=2), name="RSI"))
                        fig2.add_hline(y=70, line_dash="dot", line_color="#FF5252")
                        fig2.add_hline(y=30, line_dash="dot", line_color="#00E676")
                        fig2.update_layout(height=350, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=40),
                                          legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                                          xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', rangeselector=zoom_config, range=[default_start, default_end]))
                        fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
                        fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
                        st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': False})

                    with col_c2:
                        st.markdown("### ⚖️ Chỉ số ADX & DI")
                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='white', width=2), name="ADX (Sức mạnh)"))
                        fig3.add_trace(go.Scatter(x=df.index, y=df['+DI'], line=dict(color='#00E676', width=1.5), name="+DI (Mua)"))
                        fig3.add_trace(go.Scatter(x=df.index, y=df['-DI'], line=dict(color='#FF5252', width=1.5), name="-DI (Bán)"))
                        fig3.add_hline(y=25, line_dash="dot", line_color="gray")
                        fig3.update_layout(height=350, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=40),
                                          legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                                          xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', rangeselector=zoom_config, range=[default_start, default_end]))
                        fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
                        fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
                        st.plotly_chart(fig3, use_container_width=True, config={'scrollZoom': False})

            except Exception as e:
                st.error(f"Đã xảy ra lỗi hệ thống: {e}")
