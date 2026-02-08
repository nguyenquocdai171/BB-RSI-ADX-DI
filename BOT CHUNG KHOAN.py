import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import random
from datetime import datetime, timedelta

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

    /* DISCLAIMER */
    .disclaimer-box {
        background-color: #1E1E1E; border: 1px solid #444; border-radius: 8px;
        padding: 20px; margin: 0 auto 30px auto; text-align: center; max-width: 800px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .disclaimer-title { color: #FF5252; font-weight: bold; font-size: 1rem; text-transform: uppercase; margin-bottom: 12px; letter-spacing: 1px; }
    .d-line-1 { color: #AAA; font-size: 0.95rem; margin-bottom: 5px; }
    .d-line-2 { color: #E0E0E0; font-size: 1rem; font-weight: bold; margin-bottom: 5px; text-decoration: underline; text-decoration-color: #555; }
    .d-line-3 { color: #888; font-size: 0.85rem; font-style: italic; }

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
    
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; height: 50px; font-size: 1.1rem; }
    
    /* BACKTEST RESULT BOX */
    .backtest-box {
        background: linear-gradient(135deg, #263238 0%, #37474F 100%);
        border-radius: 10px; padding: 20px; margin-top: 20px; text-align: center;
        border: 1px solid #546E7A;
    }
    .backtest-label { color: #CFD8DC; font-size: 1rem; margin-bottom: 5px; }
    .backtest-val { color: #00E676; font-size: 2.5rem; font-weight: 900; }
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

# --- HÀM VẼ GIAO DIỆN CHỈ SỐ ---
def render_metric_card(label, value, delta=None, color=None):
    delta_html = ""
    if delta is not None:
        delta_color = "#00E676" if delta > 0 else ("#FF5252" if delta < 0 else "#888")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "")
        delta_val = f"{abs(delta):.1f}"
        delta_html = f"<div style='font-size:0.9rem; margin-top:5px; color:{delta_color}'>{arrow} {delta_val} vs phiên trước</div>"
    
    if color:
        value_html = f"<div class='trend-badge' style='background-color:{color}'>{value}</div>"
    else:
        value_html = f"<div class='metric-value'>{value}</div>"

    card_html = f"<div class='metric-container'><div class='metric-label'>{label}</div><div class='metric-value-box'>{value_html}{delta_html}</div></div>"
    st.markdown(card_html, unsafe_allow_html=True)

# --- LOGIC CHIẾN LƯỢC ---
def check_signals(curr, prev, prev2):
    price = curr['Close']; rsi = curr['RSI']; adx = curr['ADX']
    lower_band = curr['Lower']; upper_band = curr['Upper']
    
    buy_trigger = (price <= lower_band * 1.01) and (rsi < 30)
    if buy_trigger:
        if adx < 25:
            if (curr['-DI'] > curr['+DI']) and (curr['-DI'] < prev['-DI']): return 1 
        elif adx > 50:
            if (curr['ADX'] < prev['ADX'] < prev2['ADX']) and (curr['-DI'] < prev['-DI'] < prev2['-DI']): return 1
        else: 
            if (curr['-DI'] > curr['+DI']) and (curr['-DI'] < prev['-DI']): return 1
            
    sell_trigger = (price >= upper_band * 0.99) and (rsi > 70)
    if sell_trigger:
        if adx < 25:
            if (curr['+DI'] > curr['-DI']) and (curr['+DI'] < prev['+DI']): return -1
        elif adx > 50:
            if (curr['ADX'] < prev['ADX'] < prev2['ADX']) and (curr['+DI'] < prev['+DI'] < prev2['+DI']): return -1
        else:
            if (curr['+DI'] > curr['-DI']) and (curr['+DI'] < prev['+DI']): return -1
            
    return 0

# --- HÀM PHÂN TÍCH HIỆN TẠI ---
def analyze_current_market(df):
    if len(df) < 25: return "Không đủ dữ liệu", "NEUTRAL", "gray", "Chưa đủ dữ liệu."
    curr = df.iloc[-1]; prev = df.iloc[-2]; prev2 = df.iloc[-3]
    
    signal = check_signals(curr, prev, prev2)
    
    rec, reason, color_class = "QUAN SÁT (HOLD)", "Chưa có tín hiệu giao dịch đặc biệt.", "bg-blue"
    
    if signal == 1:
        rec = "MUA NGAY"
        reason = "Giá chạm đáy BB, RSI thấp. Các chỉ báo ADX/DI cho tín hiệu đảo chiều tăng."
        color_class = "bg-green"
    elif signal == -1:
        rec = "BÁN NGAY"
        reason = "Giá chạm đỉnh BB, RSI cao. Các chỉ báo ADX/DI cho tín hiệu đảo chiều giảm."
        color_class = "bg-red"

    trend_state = "TĂNG" if curr['+DI'] > curr['-DI'] else "GIẢM"
    trend_strength = "YẾU (Sideway)" if curr['ADX'] < 25 else ("CỰC MẠNH" if curr['ADX'] > 50 else "TRUNG BÌNH")
    trend_color = "#00E676" if curr['+DI'] > curr['-DI'] else "#FF5252" 
    
    price_pos = "trong biên độ an toàn"
    if curr['Close'] <= curr['Lower'] * 1.01: price_pos = "<span style='color:#4CAF50; font-weight:bold'>chạm dải dưới (Rẻ)</span>"
    elif curr['Close'] >= curr['Upper'] * 0.99: price_pos = "<span style='color:#FF5252; font-weight:bold'>chạm dải trên (Đắt)</span>"
    
    rsi_state = "Trung tính"
    if curr['RSI'] < 30: rsi_state = "<span style='color:#4CAF50; font-weight:bold'>QUÁ BÁN (Cơ hội)</span>"
    elif curr['RSI'] > 70: rsi_state = "<span style='color:#FF5252; font-weight:bold'>QUÁ MUA (Rủi ro)</span>"

    report = f"""
    <div class='report-box'>
        <div class='report-header'>📝 PHÂN TÍCH CHI TIẾT</div>
        <div class='report-item'><span class='icon-dot'>🌊</span> <span>Xu hướng: Thị trường đang <b style='color:{trend_color}'>{trend_state}</b> với cường độ <b>{trend_strength}</b> (ADX={curr['ADX']:.1f}).</span></div>
        <div class='report-item'><span class='icon-dot'>📍</span> <span>Vị thế giá: Giá hiện tại đang {price_pos} của Bollinger Bands.</span></div>
        <div class='report-item'><span class='icon-dot'>🚀</span> <span>Động lượng: Chỉ số RSI đạt <b>{curr['RSI']:.1f}</b>, trạng thái {rsi_state}.</span></div>
        <div class='report-item'><span class='icon-dot'>⚖️</span> <span>Tín hiệu ADX/DI: { "Phe Mua đang kiểm soát (+DI > -DI)" if curr['+DI'] > curr['-DI'] else "Phe Bán đang kiểm soát (-DI > +DI)" }.</span></div>
    </div>
    """
    return rec, reason, color_class, report

# --- HÀM BACKTEST ---
def run_simulation(df, stop_loss_pct):
    initial_capital = 100_000_000
    cash = initial_capital
    shares = 0
    position = False
    entry_price = 0
    
    # 0 = Tắt
    use_sl = stop_loss_pct > 0
    
    for i in range(50, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        price = curr['Close']
        
        if position:
            if use_sl:
                pct_change = (price - entry_price) / entry_price
                if pct_change <= -(stop_loss_pct / 100.0):
                    cash += shares * price * (1 - 0.0015)
                    shares = 0
                    position = False
                    continue
            
            signal = check_signals(curr, prev, prev2)
            if signal == -1:
                cash += shares * price * (1 - 0.0015)
                shares = 0
                position = False
                continue
        
        if not position:
            signal = check_signals(curr, prev, prev2)
            if signal == 1:
                shares = int(cash / price)
                if shares > 0:
                    cash -= shares * price * (1 + 0.0015)
                    entry_price = price
                    position = True
    
    final_val = cash
    if position:
        final_val += shares * df.iloc[-1]['Close']
    
    total_return_pct = ((final_val - initial_capital) / initial_capital) * 100
    
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    avg_annual_return = total_return_pct / years if years > 0 else 0
    
    return total_return_pct, avg_annual_return

# --- GIAO DIỆN CHÍNH ---
st.markdown("<h1 class='main-title'>STOCK ADVISOR PRO</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Hệ thống Hỗ trợ Phân tích & Quản trị Rủi ro Đầu tư</p>", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer-box'>
    <div class='disclaimer-title'>⚠️ TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM</div>
    <div class='d-line-1'>Công cụ sử dụng thuật toán kỹ thuật (BB, RSI, ADX) để hỗ trợ tham khảo.</div>
    <div class='d-line-2'>KHÔNG phải lời khuyên đầu tư tài chính chính thức.</div>
    <div class='d-line-3'>Người dùng tự chịu trách nhiệm. Dữ liệu Yahoo Finance (Trễ 15p).</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1]) 
with col2:
    with st.form(key='search_form'):
        # Quay về giao diện 2 cột đơn giản, dễ nhìn, không lệch
        c_ticker, c_sl = st.columns([2, 1])
        
        with c_ticker:
            ticker_input = st.text_input("Mã cổ phiếu:", value="", placeholder="VD: HPG, VNM...").upper()
            
        with c_sl:
            stop_loss_input = st.number_input("Cắt lỗ % (0 = Tắt):", min_value=0.0, max_value=20.0, value=7.0, step=0.5)
            
        submit_button = st.form_submit_button(label='🚀 PHÂN TÍCH & BACKTEST', use_container_width=True)

if submit_button or 'data' in st.session_state:
    js_hack = f"""<script>function forceBlur(){{const activeElement=window.parent.document.activeElement;if(activeElement){{activeElement.blur();}}window.parent.document.body.focus();}}forceBlur();setTimeout(forceBlur,200);</script><div style="display:none;">{random.random()}</div>"""
    components.html(js_hack, height=0)

    if submit_button:
        ticker = ticker_input.strip()
        st.session_state['ticker'] = ticker
        st.session_state['sl_pct'] = stop_loss_input
    elif 'ticker' in st.session_state:
        ticker = st.session_state['ticker']
        stop_loss_input = st.session_state.get('sl_pct', 7.0)

    if not ticker:
        st.warning("⚠️ Vui lòng nhập mã cổ phiếu!")
    else:
        symbol = ticker if ".VN" in ticker else f"{ticker}.VN"
        
        if 'data' not in st.session_state or st.session_state.get('current_symbol') != symbol:
            with st.spinner(f'Đang tải dữ liệu {ticker} và chạy Backtest...'):
                try:
                    df_full = yf.download(symbol, period="max", interval="1d", progress=False)
                    if df_full.empty:
                        st.error(f"❌ Không tìm thấy mã **{ticker}**!")
                        st.stop()
                    
                    if isinstance(df_full.columns, pd.MultiIndex): df_full.columns = df_full.columns.get_level_values(0)
                    df_full = calculate_indicators(df_full)
                    st.session_state['data'] = df_full
                    st.session_state['current_symbol'] = symbol
                    
                    df_intra = yf.download(symbol, period="1d", interval="5m", progress=False)
                    if isinstance(df_intra.columns, pd.MultiIndex): df_intra.columns = df_intra.columns.get_level_values(0)
                    if not df_intra.empty:
                        if df_intra.index.tzinfo is None:
                            df_intra.index = df_intra.index + timedelta(hours=7)
                        else:
                            df_intra.index = df_intra.index.tz_convert('Asia/Ho_Chi_Minh')
                    st.session_state['data_intra'] = df_intra

                except Exception as e:
                    st.error(f"Lỗi tải dữ liệu: {e}")
                    st.stop()

        try:
            df = st.session_state['data']
            df_intra = st.session_state['data_intra']
            
            rec, reason, bg_class, report = analyze_current_market(df)
            curr = df.iloc[-1]; prev = df.iloc[-2]

            st.markdown(f"<div class='result-card {bg_class}'><div class='result-title'>{rec}</div><div class='result-reason'>💡 Lý do: {reason}</div></div>", unsafe_allow_html=True)
            
            # BACKTEST RESULT
            total_return, avg_return = run_simulation(df, stop_loss_input)
            bk_color = "#00E676" if avg_return > 0 else "#FF5252"
            sl_text = f"Stoploss {stop_loss_input}%" if stop_loss_input > 0 else "KHÔNG Cắt Lỗ"
            
            st.markdown(f"""
            <div class='backtest-box'>
                <div style='display:flex; justify-content:space-around; align-items:center;'>
                    <div>
                        <div class='backtest-label'>TB NĂM (Backtest)</div>
                        <div class='backtest-val' style='color:{bk_color}'>{avg_return:+.1f}%</div>
                        <div style='font-size:0.8rem; color:#AAA;'>({sl_text})</div>
                    </div>
                    <div style='border-left:1px solid #546E7A; height:50px;'></div>
                    <div>
                        <div class='backtest-label'>TỔNG LỢI NHUẬN</div>
                        <div class='backtest-val' style='font-size:1.8rem; color:{bk_color}'>{total_return:+.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(report, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # BIỂU ĐỒ INTRADAY
            if not df_intra.empty:
                st.divider()
                latest_date = df_intra.index[0].strftime('%d/%m/%Y')
                st.markdown(f"### ⏱️ Diễn biến giá trong ngày ({latest_date}) - {ticker}")
                ref_price = df['Close'].iloc[-2]
                current_price = df_intra['Close'].iloc[-1]
                line_color = '#00E676' if current_price >= ref_price else '#FF5252'
                fig_intra = go.Figure()
                fig_intra.add_trace(go.Scatter(x=df_intra.index, y=df_intra['Close'], mode='lines', line=dict(color=line_color, width=2), name='Giá Intraday'))
                fig_intra.update_layout(height=350, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', tickformat="%H:%M"), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', autorange=True))
                st.plotly_chart(fig_intra, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
            else: st.info("⚠️ Chưa có dữ liệu Intraday.")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1: render_metric_card("GIÁ ĐÓNG CỬA", f"{curr['Close']:,.0f}", curr['Close'] - prev['Close'])
            with col_m2: render_metric_card("RSI (14)", f"{curr['RSI']:.1f}", curr['RSI'] - prev['RSI'])
            with col_m3: render_metric_card("ADX (14)", f"{curr['ADX']:.1f}", curr['ADX'] - prev['ADX'])
            with col_m4:
                trend_txt = "TĂNG" if curr['+DI'] > curr['-DI'] else "GIẢM"
                render_metric_card("XU HƯỚNG", trend_txt, None, color="#00E676" if trend_txt == "TĂNG" else "#FF5252")

            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()
            
            # --- BIỂU ĐỒ KỸ THUẬT ---
            st.markdown(f"### 📊 Biểu đồ Kỹ Thuật ({ticker})")
            st.caption(f"ℹ️ Điều chỉnh khung thời gian bên dưới sẽ áp dụng cho cả Biểu đồ Giá, RSI và ADX:")
            time_tabs = st.radio("Chọn khung thời gian:", ["1 Tháng", "3 Tháng", "6 Tháng", "1 Năm", "3 Năm", "Tất cả"], horizontal=True, index=3)
            
            df_chart = df.copy()
            if time_tabs == "1 Tháng": df_chart = df.iloc[-22:]
            elif time_tabs == "3 Tháng": df_chart = df.iloc[-66:]
            elif time_tabs == "6 Tháng": df_chart = df.iloc[-132:]
            elif time_tabs == "1 Năm": df_chart = df.iloc[-252:]
            elif time_tabs == "3 Năm": df_chart = df.iloc[-756:]

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Upper'], line=dict(color='rgba(255,255,255,0.5)', width=1, dash='dash'), name="Upper Band"))
            fig1.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Lower'], line=dict(color='rgba(255,255,255,0.5)', width=1, dash='dash'), name="Lower Band"))
            fig1.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA20'], line=dict(color='#FF914D', width=1.5), name="SMA 20"))
            fig1.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Giá"))
            fig1.update_layout(height=500, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=40), legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5), xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333'), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', autorange=True))
            st.plotly_chart(fig1, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("### 🚀 Chỉ số RSI")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#E040FB', width=2), name="RSI"))
                fig2.add_hline(y=70, line_dash="dot", line_color="#FF5252")
                fig2.add_hline(y=30, line_dash="dot", line_color="#00E676")
                fig2.update_layout(height=350, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=40), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333'), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', autorange=True))
                st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': False})

            with col_c2:
                st.markdown("### ⚖️ Chỉ số ADX & DI")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_chart.index, y=df_chart['ADX'], line=dict(color='white', width=2), name="ADX"))
                fig3.add_trace(go.Scatter(x=df_chart.index, y=df_chart['+DI'], line=dict(color='#00E676', width=1.5), name="+DI"))
                fig3.add_trace(go.Scatter(x=df_chart.index, y=df_chart['-DI'], line=dict(color='#FF5252', width=1.5), name="-DI"))
                fig3.add_hline(y=25, line_dash="dot", line_color="gray")
                fig3.update_layout(height=350, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=40), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333'), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', autorange=True))
                st.plotly_chart(fig3, use_container_width=True, config={'scrollZoom': False})

        except Exception as e:
            st.error(f"Đã xảy ra lỗi hiển thị: {e}")
