import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CẤU HÌNH ---
# Tắt các cảnh báo (Warning) gây rối mắt
import warnings
warnings.filterwarnings('ignore')

def calculate_indicators(df):
    """
    Tính toán Bollinger Bands, RSI, ADX, DI+, DI-
    """
    # 1. Bollinger Bands (20, 2)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA20'] + (2 * df['StdDev'])
    df['Lower'] = df['SMA20'] - (2 * df['StdDev'])

    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. ADX & DI (14)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)

    # Làm mượt (Smoothing)
    df['TR14'] = df['TR'].ewm(alpha=1/14, adjust=False).mean()
    df['+DM14'] = df['+DM'].ewm(alpha=1/14, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=1/14, adjust=False).mean()

    df['+DI'] = 100 * (df['+DM14'] / df['TR14'])
    df['-DI'] = 100 * (df['-DM14'] / df['TR14'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].ewm(alpha=1/14, adjust=False).mean()

    return df

def analyze_strategy(df):
    """
    Logic phân tích Mua/Bán (Phiên bản chuẩn BB + RSI + ADX)
    """
    if len(df) < 25:
        return "Không đủ dữ liệu", "NEUTRAL"

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    price = curr['Close']
    rsi = curr['RSI']
    adx = curr['ADX']
    di_plus = curr['+DI']
    di_minus = curr['-DI']
    lower_band = curr['Lower']
    upper_band = curr['Upper']

    recommendation = "QUAN SÁT (HOLD)"
    reason = "Chưa có tín hiệu đặc biệt."

    # --- 1. CHIẾN LƯỢC MUA (BẮT ĐÁY) ---
    buy_trigger = (price <= lower_band * 1.01) and (rsi < 30)

    if buy_trigger:
        if adx < 25:
            if (di_minus > di_plus) and (di_minus < prev['-DI']):
                recommendation = "MUA NGAY (BẮT ĐÁY)"
                reason = "Giá chạm dải dưới, RSI thấp. ADX thấp (<25). DI- đang suy yếu."
            else:
                recommendation = "CHỜ MUA"
                reason = "Thỏa điều kiện giá rẻ, nhưng lực bán (DI-) vẫn chưa giảm nhiệt."

        elif adx > 50:
            adx_cooling = (curr['ADX'] < prev['ADX']) and (prev['ADX'] < prev2['ADX'])
            dim_cooling = (curr['-DI'] < prev['-DI']) and (prev['-DI'] < prev2['-DI'])

            if adx_cooling and dim_cooling:
                recommendation = "MUA NGAY (BẮT ĐÁY)"
                reason = "Thị trường sập mạnh nhưng đà giảm đã gãy (ADX và DI- giảm 2 phiên liên tiếp)."
            else:
                recommendation = "KHÔNG MUA (CHỜ ĐỢI)"
                reason = f"Đang sập mạnh (ADX={adx:.1f}). Chờ ADX và DI- giảm 2 phiên liên tiếp."
        else:
            if (di_minus > di_plus) and (curr['-DI'] < prev['-DI']):
                recommendation = "MUA THĂM DÒ"
                reason = "Giá rẻ, xu hướng giảm trung bình. Có thể giải ngân từng phần."

    # --- 2. CHIẾN LƯỢC BÁN (CHỐT LỜI) ---
    elif (price >= upper_band * 0.99) and (rsi > 70):
        
        if adx < 25:
             if (di_plus > di_minus) and (di_plus < prev['+DI']):
                recommendation = "BÁN NGAY"
                reason = "Giá chạm đỉnh, RSI cao. ADX thấp, giá sẽ sớm đảo chiều."
        
        elif adx > 50:
            adx_cooling = (curr['ADX'] < prev['ADX']) and (prev['ADX'] < prev2['ADX'])
            dip_cooling = (curr['+DI'] < prev['+DI']) and (prev['+DI'] < prev2['+DI'])

            if adx_cooling and dip_cooling:
                recommendation = "BÁN NGAY (CHỐT LỜI)"
                reason = "Siêu sóng đã có dấu hiệu kết thúc (ADX và DI+ giảm 2 phiên liên tiếp)."
            else:
                recommendation = "NẮM GIỮ (GỒNG LÃI)"
                reason = f"Xu hướng tăng đang cực mạnh (ADX={adx:.1f}). Đừng bán non!"
        else:
             recommendation = "CÂN NHẮC BÁN"
             reason = "Giá đã vào vùng quá mua."

    return recommendation, reason

def main():
    print("\n" + "="*50)
    print("   TRỢ LÝ ĐẦU TƯ CHỨNG KHOÁN (TERMINAL VERSION)")
    print("="*50 + "\n")
    
    ticker_input = input("Nhập mã cổ phiếu (ví dụ HPG, VNM): ").upper().strip()
    if not ticker_input:
        print("Bạn chưa nhập mã nào cả.")
        return

    # Thêm đuôi .VN nếu thiếu
    ticker = ticker_input if ".VN" in ticker_input else f"{ticker_input}.VN"
    
    print(f"\n⏳ Đang tải dữ liệu và tính toán cho mã {ticker}...")

    try:
        # Tải dữ liệu
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        if data.empty:
            print("❌ Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã cổ phiếu.")
            return

        # Fix lỗi MultiIndex của yfinance mới
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Tính toán
        df = calculate_indicators(data)
        rec, reason = analyze_strategy(df)
        curr = df.iloc[-1]

        # In kết quả ra màn hình (Console)
        print("\n" + "-"*30)
        print(f"   KẾT QUẢ PHÂN TÍCH: {ticker}")
        print("-"*30)
        print(f"Giá hiện tại: {curr['Close']:,.0f}")
        print(f"RSI (14):     {curr['RSI']:.1f}")
        print(f"ADX (14):     {curr['ADX']:.1f}")
        print(f"Trạng thái:   {'+DI > -DI (Phe Mua)' if curr['+DI'] > curr['-DI'] else '-DI > +DI (Phe Bán)'}")
        print("-"*30)
        print(f"KHUYẾN NGHỊ:  >> {rec} <<")
        print(f"LÝ DO:        {reason}")
        print("-"*30)

        print("\n📈 Đang mở biểu đồ phân tích trong trình duyệt...")

        # Vẽ biểu đồ
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_heights=[0.5, 0.25, 0.25],
                            subplot_titles=("Giá & Bollinger Bands", "RSI (14)", "ADX & DI"))

        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Giá"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=1, dash='dash'), name="Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=1, dash='dash'), name="Lower"), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
        fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
        
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='black', width=2), name="ADX"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], line=dict(color='green', width=1), name="+DI"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], line=dict(color='red', width=1), name="-DI"), row=3, col=1)
        fig.add_hline(y=25, line_dash="dot", row=3, col=1, line_color="gray")
        fig.add_hline(y=50, line_dash="dot", row=3, col=1, line_color="red")

        fig.update_layout(height=800, title=f"Biểu đồ kỹ thuật: {ticker}", xaxis_rangeslider_visible=False)
        fig.show() # Lệnh này sẽ bật cửa sổ trình duyệt hiển thị biểu đồ

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()
