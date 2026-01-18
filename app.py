# app.py
# Engineer Alpha — 小白友好版：Signal / 风险 / 买点 / 加仓位（v1.2 + Crypto）
# 仅供研究与教育用途，不构成投资建议

from streamlit_autorefresh import st_autorefresh
import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Engineer Alpha 风险&买点工具", layout="centered")
st.title("Engineer Alpha 风险等级 & 买点区间（V1.2 + Crypto）")
st.caption("输入股票代码，自动给出：建议动作（观察/试探/建仓/加仓）+ 风险等级 + 分批买点区间 + 加仓位置 + 仓位建议。并新增 BTC/ETH 追踪面板。")

# -----------------------------
# Utils (format)
# -----------------------------
def money(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"${float(x):,.2f}"


def pct(x, nd=0):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{float(x) * 100:.{nd}f}%"


def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# -----------------------------
# Indicators
# -----------------------------
def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def pct_rank_window(close: pd.Series, window: int) -> float:
    if len(close) < window:
        return float("nan")
    w = close.iloc[-window:]
    return w.rank(pct=True).iloc[-1].item()


def ma(series: pd.Series, n: int) -> float:
    if len(series) < n:
        return float("nan")
    return float(series.rolling(n).mean().iloc[-1])


def annualized_vol(close: pd.Series) -> float:
    rets = close.pct_change().dropna()
    if len(rets) < 50:
        return float("nan")
    return float(rets.std() * math.sqrt(252))


def drawdown_1y(close: pd.Series) -> float:
    w = close.tail(252)
    if len(w) < 50:
        return float("nan")
    peak = w.max()
    last = w.iloc[-1]
    return float((last - peak) / peak)  # negative


def atr(df: pd.DataFrame, n: int = 14) -> float:
    """
    ATR(14). 若 High/Low 缺失，则自动降级为 close 波动近似，避免崩溃/NaN。
    """
    if not all(c in df.columns for c in ["High", "Low", "Close"]):
        close = df["Close"].dropna().astype(float) if "Close" in df.columns else None
        if close is None or len(close) < max(50, n + 5):
            return float("nan")
        rets = close.pct_change().dropna()
        v = rets.rolling(n).std().iloc[-1]
        return float(v * close.iloc[-1]) if pd.notna(v) else float("nan")

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    v = tr.rolling(n).mean().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


# -----------------------------
# Core: Signal / Risk / Zones
# -----------------------------
def signal_abc(df: pd.DataFrame) -> dict:
    close = df["Close"].dropna().astype(float)
    rsi = rsi_wilder(close, 14)

    last = float(close.iloc[-1])
    rsi_last = float(rsi.iloc[-1])

    pr_3y = pct_rank_window(close, 756)   # ~3y
    pr_5y = pct_rank_window(close, 1260)  # ~5y

    # A：位置偏低（分位低）
    A = (pd.notna(pr_3y) and pr_3y < 0.30) or (pd.notna(pr_5y) and pr_5y < 0.30)

    # B：情绪偏冷（RSI低）
    B = (rsi_last < 35)

    # C：回暖确认（更稳）
    rsi_dropna = rsi.dropna()
    C = False
    if len(rsi_dropna) >= 7:
        r1 = float(rsi_dropna.iloc[-1])
        r2 = float(rsi_dropna.iloc[-2])
        r3 = float(rsi_dropna.iloc[-3])

        rsi5 = float(rsi_dropna.rolling(5).mean().iloc[-1])
        rsi5_prev = float(rsi_dropna.rolling(5).mean().iloc[-2])

        two_day_up = (r1 > r2) and (r2 > r3)
        cross_up = (r2 <= rsi5_prev) and (r1 > rsi5)

        C = two_day_up or cross_up
    elif len(rsi_dropna) >= 3:
        r1 = float(rsi_dropna.iloc[-1])
        r2 = float(rsi_dropna.iloc[-2])
        r3 = float(rsi_dropna.iloc[-3])
        C = (r1 > r2) and (r2 > r3)

    if A and B and C:
        sig = "加仓"
    elif A and B:
        sig = "建仓"
    elif A or B:
        sig = "试探"
    else:
        sig = "观察"

    return {
        "Signal": sig,
        "Last": last,
        "RSI": rsi_last,
        "Pct3Y": pr_3y,
        "Pct5Y": pr_5y,
        "A_pos": A,
        "B_rsi": B,
        "C_turn": C
    }


def risk_level(df: pd.DataFrame) -> dict:
    close = df["Close"].dropna().astype(float)

    last = float(close.iloc[-1])
    ma50 = ma(close, 50)
    ma200 = ma(close, 200)
    trend_up = (pd.notna(ma50) and pd.notna(ma200) and ma50 > ma200)

    vol = annualized_vol(close)
    dd = drawdown_1y(close)

    score = 0

    if pd.notna(vol):
        if vol > 0.60:
            score += 3
        elif vol > 0.45:
            score += 2
        elif vol > 0.30:
            score += 1

    if pd.notna(dd):
        if dd < -0.40:
            score += 3
        elif dd < -0.30:
            score += 2
        elif dd < -0.15:
            score += 1

    if not trend_up:
        score += 1

    if score >= 5:
        lvl = "🔴 高风险"
        max_pos = 0.08
        add_step = 0.02
    elif score >= 3:
        lvl = "🟡 中等风险"
        max_pos = 0.15
        add_step = 0.04
    else:
        lvl = "🟢 低风险"
        max_pos = 0.25
        add_step = 0.06

    return {
        "Risk": lvl,
        "RiskScore": score,
        "TrendUp": trend_up,
        "MA50": safe_float(ma50),
        "MA200": safe_float(ma200),
        "Vol": safe_float(vol),
        "DD1Y": safe_float(dd),
        "Last": last,
        "MaxPosition": max_pos,
        "AddStep": add_step
    }


def buy_zones(df: pd.DataFrame) -> dict:
    close = df["Close"].dropna().astype(float)
    last = float(close.iloc[-1])

    a = atr(df, 14)
    a = safe_float(a)

    atr_pct = (a / last) if (a is not None and last > 0) else 0.0
    width = max((1.8 * a) if a is not None else 0.0, last * max(0.06, 1.2 * atr_pct))

    ma200 = ma(close, 200)
    dev200 = ((last - ma200) / ma200) if pd.notna(ma200) else 0.0
    center_disc = 0.10 + float(np.clip(dev200, -0.2, 0.2)) * 0.10
    center_disc = float(np.clip(center_disc, 0.06, 0.18))
    center = last * (1 - center_disc)

    conservative = (center + 0.6 * width, center + 1.2 * width)
    neutral = (center - 0.4 * width, center + 0.4 * width)
    aggressive = (center - 1.2 * width, center - 0.6 * width)

    def clamp(r):
        lo, hi = r
        lo = max(float(lo), 0.01)
        hi = max(float(hi), 0.01)
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    return {
        "ATR14": a,
        "Last": last,
        "Conservative": clamp(conservative),
        "Neutral": clamp(neutral),
        "Aggressive": clamp(aggressive)
    }


# -----------------------------
# Fundamentals (best-effort)
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares = info.get("sharesOutstanding")
    mktcap = info.get("marketCap")
    pe = info.get("trailingPE")
    ps = info.get("priceToSalesTrailing12Months")
    pb = info.get("priceToBook")

    revenue_ttm = info.get("totalRevenue")
    fcf = None

    try:
        cf = tk.cashflow
        if cf is not None and not cf.empty:
            col = cf.columns[0]
            ocf = cf.loc["Total Cash From Operating Activities", col] if "Total Cash From Operating Activities" in cf.index else None
            capex = cf.loc["Capital Expenditures", col] if "Capital Expenditures" in cf.index else None
            if ocf is not None and capex is not None and pd.notna(ocf) and pd.notna(capex):
                fcf = float(ocf) - float(capex)
    except Exception:
        pass

    return {
        "Price": safe_float(price),
        "Shares": safe_float(shares),
        "MarketCap": safe_float(mktcap),
        "RevenueTTM": safe_float(revenue_ttm),
        "FCF": safe_float(fcf),
        "PE": safe_float(pe),
        "PS": safe_float(ps),
        "PB": safe_float(pb),
    }


def rough_fair_value_range(f: dict) -> dict:
    price = f.get("Price")
    shares = f.get("Shares")
    mktcap = f.get("MarketCap")
    revenue = f.get("RevenueTTM")
    fcf = f.get("FCF")

    if price is None:
        return {"Method": "N/A", "FairLow": None, "FairMid": None, "FairHigh": None}

    if fcf is not None and mktcap is not None and shares is not None and fcf > 0 and shares > 0:
        low_mc = fcf / 0.06
        mid_mc = fcf / 0.045
        high_mc = fcf / 0.03
        return {
            "Method": "FCF Yield（粗算）",
            "FairLow": low_mc / shares,
            "FairMid": mid_mc / shares,
            "FairHigh": high_mc / shares
        }

    if revenue is not None and shares is not None and revenue > 0 and shares > 0:
        return {
            "Method": "PS Multiple（粗算）",
            "FairLow": (revenue * 4.0) / shares,
            "FairMid": (revenue * 6.0) / shares,
            "FairHigh": (revenue * 8.0) / shares
        }

    return {"Method": "N/A", "FairLow": None, "FairMid": None, "FairHigh": None}


def add_levels(last: float, zones: dict, fair: dict) -> dict:
    n_lo, n_hi = zones["Neutral"]
    a_lo, a_hi = zones["Aggressive"]

    first_add = n_lo
    pullback_add = (a_lo + a_hi) / 2

    fair_low = safe_float(fair.get("FairLow"))
    fair_mid = safe_float(fair.get("FairMid"))

    value_pocket = None
    rule = None
    if fair_low is not None and fair_low > 0:
        value_pocket = fair_low * 0.90
        rule = "价格 ≤ 0.9 × FairLow（估值折扣）"
    elif fair_mid is not None and fair_mid > 0:
        value_pocket = fair_mid * 0.70
        rule = "价格 ≤ 0.7 × FairMid（估值折扣）"

    return {
        "FirstAdd": first_add,
        "PullbackAdd": pullback_add,
        "ValuePocketAdd": value_pocket,
        "ValuePocketRule": rule
    }


# -----------------------------
# Human-friendly explanation
# -----------------------------
def badge_signal(sig: str) -> str:
    mapping = {"观察": "⚪ 观察", "试探": "🟡 试探", "建仓": "🟢 建仓", "加仓": "🔵 加仓"}
    return mapping.get(sig, sig)


def explain_abc(sig_dict: dict):
    A = sig_dict["A_pos"]
    B = sig_dict["B_rsi"]
    C = sig_dict["C_turn"]
    pr3 = sig_dict["Pct3Y"]
    pr5 = sig_dict["Pct5Y"]
    rsi = sig_dict["RSI"]

    lines = []
    lines.append(("A 位置偏低", A,
                  f"近3年分位：{pct(pr3, 0) if pr3==pr3 else '—'}；近5年分位：{pct(pr5, 0) if pr5==pr5 else '—'}（分位越低=越接近历史低位）"))
    lines.append(("B 情绪偏冷（RSI偏低）", B,
                  f"RSI(14)：{rsi:.1f}（<35 通常代表偏冷/超卖区附近）"))
    lines.append(("C 有回暖迹象（更稳确认）", C,
                  "RSI 连续上升或 RSI 上穿自身短均线，代表下跌动能减弱（不等于一定反转）"))
    return lines


# -----------------------------
# Data loader (cache)
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_price(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    df = normalize_columns(df)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_crypto(ticker: str, period: str = "730d", interval: str = "1d") -> pd.DataFrame:
    """
    Crypto 专用：用 period+interval，更适合追钟。
    注意：Yahoo 对某些 interval 有历史长度限制；失败就自动降级到 1d。
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        df = normalize_columns(df)
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("empty")
        return df
    except Exception:
        df = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=False)
        df = normalize_columns(df)
        return df


def weekly_from_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
    d = df_daily.copy()
    d = d.dropna(subset=["Close"])
    # 周线：收盘取最后，最高最低做 max/min
    wk = pd.DataFrame()
    wk["Close"] = d["Close"].resample("W").last()
    if "High" in d.columns and "Low" in d.columns:
        wk["High"] = d["High"].resample("W").max()
        wk["Low"] = d["Low"].resample("W").min()
    return wk.dropna()


def crypto_dashboard(symbol: str) -> dict:
    """
    输出：价格、RSI(1D/1W)、MA200D、MA200W、趋势判断、指导建议
    """
    df1d = load_crypto(symbol, period="5y", interval="1d")
    close1d = df1d["Close"].dropna().astype(float)

    last = float(close1d.iloc[-1])
    rsi1d = float(rsi_wilder(close1d, 14).iloc[-1])

    ma200d = ma(close1d, 200)

    # 周线
    df1d_idx = df1d.copy()
    if df1d_idx.index.tz is None:
        # yfinance 有时返回 naive index；不强制 tz，只要能 resample 就行
        pass
    wk = weekly_from_daily(df1d_idx)
    close1w = wk["Close"].dropna().astype(float)

    rsi1w = float(rsi_wilder(close1w, 14).iloc[-1]) if len(close1w) >= 20 else float("nan")
    ma200w = ma(close1w, 200) if len(close1w) >= 220 else float("nan")

    # 指导建议（人话 + 可执行，不装“预测”）
    advice = []
    # 大趋势：200W 很关键（类似 BTC 大周期牛熊）
    if pd.notna(ma200w):
        if last >= ma200w:
            advice.append("✅ 大周期：价格在 200W 均线之上，长期结构偏强（更像牛市/复苏期）。")
        else:
            advice.append("⚠️ 大周期：价格在 200W 均线之下，优先防守，别急着重仓抄底。")
    else:
        advice.append("ℹ️ 200W 数据不足，先用周线 RSI + 200D 作为替代判断。")

    # 中期：200D
    if pd.notna(ma200d):
        if last >= ma200d:
            advice.append("✅ 中期：在 200D 之上，回调更像‘分批机会’，但也要看周线动能。")
        else:
            advice.append("⚠️ 中期：在 200D 之下，反弹可能是‘解套盘’，更适合小仓位试探。")

    # 动能：RSI 组合
    # 1W > 50 强，<45 弱
    if pd.notna(rsi1w):
        if rsi1w >= 55:
            advice.append("🔥 周线动能偏强（RSI>55）：更适合‘逢回调分批’，不建议追涨梭哈。")
        elif rsi1w <= 45:
            advice.append("🧊 周线动能偏弱（RSI<45）：‘耐心等更低/更明确回暖’，分批慢一点。")
        else:
            advice.append("🟨 周线动能中性（RSI 45–55）：容易震荡洗盘，用分批纪律，不用情绪。")

    # 日线：超卖/超买提示（但提醒 BTC 会粘住强趋势）
    if rsi1d <= 30:
        advice.append("🎯 日线超卖（RSI<30）：可考虑‘试探/小加’，但最好等日线 RSI 回升再加速。")
    elif rsi1d >= 70:
        advice.append("🥵 日线偏热（RSI>70）：强趋势可能继续，但新仓别追太急，等回踩更舒服。")
    else:
        advice.append("📌 日线 RSI 中性：更看‘趋势线 + 均线位置’，按计划分批。")

    # 简单仓位模板（给小白“怎么做”）
    # 注意：这里只给模板，不给交易建议
    playbook = []
    playbook.append("仓位模板（示例）：")
    playbook.append("1) 周线强 + 价格在 200D/200W 上：最大仓位 20–35%，回调分 3–5 笔。")
    playbook.append("2) 周线弱 或 跌破 200W：最大仓位 5–15%，只做试探，不做重仓。")
    playbook.append("3) RSI<30 也不要一把梭：用‘每次 2–5%’慢慢加。")

    return {
        "Symbol": symbol,
        "Last": last,
        "RSI_1D": rsi1d,
        "RSI_1W": rsi1w,
        "MA200D": safe_float(ma200d),
        "MA200W": safe_float(ma200w),
        "Advice": advice,
        "Playbook": playbook
    }


# -----------------------------
# UI: Tabs
# -----------------------------
tab_stock, tab_crypto = st.tabs(["📈 股票/ETF 分析", "₿ BTC/ETH 追踪（追钟）"])

# ================
# Tab 1: Stock
# ================
with tab_stock:
    colA, colB = st.columns([2, 1])
    with colA:
        ticker = st.text_input("输入股票 Ticker（例如 MSFT / COIN / MSTR / RKLB）", value="MSFT").strip().upper()
    with colB:
        mode = st.selectbox("你的风格", ["标准（推荐）", "保守", "激进"], index=0)

    years = st.slider("历史回看长度（年）", 2, 15, 10)
    run = st.button("生成分析（股票/ETF）")

    if run:
        if not ticker:
            st.error("Ticker 不能为空。")
            st.stop()

        start = (pd.Timestamp.today(tz="UTC") - pd.Timedelta(days=365 * years)).date().isoformat()

        with st.spinner("拉取价格数据…"):
            df = load_price(ticker, start=start)

        if df is None or df.empty or "Close" not in df.columns or len(df) < 260:
            st.error("数据不足或拉取失败（可能 ticker 错误、停牌、或历史太短）。")
            st.stop()

        sig = signal_abc(df)
        risk = risk_level(df)
        zones = buy_zones(df)

        with st.spinner("拉取基本面（若缺失会自动降级）…"):
            f = get_fundamentals(ticker)
            fair = rough_fair_value_range(f)

        adds = add_levels(sig["Last"], zones, fair)

        st.subheader(f"{ticker} — 结果")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("建议动作", badge_signal(sig["Signal"]))
        with m2:
            st.metric("风险等级", risk["Risk"])
        with m3:
            st.metric("当前价格", money(sig["Last"]))

        one_liner = []
        one_liner.append("位置偏低" if sig["A_pos"] else "位置不低")
        one_liner.append("RSI偏冷" if sig["B_rsi"] else "RSI不冷")
        one_liner.append("开始回暖" if sig["C_turn"] else "未回暖")
        st.info("一句话： " + " ｜ ".join(one_liner) + "（用于分批决策，不预测涨跌）")

        st.divider()

        st.subheader("仓位建议（按风险自动给）")
        p1, p2 = st.columns(2)
        with p1:
            st.metric("建议最大仓位上限", pct(risk["MaxPosition"], 0))
        with p2:
            st.metric("建议单次加仓步长", pct(risk["AddStep"], 0))
        st.caption("提示：这是“仓位纪律模板”，不是买卖信号。高风险票就算看好，也建议把单票仓位压小、分批慢慢来。")

        st.divider()

        st.subheader("买入区间（分批，不猜底）")
        cons = zones["Conservative"]
        neut = zones["Neutral"]
        aggr = zones["Aggressive"]

        if mode.startswith("保守"):
            rec = cons
            rec_name = "保守"
        elif mode.startswith("激进"):
            rec = aggr
            rec_name = "激进"
        else:
            rec = neut
            rec_name = "标准"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🟦 保守")
            st.caption("更稳：等回调到更舒服的位置")
            st.write(f"{money(cons[0])}  ~  {money(cons[1])}")
        with c2:
            st.markdown("### 🟩 标准")
            st.caption("主力区：适合分批建仓")
            st.write(f"{money(neut[0])}  ~  {money(neut[1])}")
        with c3:
            st.markdown("### 🟥 激进")
            st.caption("抄底带：波动大，适合敢分批抄底")
            st.write(f"{money(aggr[0])}  ~  {money(aggr[1])}")

        st.success(f"你选择的是 **{mode}** → 推荐从 **{rec_name}区间** 开始分批：{money(rec[0])} ~ {money(rec[1])}")
        st.caption("说明：区间基于 ATR（或波动降级近似）+ 均值偏离生成，是“分批带”，不是预测底部。")

        st.divider()

        st.subheader("加仓位置（更像操作手册）")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("第一加仓（标准区下沿）", money(adds["FirstAdd"]))
        with a2:
            st.metric("回调加仓（抄底带中点）", money(adds["PullbackAdd"]))
        with a3:
            st.metric("价值洼地加仓", money(adds["ValuePocketAdd"]))

        if adds["ValuePocketRule"]:
            st.caption(f"价值洼地规则：{adds['ValuePocketRule']}（基本面字段缺失时可能不显示）")

        st.divider()

        st.subheader("为什么会给这个建议？（人话解释）")
        for title, ok, desc in explain_abc(sig):
            st.write(("✅ " if ok else "❌ ") + title)
            st.caption(desc)

        with st.expander("高级数据（给懂的人看）"):
            st.write({
                "MA50": money(risk["MA50"]),
                "MA200": money(risk["MA200"]),
                "趋势(MA50>MA200)": risk["TrendUp"],
                "年化波动率": pct(risk["Vol"], 0),
                "近1年回撤(从高点到现在)": pct(risk["DD1Y"], 1),
                "ATR(14)": money(zones["ATR14"]),
                "估值方法": fair.get("Method"),
                "FairLow": money(fair.get("FairLow")),
                "FairMid": money(fair.get("FairMid")),
                "FairHigh": money(fair.get("FairHigh")),
                "PE": f.get("PE"),
                "PS": f.get("PS"),
                "PB": f.get("PB"),
                "建议最大仓位上限": pct(risk["MaxPosition"], 0),
                "建议单次加仓步长": pct(risk["AddStep"], 0),
            })

        st.warning("免责声明：本工具仅用于研究与教育，不构成投资建议。市场有风险，投资需谨慎。")


# ================
# Tab 2: Crypto (追钟)
# ================
# ================
# Tab 2: Crypto (手动刷新)
# ================
with tab_crypto:
    st.subheader("BTC/ETH 追踪面板（手动刷新）")
    st.caption("这里是‘盯趋势 + 看动能 + 给纪律模板’，不做预测，不给买卖建议。")

    c1, c2 = st.columns([1, 2])
    with c1:
        crypto = st.selectbox("选择币种", ["BTC-USD", "ETH-USD"], index=0)
    with c2:
        st.caption("不自动刷新；需要更新时点一下按钮即可。")
        if st.button("手动刷新 Crypto 数据"):
            st.cache_data.clear()  # 强制清缓存，确保拿到最新数据

    with st.spinner("拉取 Crypto 数据…"):
        dash = crypto_dashboard(crypto)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("币种", dash["Symbol"])
    with m2:
        st.metric("当前价格", money(dash["Last"]))
    with m3:
        st.metric("RSI(14) 日线", f"{dash['RSI_1D']:.1f}")
    with m4:
        st.metric("RSI(14) 周线", f"{dash['RSI_1W']:.1f}" if dash["RSI_1W"] == dash["RSI_1W"] else "—")

    t1, t2 = st.columns(2)
    with t1:
        st.write("**趋势锚点**")
        st.write(f"200D MA：{money(dash['MA200D'])}")
        st.write(f"200W MA：{money(dash['MA200W'])}")
    with t2:
        st.write("**指导建议（人话）**")
        for s in dash["Advice"]:
            st.write("- " + s)

    st.divider()
    st.write("**执行模板（给小白用）**")
    for s in dash["Playbook"]:
        st.write("- " + s)

    st.warning("免责声明：本工具仅用于研究与教育，不构成投资建议。加密资产波动极大，请控制仓位与杠杆风险。")

