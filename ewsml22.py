import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stock Risk Early-Warning System", layout="wide")

RNG = 42
TRADING_DAYS = 252

# ------------------- DATA LOADING & CLEANING -------------------
@st.cache_data(show_spinner=False)
def load_and_clean_data(file):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    synonyms = {
        "date": "Date", "timestamp": "Date", "time": "Date",
        "open": "Open", "opening_price": "Open",
        "high": "High", "low": "Low",
        "close": "Close", "adj close": "Close", "closing_price": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=synonyms)
    required = {'Date', 'Open', 'High', 'Low', 'Close'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Missing columns. Need: {required}")

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    return df

# ------------------- QUANTITATIVE RISK METRICS ENGINE -------------------
def calculate_annualized_metrics(returns_series, rf_annual=0.04, var_conf=0.95):
    """
    Computes the three core annualized quantitative risk metrics:
    1. Annualized Volatility (std * sqrt(252))
    2. Annualized Value at Risk (Historical percentile scaled by sqrt(252))
    3. Annualized Sharpe Ratio ((mean * 252 - Rf) / Ann_Vol)
    """
    clean_returns = returns_series.dropna()
    if len(clean_returns) < 5:
        return 0.0, 0.0, 0.0

    # 1. Annualized Volatility
    daily_vol = clean_returns.std()
    ann_vol = daily_vol * np.sqrt(TRADING_DAYS)

    # 2. Annualized Historical VaR (at given confidence level)
    daily_var = -np.percentile(clean_returns, (1.0 - var_conf) * 100)
    ann_var = daily_var * np.sqrt(TRADING_DAYS)

    # 3. Annualized Sharpe Ratio
    ann_return = clean_returns.mean() * TRADING_DAYS
    excess_return = ann_return - rf_annual
    ann_sharpe = excess_return / ann_vol if ann_vol > 0 else 0.0

    return ann_vol, ann_var, ann_sharpe

# ------------------- ENHANCED FEATURE ENGINEERING -------------------
@st.cache_data(show_spinner=False)
def engineer_features(df, vol_window):
    data = df.copy()
    data["Daily_Return"] = data["Close"].pct_change()

    # Rolling annualised volatility
    roll = data["Daily_Return"].rolling(vol_window, min_periods=vol_window)
    data["Vol_Rolling_Ann"] = roll.std() * np.sqrt(TRADING_DAYS)

    # Momentum
    data["Momentum_5"] = data["Close"].pct_change(5)
    data["Momentum_20"] = data["Close"].pct_change(20)

    # ATR (Lagged strictly by 1 day to prevent same-day high/low leakage)
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift(1)).abs()
    low_close = (data["Low"] - data["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR_14_Lag1"] = (true_range.rolling(14).mean() / data["Close"]).shift(1)

    # RSI (Wilder's Exponential Smoothing)
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # Overnight gap
    data["Overnight_Gap"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)

    feature_cols = [
        "Vol_Rolling_Ann", "Momentum_5", "Momentum_20",
        "ATR_14_Lag1", "RSI_14", "Overnight_Gap"
    ]

    # Volume Regime Features
    if "Volume" in data.columns and data["Volume"].sum() > 0:
        data["Vol_Change"] = data["Volume"].pct_change()
        data["Vol_MA_Ratio"] = data["Volume"] / data["Volume"].rolling(20).mean()
        data["ATR_Vol_Ratio"] = data["ATR_14_Lag1"] / (data["Vol_MA_Ratio"] + 1e-5)
        feature_cols.extend(["Vol_Change", "Vol_MA_Ratio", "ATR_Vol_Ratio"])

    data = data.dropna(subset=feature_cols).reset_index(drop=True)
    return data, feature_cols

# ------------------- FORWARD LABELS -------------------
@st.cache_data(show_spinner=False)
def generate_forward_labels(df, horizon):
    data = df.copy()
    n = len(data)
    returns = data["Daily_Return"].values
    fwd_vol_ann = np.full(n, np.nan)

    for i in range(n - horizon):
        window = returns[i + 1 : i + 1 + horizon]
        fwd_vol_ann[i] = window.std() * np.sqrt(TRADING_DAYS)

    data["Fwd_Vol_Ann"] = fwd_vol_ann
    labeled_df = data.dropna(subset=["Fwd_Vol_Ann"]).reset_index(drop=True)
    live_df = data[data["Fwd_Vol_Ann"].isna()].reset_index(drop=True)
    return labeled_df, live_df

# ------------------- MAIN APP UI -------------------
st.title("📊 Institutional Stock Risk Early‑Warning System")
st.markdown(
    "**Isotonic Calibrated Random Forest** + **F2-Score Recall Optimization** + **Annualized Core Risk Metrics**."
)

# Sidebar Controls
uploaded_file = st.sidebar.file_uploader("Upload daily OHLC CSV", type=["csv"])
target_mode = st.sidebar.selectbox(
    "Target Regime Definition",
    ["Binary (Vol Expansion vs Contraction)", "3‑Class (Low / Normal / High Risk)"]
)
is_binary = target_mode.startswith("Binary")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Quantitative Risk Parameters")
st.sidebar.caption("Adjust Rf rate here to match your local Treasury yield:")
rf_rate = st.sidebar.number_input("Annualized Risk-Free Rate (%)", value=4.0, step=0.25) / 100.0
var_confidence = st.sidebar.selectbox("VaR Confidence Level", [0.90, 0.95, 0.99], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Model & Alert Controls")
alert_threshold = st.sidebar.slider("Manual Alert Probability Threshold", 5, 90, 25) / 100.0
horizon = st.sidebar.slider("Forward Horizon (days)", 5, 30, 14)
vol_window = st.sidebar.slider("Volatility Lookback (days)", 20, 120, 60)
test_frac = st.sidebar.slider("Test Set Fraction", 0.1, 0.4, 0.2)

if uploaded_file is not None:
    try:
        df = load_and_clean_data(uploaded_file)
        if len(df) < (vol_window + horizon + 50):
            st.error(f"Need at least {vol_window + horizon + 50} rows, got {len(df)}")
            st.stop()

        features_df, feature_cols = engineer_features(df, vol_window)
        labeled_df, live_df = generate_forward_labels(features_df, horizon)

        # --- Manual train/test split with Embargo ---
        test_size = int(len(labeled_df) * test_frac)
        test_start = len(labeled_df) - test_size
        train_end = max(0, test_start - horizon) 

        train_slice = labeled_df.iloc[:train_end].copy()
        test_slice = labeled_df.iloc[test_start:].copy()

        # --- Leakage-Free Label Assignment ---
        if is_binary:
            train_slice["Risk_Level"] = np.where(train_slice["Fwd_Vol_Ann"] > train_slice["Vol_Rolling_Ann"], 1, 0)
            test_slice["Risk_Level"] = np.where(test_slice["Fwd_Vol_Ann"] > test_slice["Vol_Rolling_Ann"], 1, 0)
            target_class = 1
            labels_display = ["🟢 Calm", "🔴 Vol Spike"]
        else:
            q33 = train_slice["Fwd_Vol_Ann"].quantile(0.33)
            q67 = train_slice["Fwd_Vol_Ann"].quantile(0.67)

            def assign_3class(val):
                if val <= q33: return 0      
                elif val <= q67: return 1    
                else: return 2               

            train_slice["Risk_Level"] = train_slice["Fwd_Vol_Ann"].apply(assign_3class)
            test_slice["Risk_Level"] = test_slice["Fwd_Vol_Ann"].apply(assign_3class)
            target_class = 2  
            labels_display = ["🟢 Low Vol", "🟡 Normal", "🔴 High Spike"]

        X_train, y_train = train_slice[feature_cols], train_slice["Risk_Level"]
        X_test, y_test = test_slice[feature_cols], test_slice["Risk_Level"]

        # --- Train Calibrated Random Forest with STRICT TimeSeriesSplit ---
        with st.spinner("Training & Calibrating Random Forest (Isotonic Regression)..."):
            base_clf = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,             
                min_samples_leaf=4,      
                max_features="sqrt",
                class_weight="balanced", 
                random_state=RNG
            )
            # FIXED: Isotonic calibration prevents tail probabilities from squashing toward zero
            tscv = TimeSeriesSplit(n_splits=3)
            clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=tscv)
            clf.fit(X_train, y_train)

        # --- LEAKAGE-FREE Threshold Discovery (Optimized for F2-Score / Recall) ---
        train_proba = clf.predict_proba(X_train)
        train_risk_probs = train_proba[:, 1] if is_binary else (train_proba[:, 2] if train_proba.shape[1] > 2 else train_proba[:, -1])
        binary_y_train = (y_train == target_class).astype(int)
        
        prec_train, rec_train, thresh_train = precision_recall_curve(binary_y_train, train_risk_probs)
        
        # FIXED: Use F2-Score (beta=2.0) to weight Recall 2x heavier than Precision
        beta = 2.0
        f2_train = (1 + beta**2) * (prec_train * rec_train) / ((beta**2 * prec_train) + rec_train + 1e-8)
        best_idx = np.argmax(f2_train)
        leakage_free_optimal_thresh = thresh_train[best_idx] if best_idx < len(thresh_train) else 0.22

        # --- Out-of-Sample Evaluation on X_test ---
        y_pred_default = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        
        if is_binary:
            probs_risk = y_proba[:, 1]
        else:
            probs_risk = y_proba[:, 2] if y_proba.shape[1] > 2 else y_proba[:, -1]

        # Evaluate at User Manual Threshold
        y_pred_thresh = np.where(probs_risk >= alert_threshold, target_class, 0)
        if not is_binary:
            y_pred_thresh = np.where(probs_risk >= alert_threshold, 2, np.argmax(y_proba[:, :2], axis=1))

        # Evaluate at Leakage-Free Optimal F2 Threshold
        y_pred_optimal = np.where(probs_risk >= leakage_free_optimal_thresh, target_class, 0)
        if not is_binary:
            y_pred_optimal = np.where(probs_risk >= leakage_free_optimal_thresh, 2, np.argmax(y_proba[:, :2], axis=1))

        binary_y_test = (y_test == target_class).astype(int)
        
        acc_default = accuracy_score(y_test, y_pred_default)
        acc_thresh = accuracy_score(y_test, y_pred_thresh)
        prec_manual = precision_score(binary_y_test, (y_pred_thresh == target_class).astype(int), zero_division=0)
        rec_manual = recall_score(binary_y_test, (y_pred_thresh == target_class).astype(int), zero_division=0)
        f2_manual = (1 + 4) * (prec_manual * rec_manual) / ((4 * prec_manual) + rec_manual + 1e-8)

        prec_opt = precision_score(binary_y_test, (y_pred_optimal == target_class).astype(int), zero_division=0)
        rec_opt = recall_score(binary_y_test, (y_pred_optimal == target_class).astype(int), zero_division=0)
        f2_opt = (1 + 4) * (prec_opt * rec_opt) / ((4 * prec_opt) + rec_opt + 1e-8)

        # --- DISPLAY SECTION 1: ANNUALIZED CORE QUANTITATIVE METRICS ---
        st.subheader("📈 Institutional Portfolio Risk Metrics (Annualized)")
        st.markdown(f"*Computed across the full historical dataset assuming **252 trading days** and an annualized Rf rate of **{rf_rate:.2%}**.*")
        
        full_returns = pd.concat([train_slice["Daily_Return"], test_slice["Daily_Return"]])
        ann_vol, ann_var, ann_sharpe = calculate_annualized_metrics(full_returns, rf_annual=rf_rate, var_conf=var_confidence)
        test_vol, test_var, test_sharpe = calculate_annualized_metrics(test_slice["Daily_Return"], rf_annual=rf_rate, var_conf=var_confidence)

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Annualized Volatility (σ)", 
            f"{ann_vol:.2%}", 
            delta=f"{test_vol - ann_vol:+.2%} in Test Period",
            delta_color="inverse"
        )
        m2.metric(
            f"Annualized {var_confidence:.0%} VaR", 
            f"{ann_var:.2%}", 
            delta=f"{test_var - ann_var:+.2%} in Test Period",
            delta_color="inverse",
            help="The estimated maximum annualized loss at the selected confidence level."
        )
        m3.metric(
            "Annualized Sharpe Ratio", 
            f"{ann_sharpe:.2f}", 
            delta=f"{test_sharpe - ann_sharpe:+.2f} in Test Period",
            help="Risk-adjusted excess return per unit of annualized volatility."
        )

        st.markdown("---")

        # --- DISPLAY SECTION 2: MACHINE LEARNING TEST PERFORMANCE ---
        st.subheader("🎯 Out-of-Sample ML Alert Performance")
        
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy (Default 50%)", f"{acc_default:.1%}")
        col2.metric("Accuracy (Manual Thresh)", f"{acc_thresh:.1%}")
        col3.metric("Precision (Manual Thresh)", f"{prec_manual:.1%}", delta=f"{prec_manual - prec_opt:+.1%} vs F2-Opt")
        col4.metric("Recall (Manual Thresh)", f"{rec_manual:.1%}", delta=f"{rec_manual - rec_opt:+.1%} vs F2-Opt")
        col5.metric("F2-Score (Recall Weighted)", f"{f2_manual:.2f}", delta=f"{f2_manual - f2_opt:+.2f} vs F2-Opt")

        # --- Precision-Recall Curve ---
        precisions_test, recalls_test, thresholds_test = precision_recall_curve(binary_y_test, probs_risk)
        f2_test_curve = (1 + 4) * (precisions_test * recalls_test) / ((4 * precisions_test) + recalls_test + 1e-8)

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=thresholds_test, y=precisions_test[:-1], mode='lines', name='Test Precision', line=dict(color='blue')))
        fig_pr.add_trace(go.Scatter(x=thresholds_test, y=recalls_test[:-1], mode='lines', name='Test Recall (Crash Catch Rate)', line=dict(color='orange', width=3)))
        fig_pr.add_trace(go.Scatter(x=thresholds_test, y=f2_test_curve[:-1], mode='lines', name='Test F2-Score (2x Recall Weight)', line=dict(color='green', width=2, dash='dash')))
        
        fig_pr.add_vline(x=alert_threshold, line_width=2, line_dash="solid", line_color="red", annotation_text=f"Manual Slider ({alert_threshold:.0%})")
        fig_pr.add_vline(x=leakage_free_optimal_thresh, line_width=2, line_dash="dot", line_color="green", annotation_text=f"F2-Optimal ({leakage_free_optimal_thresh:.0%})")
        
        fig_pr.update_layout(height=380, xaxis_title="Decision Probability Threshold", yaxis_title="Score (0.0 to 1.0)", hovermode="x unified")
        st.plotly_chart(fig_pr, use_container_width=True)

        # Confusion Matrix at Manual Threshold
        cm = confusion_matrix(y_test, y_pred_thresh)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=labels_display[:len(cm)], y=labels_display[:len(cm)],
            text=cm, texttemplate="%{text}",
            colorscale="Blues", showscale=False
        ))
        fig_cm.update_layout(
            title=f"Confusion Matrix (Active Manual Trigger @ {alert_threshold:.0%} Confidence)", 
            height=350, xaxis_title="Predicted Regime", yaxis_title="Actual Market Regime"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # --- DISPLAY SECTION 3: LIVE WARNINGS & ROLLING METRICS ---
        st.subheader(f"⚡ Latest {len(live_df)} Days: Live Warnings & Rolling Risk")
        if len(live_df) > 0:
            X_live = live_df[feature_cols]
            live_probs = clf.predict_proba(X_live)
            
            if is_binary:
                live_risk_probs = live_probs[:, 1]
                live_preds = np.where(live_risk_probs >= alert_threshold, 1, 0)
                labels_map = {0: "🟢 Calm", 1: "🔴 Vol Spike"}
            else:
                live_risk_probs = live_probs[:, 2] if live_probs.shape[1] > 2 else live_probs[:, -1]
                live_preds = np.where(live_risk_probs >= alert_threshold, 2, np.argmax(live_probs[:, :2], axis=1))
                labels_map = {0: "🟢 Low Vol", 1: "🟡 Normal", 2: "🔴 High Spike"}

            live_out = live_df[["Date", "Close", "Vol_Rolling_Ann"]].copy()
            live_out["Forecast Regime"] = pd.Series(live_preds).map(labels_map)
            live_out["Spike Probability"] = live_risk_probs
            
            # Compute rolling 20-day annualized Sharpe on live data for context
            live_returns = live_df["Daily_Return"]
            rolling_sharpe = ((live_returns.rolling(20).mean() * TRADING_DAYS - rf_rate) / 
                              (live_returns.rolling(20).std() * np.sqrt(TRADING_DAYS)))
            live_out["20D Ann. Sharpe"] = rolling_sharpe.fillna(0.0)

            st.dataframe(
                live_out.tail(10).style.format({
                    "Close": "${:.2f}", 
                    "Vol_Rolling_Ann": "{:.2%}",
                    "Spike Probability": "{:.1%}",
                    "20D Ann. Sharpe": "{:.2f}"
                }),
                use_container_width=True
            )
        else:
            st.info("No live rows available. Reduce forward horizon to see live unlabelled days.")

        # --- Download CSV ---
        st.subheader("📥 Download Comprehensive Backtest & Metrics")
        full_df = pd.concat([train_slice, test_slice]).sort_values("Date")
        export_df = full_df[["Date", "Close", "Daily_Return", "Vol_Rolling_Ann", "Risk_Level"]].copy()
        export_df["Predicted_Regime"] = clf.predict(full_df[feature_cols])
        
        st.download_button(
            label="Download Backtest CSV",
            data=export_df.to_csv(index=False),
            file_name=f"{uploaded_file.name.replace('.csv', '')}_institutional_risk_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Execution Error: {e}")
        st.exception(e)
else:
    st.info("👆 Upload a daily OHLC CSV file in the sidebar to run the system.")