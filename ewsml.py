import os
import warnings
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Risk Management ML + Gemini News", layout="wide")

st.title("ML BASED EARLY WARNING SYSTEM FOR RISK MANAGEMENT")
st.markdown("Logistic Regression with **TIME-BASED SPLIT** plus Gemini-API news sentiment for selected stocks.")

def get_gemini_sentiment(company_name, api_key):
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    search_query = f"Find the latest news and financial headlines for the company '{company_name}'."
    search_system_prompt = (
        "You are a financial news aggregator. Your task is to find the latest, most relevant news "
        "headlines for the given company using the Google Search tool and present them as a concise summary."
    )
    search_payload = {
        "contents": [{"parts": [{"text": search_query}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {
            "parts": [{"text": search_system_prompt}]
        }
    }
    
    try:
        search_response = requests.post(api_url, json=search_payload, timeout=120)
        if search_response.status_code != 200:
            return {"error": f"API Search Error: {search_response.status_code} - {search_response.text}"}
        search_result = search_response.json()
        if 'candidates' not in search_result or not search_result['candidates']:
            return {"error": "Invalid search response from API. No candidates."}
        news_context = search_result['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        return {"error": f"Network Error during search: {e}"}
    except Exception as e:
        return {"error": f"An unknown error occurred during search: {e}"}
    
    analyze_query = (
        f"Here is the latest news context for '{company_name}':\n\n{news_context}\n\n"
        f"Based *only* on this provided news context, please provide a financial risk analysis."
    )
    analyze_system_prompt = (
        "You are a world-class financial analyst. Your task is to analyze the provided news summary "
        "and provide a brief summary and a risk assessment. "
        "You must output your analysis in a structured JSON format."
    )
    json_schema = {
        "type": "OBJECT",
        "properties": {
            "risk_score": {
                "type": "NUMBER",
                "description": "A score from 0 (very low risk) to 10 (very high risk) based on the news sentiment."
            },
            "sentiment": {
                "type": "STRING",
                "description": "The overall sentiment of the news ('Positive', 'Negative', or 'Neutral')."
            },
            "summary": {
                "type": "STRING",
                "description": "A 2-3 sentence summary explaining the news and the reason for the given risk score."
            }
        },
        "required": ["risk_score", "sentiment", "summary"]
    }
    analyze_payload = {
        "contents": [{"parts": [{"text": analyze_query}]}],
        "systemInstruction": {
            "parts": [{"text": analyze_system_prompt}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": json_schema
        }
    }
    
    try:
        analyze_response = requests.post(api_url, json=analyze_payload, timeout=120)
        if analyze_response.status_code == 200:
            analyze_result = analyze_response.json()
            if 'candidates' not in analyze_result or not analyze_result['candidates']:
                return {"error": "Invalid analysis response from API. No candidates."}
            text_part = analyze_result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(text_part)
        else:
            return {"error": f"API Analysis Error: {analyze_response.status_code} - {analyze_response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network Error during analysis: {e}"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON response: {e}"}
    except Exception as e:
        return {"error": f"An unknown error occurred during analysis: {e}"}

st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

st.sidebar.header("📰 Gemini News Analysis")
enable_news = st.sidebar.checkbox("Enable news‑based sentiment with Gemini", value=True)
company_name = st.sidebar.text_input("Company name", value="Apple Inc.")
symbol = st.sidebar.text_input("Stock symbol (for context only)", value="AAPL")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get your key from https://aistudio.google.com/app/apikey")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded!")
        
        st.header("1️⃣ Data Cleaning")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Original:** {df.shape}")
            st.write("**Missing:**")
            st.write(df.isnull().sum())

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df = df.dropna()

        with col2:
            st.write(f"**Cleaned:** {df.shape}")
            st.write(f"**Period:** {df['Date'].min().date()} → {df['Date'].max().date()}")
            
        st.header("2️⃣ Risk Metrics (No Future Leakage)")

        df["Daily_Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Daily_Return"].expanding().std()
        df["Volatility_Annual"] = df["Volatility"] * np.sqrt(252)

        df["VaR_95"] = df["Daily_Return"].expanding().quantile(0.05)
        df["VaR_95_Annual"] = df["VaR_95"] * np.sqrt(252)

        rf_rate = 0.05 / 252
        df["Sharpe_Ratio"] = (
            (df["Daily_Return"] - rf_rate).expanding().mean()
            / df["Daily_Return"].expanding().std()
        )

        df["Momentum_5"] = df["Close"].pct_change(5)
        df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]

        df = df.dropna().reset_index(drop=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Volatility", f"{df['Volatility_Annual'].mean():.1%}")
        with col2:
            st.metric("Sharpe", f"{df['Sharpe_Ratio'].mean():.2f}")
        with col3:
            st.metric("VaR 95%", f"{df['VaR_95_Annual'].mean():.1%}")
            
        st.header("3️⃣ Risk Labels")

        def classify_risk(row):
            vol = row["Volatility_Annual"]
            var = row["VaR_95_Annual"]
            sharpe = row["Sharpe_Ratio"]
            score = 0
            if vol > df["Volatility_Annual"].quantile(0.75):
                score += 2
            elif vol > df["Volatility_Annual"].quantile(0.25):
                score += 1
            if var < -0.02:
                score += 2
            elif var < -0.01:
                score += 1
            if sharpe < 0:
                score += 1
            return 2 if score >= 4 else 1 if score >= 2 else 0

        df["Risk_Level"] = df.apply(classify_risk, axis=1)

        risk_dist = df["Risk_Level"].value_counts().sort_index()
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Low", "Medium", "High"],
                    y=risk_dist.values,
                    marker_color=["green", "orange", "red"],
                )
            ]
        )
        fig.update_layout(title="Risk Distribution", height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.header("4️⃣ Logistic Regression (Time-Safe)")

        feature_cols = [
            "Volatility_Annual",
            "VaR_95_Annual",
            "Sharpe_Ratio",
            "Momentum_5",
            "HL_Range",
        ]
        X = df[feature_cols].copy()
        y = df["Risk_Level"].copy()
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        st.write(f"**Train:** {len(X_train)} days ({X_train.index[0]} → {X_train.index[-1]})")
        st.write(f"**Test:** {len(X_test)} days ({X_test.index[0]} → {X_test.index[-1]})")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # ✅ FIXED: Use correct parameters for multi-class LogisticRegression
        log_reg = LogisticRegression(
            solver="lbfgs",
            C=0.01,  # strong L2 regularization
            penalty="l2",
            max_iter=2000,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

        log_reg.fit(X_train_scaled, y_train)
        y_pred = log_reg.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        st.subheader("Realistic Performance (Time-Safe Split)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("Precision", f"{precision:.3f}")
        col3.metric("Recall", f"{recall:.3f}")
        col4.metric("F1-Score", f"{f1:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Low", "Medium", "High"],
                y=["Low", "Medium", "High"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )
        fig.update_layout(title="Confusion Matrix (Test Set)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.header("5️⃣ Risk Predictions")
        X_full_scaled = scaler.transform(X)
        full_pred = log_reg.predict(X_full_scaled)
        df["Predicted_Risk"] = full_pred
        df["Risk_Label"] = df["Predicted_Risk"].map(
            {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        )

        st.dataframe(df[["Date", "Close", "Risk_Label"]].tail(20), use_container_width=True)

        st.header("6️⃣ Risk Graphs")
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(
                go.Scatter(x=df["Date"], y=df["Volatility_Annual"], line=dict(color="blue"))
            )
            fig.update_layout(title="Volatility", height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(
                go.Scatter(x=df["Date"], y=df["VaR_95_Annual"], line=dict(color="red"))
            )
            fig.update_layout(title="VaR 95%", height=350)
            st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure(
            go.Scatter(x=df["Date"], y=df["Sharpe_Ratio"], line=dict(color="green"))
        )
        fig.update_layout(title="Sharpe Ratio", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.header("7️⃣ Download")
        output_df = df[
            [
                "Date",
                "Close",
                "Volatility_Annual",
                "Sharpe_Ratio",
                "VaR_95_Annual",
                "Risk_Label",
            ]
        ].copy()
        csv = output_df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "risk_predictions.csv", "text/csv")

        st.success(f"✅ Done! Realistic accuracy: {accuracy:.1%}")
        
        st.header("8️⃣ News-Based Qualitative Risk (Gemini)")

        if enable_news:
            st.subheader("News & Sentiment Risk Analysis (Gemini)")
            if company_name and gemini_api_key:
                if st.button(f"Analyze News & Sentiment for {company_name}"):
                    with st.spinner(f"Gemini is analyzing the latest news for {company_name}..."):
                        sentiment_data = get_gemini_sentiment(company_name, gemini_api_key)
                        if "error" in sentiment_data:
                            st.error(f"Error from Gemini API: {sentiment_data['error']}")
                        else:
                            st.subheader(f"Gemini Risk Analysis for {company_name}")
                            score = sentiment_data.get('risk_score', 0)
                            st.metric("Gemini Risk Score (0-10)", f"{score}/10")

                            sentiment = sentiment_data.get('sentiment', 'N/A')
                            if sentiment == "Positive":
                                st.success(f"Sentiment: **{sentiment}**")
                            elif sentiment == "Negative":
                                st.error(f"Sentiment: **{sentiment}**")
                            else:
                                st.info(f"Sentiment: **{sentiment}**")

                            summary = sentiment_data.get('summary', 'No summary available')
                            st.markdown(f"**Summary:** {summary}")
            else:
                st.info("Enter a Company Name and your Gemini API Key above to enable real-time news analysis.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.exception(e)
else:
    st.info("👆 Upload a CSV file with columns: Date, Open, High, Low, Close, Volume")
    st.info("Enter your Gemini API Key to enable real-time news analysis.")
