import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from pathlib import Path

# Set up file path
DATA_FILENAME = Path(__file__).parent / "data" / "Measles 2025.csv"

# Streamlit app to forecast measles cases for 2025
st.set_page_config(page_title="📈 Measles Forecast 2025", layout="wide")
st.title("🦠 Measles - Weekly Forecast for 2025")

# Read the CSV file
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILENAME)

data = load_data()

# Session state to hold the edited data across reruns
if "data" not in st.session_state:
    st.session_state.data = None

required_cols = {'week_start', 'cases'}
if not required_cols.issubset(data.columns):
    st.error(f"CSV must contain the columns: {required_cols}")
    st.stop()

data = data.dropna(subset=['week_start', 'cases'])
data['week_start'] = pd.to_datetime(data['week_start'])

# Layout: Side-by-side columns
left_col, right_col = st.columns([1.1, 1.9])

with left_col:
    st.subheader("✏️ Review Weekly Case Data")

    # Sort so newest is at the bottom
    editable_data = data.copy().sort_values(by="week_start").reset_index(drop=True)

    # Scroll-height adjusted table
    edited_data = st.data_editor(
        editable_data,
        num_rows="dynamic",
        use_container_width=True,
        height=450,
        key="editable_table"
    )

    if st.button("💾 Save & Update Forecast"):
        try:
            edited_data.to_csv(DATA_FILENAME, index=False)
            st.session_state.data = edited_data
            st.success("✅ Data saved and forecast updated.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to save: {e}")
            st.stop()

# Use saved edited data if available, else fallback to original
working_data = st.session_state.data if st.session_state.data is not None else data

# === Forecasting Pipeline ===
weekly_cases = working_data.groupby('week_start')['cases'].sum().reset_index()
weekly_cases.rename(columns={'week_start': 'ds', 'cases': 'y'}, inplace=True)

start_date = pd.Timestamp('2025-01-01')
if weekly_cases['ds'].min() > start_date:
    missing_dates = pd.date_range(start=start_date, end=weekly_cases['ds'].min() - pd.Timedelta(days=1), freq='W-SUN')
    missing_data = pd.DataFrame({'ds': missing_dates, 'y': 0})
    weekly_cases = pd.concat([missing_data, weekly_cases]).sort_values(by='ds').reset_index(drop=True)

model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
model.add_seasonality(name='custom_yearly', period=52.1775, fourier_order=10)
model.fit(weekly_cases)

future_start = weekly_cases['ds'].max() + pd.Timedelta(weeks=1)
future_end = pd.Timestamp('2025-12-31')
future_dates = pd.date_range(start=future_start, end=future_end, freq='W-SUN')
future = pd.DataFrame({'ds': future_dates})
forecast = model.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast['type'] = 'Forecast'

historical = weekly_cases[['ds', 'y']].rename(columns={'y': 'yhat'})
historical['type'] = 'Actual'

combined = pd.concat([historical, forecast], ignore_index=True)
combined.sort_values(by='ds', inplace=True)
combined['cumulative_yhat'] = combined.loc[combined['ds'] >= start_date, 'yhat'].cumsum()
combined['cumulative_yhat'].fillna(0, inplace=True)

# === Plotting ===
with right_col:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical['ds'], y=historical['yhat'],
        mode='markers', name='Actual Weekly Cases',
        marker=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Forecasted Weekly Cases',
        line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=combined['ds'], y=combined['cumulative_yhat'],
        mode='lines', name='Cumulative Cases from Jan 1, 2025',
        line=dict(color='green', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        mode='lines', line=dict(color='orange', dash='dot'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        mode='lines', line=dict(color='orange', dash='dot'),
        fill='tonexty', showlegend=False
    ))
    fig.add_shape(
        type="line", x0=future_start, x1=future_start, y0=0, y1=1,
        xref="x", yref="paper", line=dict(color="red", dash="dash")
    )
    fig.add_annotation(
        x=future_start, y=1, xref="x", yref="paper",
        text="Forecast Start", showarrow=False, font=dict(color="red")
    )

    fig.update_layout(
        title="📊 Weekly Measles Forecast with Cumulative Totals (2025)",
        xaxis_title="Date - Week Start", yaxis_title="Cases",
        template="plotly_white", height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# Final stat
final_total = combined[combined['ds'] <= future_end]['cumulative_yhat'].iloc[-1]
st.success(f"📌 Total actual + forecasted cases through 2025-12-31: {final_total:,.0f}")