
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Millet Sales Forecasting", layout="wide")

@st.cache_data
def load():
    df=pd.read_csv("millet_sales_data.csv")
    df["Date"]=pd.to_datetime(df["Date"])
    return df

df=load()

st.title("🌾 Millet Sales Forecasting & Analytics Dashboard")

c1,c2,c3=st.columns(3)
c1.metric("Total Revenue", f"₹{df['Revenue'].sum():,.0f}")
c2.metric("Total Profit", f"₹{df['Profit'].sum():,.0f}")
c3.metric("Total Quantity", f"{df['Quantity'].sum():,}")

st.subheader("Revenue Trend")
daily=df.groupby("Date")["Revenue"].sum()

fig,ax=plt.subplots(figsize=(10,4))
ax.plot(daily.index,daily.values)
st.pyplot(fig)

col1,col2=st.columns(2)

with col1:
    st.subheader("Product Performance")
    st.bar_chart(df.groupby("Product")["Revenue"].sum())

with col2:
    st.subheader("State Performance")
    st.bar_chart(df.groupby("State")["Revenue"].sum())

st.subheader("30-Day Sales Forecast")
model=ARIMA(daily,order=(3,1,2))
fit=model.fit()
forecast=fit.forecast(30)

fig2,ax2=plt.subplots(figsize=(10,4))
ax2.plot(daily.tail(120),label="Historical")
ax2.plot(forecast.index,forecast.values,label="Forecast")
ax2.legend()
st.pyplot(fig2)

st.dataframe(forecast.reset_index().rename(columns={0:"Forecast Revenue"}))
