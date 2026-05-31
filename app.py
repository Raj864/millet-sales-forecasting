
import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Millet Analytics Pro",layout="wide")

st.markdown("<h1 style='text-align:center;'>🌾 Millet Sales Forecasting & Analytics Pro</h1>", unsafe_allow_html=True)

@st.cache_data
def load():
    df=pd.read_csv("millet_sales_data.csv")
    df["Date"]=pd.to_datetime(df["Date"])
    return df

df=load()

st.sidebar.header("Filters")
state=st.sidebar.multiselect("State", sorted(df["State"].unique()))
product=st.sidebar.multiselect("Product", sorted(df["Product"].unique()))

f=df.copy()
if state: f=f[f["State"].isin(state)]
if product: f=f[f["Product"].isin(product)]

c1,c2,c3,c4=st.columns(4)
c1.metric("Revenue", f"₹{f['Revenue'].sum():,.0f}")
c2.metric("Profit", f"₹{f['Profit'].sum():,.0f}")
c3.metric("Quantity", f"{f['Quantity'].sum():,}")
c4.metric("Products", f["Product"].nunique())





st.subheader("🤖 AI Business Insights")

total_revenue = f["Revenue"].sum()
total_profit = f["Profit"].sum()

best_product = (
    f.groupby("Product")["Revenue"]
    .sum()
    .idxmax()
)

best_state = (
    f.groupby("State")["Revenue"]
    .sum()
    .idxmax()
)

best_channel = (
    f.groupby("Channel")["Revenue"]
    .sum()
    .idxmax()
)

profit_margin = (
    total_profit / total_revenue
) * 100

st.info(f"""
📈 Total Revenue Generated: ₹{total_revenue:,.0f}

💰 Overall Profit: ₹{total_profit:,.0f}

🏆 Best Selling Product: {best_product}

🌍 Best Performing State: {best_state}

🛒 Most Successful Sales Channel: {best_channel}

📊 Profit Margin: {profit_margin:.2f}%

🎯 Recommendation:
Increase marketing investment in {best_state}
and prioritize inventory for {best_product}.
""")















daily=f.groupby("Date")["Revenue"].sum().reset_index()
st.plotly_chart(px.line(daily,x="Date",y="Revenue",title="Revenue Trend"),use_container_width=True)

a,b=st.columns(2)
with a:
    prod=f.groupby("Product")["Revenue"].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(prod,x="Product",y="Revenue",title="Product Performance"),use_container_width=True)

with b:
    state_df=f.groupby("State")["Revenue"].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(state_df,x="State",y="Revenue",title="State Performance"),use_container_width=True)

st.subheader("📈 30-Day Forecast")
series=df.groupby("Date")["Revenue"].sum()
model=ARIMA(series,order=(3,1,2))
fit=model.fit()
fc=fit.forecast(30)
fc_df=fc.reset_index()
fc_df.columns=["Date","Forecast Revenue"]
st.plotly_chart(px.line(fc_df,x="Date",y="Forecast Revenue",title="Future Revenue Forecast"),use_container_width=True)
st.dataframe(fc_df,use_container_width=True)

csv=f.to_csv(index=False).encode()
st.download_button("⬇ Download Filtered Dataset",csv,"filtered_sales.csv","text/csv")
