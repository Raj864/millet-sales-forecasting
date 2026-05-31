
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

# st.sidebar.header("Filters")
# state=st.sidebar.multiselect("State", sorted(df["State"].unique()))
# product=st.sidebar.multiselect("Product", sorted(df["Product"].unique()))





st.sidebar.header("🎛 Dashboard Filters")

# State Filter
selected_states = st.sidebar.multiselect(
    "🌍 Select State",
    sorted(df["State"].unique())
)

# Product Filter
selected_products = st.sidebar.multiselect(
    "🌾 Select Product",
    sorted(df["Product"].unique())
)

# Channel Filter
selected_channels = st.sidebar.multiselect(
    "🛒 Sales Channel",
    sorted(df["Channel"].unique())
)

# Date Range Filter
start_date = st.sidebar.date_input(
    "📅 Start Date",
    df["Date"].min()
)

end_date = st.sidebar.date_input(
    "📅 End Date",
    df["Date"].max()
)






# f=df.copy()
# if state: f=f[f["State"].isin(state)]
# if product: f=f[f["Product"].isin(product)]




f = df.copy()

if selected_states:
    f = f[f["State"].isin(selected_states)]

if selected_products:
    f = f[f["Product"].isin(selected_products)]

if selected_channels:
    f = f[f["Channel"].isin(selected_channels)]

f = f[
    (f["Date"] >= pd.to_datetime(start_date))
    &
    (f["Date"] <= pd.to_datetime(end_date))
]






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







st.subheader("📦 Production Planning Simulator")

increase_production = st.slider(
    "Increase Production (%)",
    0,
    100,
    20
)

current_quantity = f["Quantity"].sum()

future_quantity = (
    current_quantity *
    (1 + increase_production/100)
)

avg_price = (
    f["Revenue"].sum() /
    f["Quantity"].sum()
)

expected_revenue = (
    future_quantity *
    avg_price
)

st.metric(
    "Expected Revenue",
    f"₹{expected_revenue:,.0f}"
)

st.write(
    f"""
Current Quantity: {current_quantity:,.0f}

Future Quantity: {future_quantity:,.0f}

Expected Revenue:
₹{expected_revenue:,.0f}
"""
)
















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
