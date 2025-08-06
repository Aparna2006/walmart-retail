import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly

# 🔼 2. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("ml/sales_forecast_model.pkl")

model = load_model()

# 🔽 3. Streamlit UI starts here
st.title("📊 Walmart Retail Dashboard")
st.write("Welcome to the AI-powered sales forecasting dashboard!")

# (Optional) Existing charts or overview code...

# 🔽 4. Forecasting Section (Place this here!)
st.markdown("---")
st.subheader("🔮 Predict Future Sales")

# Slider input
n_days = st.slider("Select number of future days to forecast:", min_value=7, max_value=90, step=7, value=30)

# Button to trigger prediction
if st.button("Predict Future Sales"):
    # Step 1: Create future dates
    future = model.make_future_dataframe(periods=n_days)

    # Step 2: Predict
    forecast = model.predict(future)

    # Step 3: Success Message
    st.success(f"Forecast generated for the next {n_days} days.")

    # Step 4: Table Output
    st.write("📋 Predicted Sales:")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days))

    # Step 5: Chart Output
    st.write("📉 Forecast Chart:")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig, use_container_width=True)

# Load data
data = pd.read_csv("data/walmart_full_dataset.csv")

# Title and description
st.title("🛒 Walmart AI Retail Dashboard")
st.markdown("This Streamlit app visualizes Walmart retail sales data using charts and summaries.")

# Show sample data
st.subheader("📊 Sample Data")
st.dataframe(data.head())

# Category-wise sales chart
st.subheader("🧾 Total Sales by Category")
category_sales = data.groupby("Category")["Sales"].sum().sort_values(ascending=False)
fig1, ax1 = plt.subplots()
category_sales.plot(kind='bar', ax=ax1)
ax1.set_ylabel("Total Sales")
st.pyplot(fig1)

# Store-wise sales chart
st.subheader("🏬 Total Sales by Store")
store_sales = data.groupby("Store ID")["Sales"].sum().sort_values(ascending=False)
fig2, ax2 = plt.subplots()
store_sales.plot(kind='bar', ax=ax2, color="green")
ax2.set_ylabel("Total Sales")
st.pyplot(fig2)

# Filter by date
st.subheader("📅 Filter Data by Date")
start_date = st.date_input("Start Date", pd.to_datetime(data["Date"]).min())
end_date = st.date_input("End Date", pd.to_datetime(data["Date"]).max())
filtered_data = data[(pd.to_datetime(data["Date"]) >= pd.to_datetime(start_date)) &
                     (pd.to_datetime(data["Date"]) <= pd.to_datetime(end_date))]

st.write(f"Filtered data between {start_date} and {end_date}")
st.dataframe(filtered_data)

# Download filtered data
st.subheader("⬇ Download Filtered Data")
st.download_button("Download CSV", filtered_data.to_csv(index=False), "filtered_walmart_data.csv", "text/csv")
