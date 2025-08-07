import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from prophet.plot import plot_plotly
from datetime import datetime

# ------------------------------------------
# 🚀 Streamlit Page Config
# ------------------------------------------
st.set_page_config(page_title="Walmart AI Dashboard", layout="wide")
st.title("🛒 Walmart AI Retail Dashboard")
st.markdown("Upload your sales CSV to generate insights, stock alerts, and AI-powered forecasts.")

# ------------------------------------------
# 📁 File Upload
# ------------------------------------------
st.subheader("📁 Upload Your Walmart Sales CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Load the Prophet model
@st.cache_resource
def load_model():
    return joblib.load("ml/sales_forecast_model.pkl")

model = load_model()

# If file is uploaded
if uploaded_file:
    # Load data from uploaded file
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip().str.lower()  # Normalize columns

    # Check required columns
    required_columns = ['product', 'store id', 'stock level', 'sales', 'category', 'date']
    missing_cols = [col for col in required_columns if col not in data.columns]

    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
    else:
        # ------------------------------------------
        # 📦 Low Stock Alerts
        # ------------------------------------------
        st.subheader("📦 Low Stock Alerts")
        low_stock_threshold = 50
        low_stock = data[data['stock level'] < low_stock_threshold][['product', 'store id', 'stock level']]

        if low_stock.empty:
            st.success("✅ All products are sufficiently stocked.")
        else:
            for _, row in low_stock.iterrows():
                st.error(f"⚠️ {row['product']} is low in stock at Store {row['store id']} (Only {row['stock level']} left)")

        # ------------------------------------------
        # 💬 Product Query Bot
        # ------------------------------------------
        st.subheader("💬 Product Location Assistant")
        product_locations = {
            "toothpaste": "Aisle 3, Personal Care section",
            "shampoo": "Aisle 4, Hair Care section",
            "milk": "Aisle 1, Dairy section",
            "bread": "Aisle 2, Bakery section",
            "eggs": "Aisle 1, Dairy section",
            "soap": "Aisle 3, Personal Care section",
            "face mask": "Aisle 8, Health & Wellness",
            "vitamins": "Aisle 9, Health & Wellness",
            "sanitizer": "Aisle 3, Personal Care section",
            "pain relief spray": "Aisle 9, Health & First Aid",
            "blood pressure monitor": "Aisle 10, Health Devices",
            "thermometer": "Aisle 10, Health Devices",
            "smartphone": "Electronics Zone, Section A",
            "headphones": "Electronics Zone, Section B",
            "laptop": "Electronics Zone, Section C",
            "usb cable": "Electronics Zone, Accessories",
            "batteries": "Aisle 15, Electronics Accessories",
            "detergent": "Aisle 6, Home Cleaning",
            "dish soap": "Aisle 6, Home Cleaning",
            "toilet paper": "Aisle 7, Home Essentials",
            "trash bags": "Aisle 7, Home Essentials",
            "rice": "Aisle 12, Grocery Staples",
            "pasta": "Aisle 12, Grocery Staples",
            "cooking oil": "Aisle 13, Grocery Staples",
            "sugar": "Aisle 12, Grocery Staples",
            "salt": "Aisle 12, Grocery Staples"
        }

        user_query = st.text_input("Ask where a product is (e.g., 'Where is shampoo?')")

        if user_query:
            found = False
            for product, location in product_locations.items():
                if product.lower() in user_query.lower():
                    st.info(f"🧾 **{product.capitalize()}** is located at: {location}")
                    found = True
                    break
            if not found:
                st.warning("❓ Sorry, I couldn't find that product in my database.")

        # ------------------------------------------
        # 🔮 Sales Forecasting
        # ------------------------------------------
        st.markdown("---")
        st.subheader("🔮 Predict Future Sales")
        n_days = st.slider("Select number of future days to forecast:", min_value=7, max_value=90, step=7, value=30)

        if st.button("📈 Predict Future Sales"):
            future = model.make_future_dataframe(periods=n_days)
            forecast = model.predict(future)

            st.success(f"✅ Forecast generated for the next {n_days} days.")
            st.write("📋 Forecast Table")
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days)
            st.dataframe(forecast_table)

            st.download_button(
                label="⬇ Download Forecast CSV",
                data=forecast_table.to_csv(index=False),
                file_name="forecast_data.csv",
                mime="text/csv"
            )

            st.write("📉 Forecast Chart")
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------
        # 📊 Sales Visualizations
        # ------------------------------------------
        st.markdown("---")
        st.subheader("📊 Walmart Sales Data Overview")
        st.write("🔍 Sample Data")
        st.dataframe(data.head())

        # Total Sales by Category
        st.subheader("🧾 Total Sales by Category")
        category_sales = data.groupby("category")["sales"].sum().sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        category_sales.plot(kind='bar', ax=ax1)
        ax1.set_ylabel("Total Sales")
        st.pyplot(fig1)

        # Total Sales by Store
        st.subheader("🏬 Total Sales by Store")
        store_sales = data.groupby("store id")["sales"].sum().sort_values(ascending=False)
        fig2, ax2 = plt.subplots()
        store_sales.plot(kind='bar', ax=ax2, color="green")
        ax2.set_ylabel("Total Sales")
        st.pyplot(fig2)

        # ------------------------------------------
        # 📅 Date Filtering + Download
        # ------------------------------------------
        st.subheader("📅 Filter Data by Date")
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        start_date = st.date_input("Start Date", data["date"].min())
        end_date = st.date_input("End Date", data["date"].max())

        filtered_data = data[
            (data["date"] >= pd.to_datetime(start_date)) &
            (data["date"] <= pd.to_datetime(end_date))
        ]

        st.write(f"Showing data from **{start_date}** to **{end_date}**")
        st.dataframe(filtered_data)

        st.download_button(
            label="⬇ Download Filtered CSV",
            data=filtered_data.to_csv(index=False),
            file_name="filtered_sales_data.csv",
            mime="text/csv"
        )

else:
    st.warning("📂 Please upload a Walmart sales CSV file to generate your dashboard.")
