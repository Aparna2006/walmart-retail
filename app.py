import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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