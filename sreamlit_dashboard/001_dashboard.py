import streamlit as st
import pandas as pd

# Title
st.title("Interactive Dashboard with Group By and Aggregations")

# Sample dataset columns
columns = ["customer_id", "product", "price", "date", "brand"]

# File Upload
st.sidebar.title("Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Load dataset
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("CSV File Loaded Successfully!")
else:
    st.info("Loading sample dataset...")
    # Sample data if no file is uploaded
    data = pd.DataFrame({
        "customer_id": [1, 1, 2, 2, 3],
        "product": ["A", "B", "A", "C", "B"],
        "price": [10, 20, 15, 30, 25],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-04"],
        "brand": ["X", "Y", "X", "Y", "Z"]
    })

# Show dataset
st.subheader("Dataset Preview")
st.write(data)

# Buttons for aggregation
st.subheader("Actions")
aggregation_type = st.radio(
    "Select an Aggregation Type:",
    ["Group By", "Sum Prices", "Average Prices", "Count Products"]
)

if aggregation_type == "Group By":
    group_by_col = st.selectbox("Select column to Group By:", columns)
    grouped_data = data.groupby(group_by_col).size().reset_index(name="Count")
    st.write("Grouped Data:")
    st.write(grouped_data)

elif aggregation_type == "Sum Prices":
    group_by_col = st.selectbox("Select column to Sum Prices:", columns)
    summed_data = data.groupby(group_by_col)["price"].sum().reset_index()
    st.write("Summed Prices:")
    st.write(summed_data)

elif aggregation_type == "Average Prices":
    group_by_col = st.selectbox("Select column to Average Prices:", columns)
    avg_data = data.groupby(group_by_col)["price"].mean().reset_index()
    st.write("Average Prices:")
    st.write(avg_data)

elif aggregation_type == "Count Products":
    group_by_col = st.selectbox("Select column to Count Products:", columns)
    count_data = data.groupby(group_by_col)["product"].count().reset_index(name="Product Count")
    st.write("Product Count:")
    st.write(count_data)

# Add download button for processed data
if st.button("Download Processed Data"):
    csv_data = grouped_data.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv_data,
        file_name="processed_data.csv",
        mime="text/csv"
    )


import streamlit as st
import pandas as pd

# Title
st.title("Interactive Dashboard with Group By and Aggregations")

# Sample dataset columns
columns = ["customer_id", "product", "price", "date", "brand"]

import streamlit as st
import pandas as pd

# Title
st.title("Interactive Dashboard with Group By and Aggregations")

# Sample dataset columns
columns = ["customer_id", "product", "price", "date", "brand"]