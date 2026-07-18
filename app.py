"""
Blinkit Sales Analysis — Streamlit App (Plotly Edition)
A Comprehensive Data Science Project on Blinkit Sales Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Blinkit Sales Analysis",
    page_icon="🛒",
    layout="wide",
)

PLOTLY_TEMPLATE = "plotly_white"

# -----------------------------------------------------------------------
# HEADER / PROJECT OVERVIEW
# -----------------------------------------------------------------------
st.title("🛒 Blinkit Sales Analysis")
st.markdown("### A Comprehensive Data Science Project on Blinkit Sales Analysis")

with st.expander("📄 Project Overview", expanded=True):
    st.markdown(
        """
The Blinkit Sales dataset contains information about grocery products sold
across different Blinkit outlets. It includes details such as item type,
item weight, fat content, visibility, maximum retail price (MRP), outlet
characteristics, and sales. The dataset consists of **8,523 records** and
**12 features**, making it suitable for data analysis and visualization.
The primary objective of this dataset is to analyze the factors affecting
product sales and identify meaningful business insights.

**Research Questions**
1. How have Blinkit sales evolved across outlet establishment years?
2. How do different product categories influence item outlet sales?
3. How does the type of outlet affect the sales performance of products?
4. What is the relationship between an item's MRP and its sales?
5. How does the location of an outlet impact overall sales performance?
6. Which factors have the greatest influence on item outlet sales?

**Source:** [Kaggle – Blinkit Sales Dataset](https://www.kaggle.com/datasets/mukeshgadri/blinkit-dataset)
        """
    )

with st.expander("⚠️ Important Considerations"):
    st.markdown(
        """
**Dataset Reliability:** The dataset was obtained from Kaggle and is intended
for educational and analytical purposes. It may not represent the complete
operational data of Blinkit.

**Data Limitations:** The dataset contains missing values in some attributes
(such as Item Weight and Outlet Size) and represents sales from a limited
number of outlets. Findings should be interpreted within these constraints.

**Scope of Analysis:** This analysis focuses on identifying sales patterns and
the factors influencing product sales using exploratory data analysis (EDA)
and visualization techniques — intended for learning and business-insight
generation rather than real-world business decisions.
        """
    )

st.divider()

# -----------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------
st.sidebar.header("📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload the Blinkit CSV file", type=["csv"])

DEFAULT_PATH = "mani.csv"


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


ds = None
if uploaded_file is not None:
    ds = load_data(uploaded_file)
else:
    try:
        ds = load_data(DEFAULT_PATH)
        st.sidebar.info(f"Using default file: {DEFAULT_PATH}")
    except FileNotFoundError:
        st.warning(
            "No data file found. Please upload the Blinkit sales CSV file "
            "using the sidebar to begin the analysis."
        )
        st.stop()

# -----------------------------------------------------------------------
# DATA CLEANING (mirrors the original notebook)
# -----------------------------------------------------------------------
@st.cache_data
def clean_data(df):
    df = df.copy()

    # Fill missing numeric values (Item_Weight) with the column mean
    if "Item_Weight" in df.columns:
        mean_weight = df["Item_Weight"].mean()
        df["Item_Weight"] = df["Item_Weight"].fillna(mean_weight)

    # Fill missing categorical values (Outlet_Size) with the column mode
    if "Outlet_Size" in df.columns:
        mode_size = df["Outlet_Size"].mode()[0]
        df["Outlet_Size"] = df["Outlet_Size"].fillna(mode_size)

    # Standardize Item_Fat_Content labels
    if "Item_Fat_Content" in df.columns:
        df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
            {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
        )

    return df


ds = clean_data(ds)

required_cols = {
    "Item_Type",
    "Item_Outlet_Sales",
    "Outlet_Type",
    "Outlet_Location_Type",
    "Item_Fat_Content",
    "Outlet_Establishment_Year",
    "Item_MRP",
    "Item_Visibility",
}
missing_cols = required_cols - set(ds.columns)
if missing_cols:
    st.error(f"The uploaded file is missing expected columns: {missing_cols}")
    st.stop()

# -----------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------
st.sidebar.header("🔍 Filters")

item_types = sorted(ds["Item_Type"].dropna().unique())
outlet_types = sorted(ds["Outlet_Type"].dropna().unique())
location_types = sorted(ds["Outlet_Location_Type"].dropna().unique())

selected_items = st.sidebar.multiselect("Item Type", item_types, default=item_types)
selected_outlets = st.sidebar.multiselect("Outlet Type", outlet_types, default=outlet_types)
selected_locations = st.sidebar.multiselect(
    "Outlet Location Type", location_types, default=location_types
)

filtered = ds[
    ds["Item_Type"].isin(selected_items)
    & ds["Outlet_Type"].isin(selected_outlets)
    & ds["Outlet_Location_Type"].isin(selected_locations)
]

if filtered.empty:
    st.warning("No data matches the selected filters. Adjust filters in the sidebar.")
    st.stop()

# -----------------------------------------------------------------------
# KEY METRICS
# -----------------------------------------------------------------------
st.subheader("📊 Key Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{filtered.shape[0]:,}")
c2.metric("Total Sales", f"{filtered['Item_Outlet_Sales'].sum():,.0f}")
c3.metric("Average Sales", f"{filtered['Item_Outlet_Sales'].mean():,.2f}")
c4.metric("Average MRP", f"{filtered['Item_MRP'].mean():,.2f}")

st.divider()

# -----------------------------------------------------------------------
# DATA PREVIEW / QUALITY
# -----------------------------------------------------------------------
with st.expander("🗂️ Data Preview"):
    st.dataframe(filtered.head(50), use_container_width=True)

with st.expander("🧹 Data Quality Summary"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Missing values (post-cleaning):**")
        st.dataframe(ds.isnull().sum().rename("null_count"))
    with col_b:
        st.write("**Duplicate rows:**", int(ds.duplicated().sum()))
        st.write("**Data types:**")
        st.dataframe(ds.dtypes.astype(str).rename("dtype"))

with st.expander("📈 Statistical Summary"):
    st.dataframe(filtered.describe(include="all"), use_container_width=True)

st.divider()

# -----------------------------------------------------------------------
# EDA VISUALIZATIONS (PLOTLY)
# -----------------------------------------------------------------------
st.header("🔎 Exploratory Data Analysis")

tabs = st.tabs(
    [
        "1. Top Item Types",
        "2. Sales by Outlet Type",
        "3. Sales by Location",
        "4. Fat Content",
        "5. Establishment Year",
        "6. MRP by Item Type",
        "7. MRP Distribution",
        "8. Visibility Distribution",
        "9. MRP vs Sales",
        "10. Correlation Heatmap",
        "11. Sales Violin Plot",
        "12. Category Counts",
    ]
)

# 1. Top Selling Item Types by Total Sales
with tabs[0]:
    st.subheader("Top Selling Item Types by Total Sales")
    agg = (
        filtered.groupby("Item_Type")["Item_Outlet_Sales"]
        .sum()
        .sort_values()
        .reset_index()
    )
    fig = px.bar(
        agg,
        x="Item_Outlet_Sales",
        y="Item_Type",
        orientation="h",
        color="Item_Outlet_Sales",
        color_continuous_scale="Blues",
        labels={"Item_Outlet_Sales": "Total Sales", "Item_Type": "Item Type"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=600, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- The chart shows total sales generated by each product category.
- It helps identify the best-selling item types and their contribution to overall sales.
- Product categories with higher sales indicate greater customer demand.
        """
    )

# 2. Average Sales by Outlet Type
with tabs[1]:
    st.subheader("Average Sales by Outlet Type")
    agg = (
        filtered.groupby("Outlet_Type")["Item_Outlet_Sales"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig = px.bar(
        agg,
        x="Outlet_Type",
        y="mean",
        error_y="std",
        color="Outlet_Type",
        labels={"mean": "Average Sales", "Outlet_Type": "Outlet Type"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False, height=550)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Compares average sales across different outlet types.
- Supermarkets generally generate higher average sales than grocery stores.
- Outlet type significantly affects sales performance.
        """
    )

# 3. Sales Performance Across Different Outlet Locations
with tabs[2]:
    st.subheader("Sales Performance Across Different Outlet Locations")
    fig = px.box(
        filtered,
        x="Outlet_Location_Type",
        y="Item_Outlet_Sales",
        color="Outlet_Location_Type",
        labels={
            "Outlet_Location_Type": "Outlet Location Type",
            "Item_Outlet_Sales": "Item Outlet Sales",
        },
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False, height=550)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Illustrates the distribution of sales across different outlet locations.
- Sales vary according to outlet location.
- Location plays an important role in influencing overall sales performance.
        """
    )

# 4. Distribution of Item Fat Content
with tabs[3]:
    st.subheader("Distribution of Item Fat Content")
    counts = filtered["Item_Fat_Content"].value_counts().reset_index()
    counts.columns = ["Item_Fat_Content", "count"]
    fig = px.pie(
        counts,
        names="Item_Fat_Content",
        values="count",
        hole=0.35,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", pull=0.03)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Represents the proportion of Low Fat and Regular products.
- Low Fat products make up the majority of the dataset.
- Provides an overview of product composition based on fat content.
        """
    )

# 5. Impact of Outlet Establishment Year on Sales
with tabs[4]:
    st.subheader("Average Sales by Outlet Establishment Year")
    agg = (
        filtered.groupby("Outlet_Establishment_Year")["Item_Outlet_Sales"]
        .mean()
        .reset_index()
    )
    fig = px.line(
        agg,
        x="Outlet_Establishment_Year",
        y="Item_Outlet_Sales",
        markers=True,
        labels={
            "Outlet_Establishment_Year": "Establishment Year",
            "Item_Outlet_Sales": "Average Sales",
        },
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Compares average sales of outlets established in different years.
- Helps determine whether older or newer outlets perform better.
- Establishment year has an impact on sales.
        """
    )

# 6. Average Item MRP by Item Type
with tabs[5]:
    st.subheader("Average Item MRP by Item Type")
    agg = (
        filtered.groupby("Item_Type")["Item_MRP"]
        .mean()
        .sort_values()
        .reset_index()
    )
    fig = px.bar(
        agg,
        x="Item_MRP",
        y="Item_Type",
        orientation="h",
        color="Item_MRP",
        color_continuous_scale="Oranges",
        labels={"Item_MRP": "Average MRP", "Item_Type": "Item Type"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=600, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Compares average MRP across product categories.
- Some product categories have higher average prices than others.
- Provides insights into the pricing patterns of different items.
        """
    )

# 7. Distribution of Item MRP
with tabs[6]:
    st.subheader("Distribution of Item MRP")
    fig = px.histogram(
        filtered,
        x="Item_MRP",
        nbins=30,
        color_discrete_sequence=["green"],
        labels={"Item_MRP": "Item MRP"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=550, yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Shows the distribution of product prices in the dataset.
- Most products are concentrated within a moderate price range.
- Only a small number of products have very high MRP values.
        """
    )

# 8. Distribution of Item Visibility
with tabs[7]:
    st.subheader("Distribution of Item Visibility")
    fig = px.histogram(
        filtered,
        x="Item_Visibility",
        nbins=30,
        color_discrete_sequence=["purple"],
        labels={"Item_Visibility": "Item Visibility"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=550, yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Displays the distribution of item visibility values.
- Most products have low to medium visibility.
- Only a few products have exceptionally high visibility in stores.
        """
    )

# 9. Relationship Between Item MRP and Sales
with tabs[8]:
    st.subheader("Relationship Between Item MRP and Sales")
    fig = px.scatter(
        filtered,
        x="Item_MRP",
        y="Item_Outlet_Sales",
        color_discrete_sequence=["red"],
        opacity=0.6,
        labels={"Item_MRP": "Item MRP", "Item_Outlet_Sales": "Item Outlet Sales"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Shows the relationship between Item MRP and sales.
- A positive trend can be observed between price and sales.
- Higher-priced products generally generate higher sales revenue.
        """
    )

# 10. Correlation Heatmap of Numerical Features
with tabs[9]:
    st.subheader("Correlation Heatmap of Numerical Features")
    corr = filtered.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Shows the correlation among numerical variables.
- Helps identify which features have stronger relationships with sales.
- Item MRP shows a stronger positive correlation with sales than other numerical features.
        """
    )

# 11. Sales Distribution by Outlet Type (Violin)
with tabs[10]:
    st.subheader("Sales Distribution by Outlet Type")
    fig = px.violin(
        filtered,
        x="Outlet_Type",
        y="Item_Outlet_Sales",
        color="Outlet_Type",
        box=True,
        points=False,
        labels={"Outlet_Type": "Outlet Type", "Item_Outlet_Sales": "Item Outlet Sales"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False, height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Compares the distribution of sales across outlet types.
- Some outlet types show higher median sales and greater variation.
- Provides insights into the consistency of sales performance among outlets.
        """
    )

# 12. Product Category Distribution
with tabs[11]:
    st.subheader("Product Category Distribution")
    counts = filtered["Item_Type"].value_counts().reset_index()
    counts.columns = ["Item_Type", "count"]
    fig = px.bar(
        counts,
        x="count",
        y="Item_Type",
        orientation="h",
        color="count",
        color_continuous_scale="Teal",
        labels={"count": "Count", "Item_Type": "Item Type"},
        template=PLOTLY_TEMPLATE,
        category_orders={"Item_Type": counts["Item_Type"].tolist()},
    )
    fig.update_layout(height=600, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
**Key Insights**
- Displays the number of products available in each category.
- Some product categories contain more products than others.
- Helps understand the overall composition of the dataset.
        """
    )

st.divider()

# -----------------------------------------------------------------------
# KEY FINDINGS AND CONCLUSIONS
# -----------------------------------------------------------------------
st.header("🧾 Key Findings and Conclusions")
st.markdown(
    """
Product attributes and outlet characteristics significantly influence sales,
with **Item MRP, outlet type, and product category** being the key factors
affecting Blinkit sales performance.
"""
)

f1, f2 = st.columns(2)
with f1:
    st.markdown(
        """
**1. Product Category**
- Different product categories contribute differently to overall sales.
- Some categories generate higher sales due to greater customer demand.
- Product category is an important factor affecting sales performance.

**2. Outlet Type**
- Sales vary significantly across different outlet types.
- Supermarkets generally record higher average sales than grocery stores.
- Outlet type plays a key role in business performance.

**3. Item MRP**
- Item MRP has a positive relationship with sales.
- Higher-priced products tend to generate greater sales revenue.
- Pricing is an important factor influencing customer purchases.
        """
    )
with f2:
    st.markdown(
        """
**4. Outlet Location**
- Sales performance differs across outlet locations.
- Location affects customer reach and product demand.
- Well-located outlets generally achieve better sales performance.

**5. Overall Conclusion**
- Both product attributes and outlet characteristics influence sales.
- Data analysis helps identify key factors affecting business performance.
- These insights can support better pricing, inventory, and marketing decisions.
        """
    )

st.subheader("🎯 Business Recommendations")
st.markdown(
    """
1. **Product Management** — Increase the availability of high-selling product categories to meet customer demand.
2. **Pricing Strategy** — Adopt competitive pricing for products while maintaining profitability.
3. **Inventory Planning** — Maintain adequate stock levels for popular products to reduce stock shortages.
4. **Outlet Expansion** — Focus on expanding outlets in locations with higher sales potential.
"""
)

st.subheader("🏪 Store Management")
st.markdown(
    """
- **Customer Experience:** Improve product visibility and store layout to enhance customer satisfaction.
- **Outlet Performance:** Monitor outlet-wise sales regularly to identify high- and low-performing stores.
- **Marketing Strategies:** Design promotional campaigns based on customer preferences and product demand.
"""
)

st.subheader("📋 Future Scope")
st.markdown(
    """
- **Sales Prediction:** Build machine learning models to forecast future sales.
- **Customer Analysis:** Include customer demographics and purchasing behavior for deeper insights.
- **Business Intelligence:** Develop interactive dashboards for real-time sales monitoring.
- **Advanced Analytics:** Apply clustering and recommendation techniques to improve business strategies.
"""
)

st.caption(
    "Data source: Kaggle – Blinkit Sales Dataset. This analysis is intended "
    "for educational and analytical purposes only."
)