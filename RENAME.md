import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Analytics & Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        div[data-testid="stMetric"] {
            background-color: #f5f7fb;
            border: 1px solid #e5e7eb;
            padding: 16px;
            border-radius: 12px;
        }

        h1, h2, h3 {
            font-weight: 650;
        }

        .subtitle {
            color: #64748b;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Retail Analytics & Intelligence System")

st.markdown(
    '<p class="subtitle">'
    'An interactive retail data analysis application using '
    'Python, Pandas, Plotly and SQL concepts.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):
    """
    Find a column using case-insensitive matching.
    Also supports common pandas duplicate-column suffixes.
    """

    def normalize(name):
        return "".join(
            character.lower()
            for character in str(name)
            if character.isalnum()
        )

    columns = list(df.columns)

    # Exact normalized matches first
    for possible_name in possible_names:
        target = normalize(possible_name)

        for column in columns:
            if normalize(column) == target:
                return column

    # Allow common suffixes such as StoreID.1
    for possible_name in possible_names:
        target = normalize(possible_name)

        for column in columns:
            current = normalize(column)

            if current.startswith(target):
                suffix = current[len(target):]

                if suffix.isdigit():
                    return column

    return None


def to_numeric_series(series):
    """Convert numeric-like values into numeric data."""

    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


def safe_sum(series):
    """Return a numeric sum, using zero for empty results."""

    return float(series.sum()) if not series.empty else 0.0


def format_number(value):
    return f"{value:,.2f}"


# ============================================================
# SIDEBAR: FILE UPLOAD
# ============================================================

st.sidebar.header("⚙️ Data Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload your retail CSV file",
    type=["csv"]
)

st.sidebar.caption(
    "Supported file format: CSV. "
    "Use your retail transaction dataset."
)


if uploaded_file is None:
    st.info(
        "👋 Welcome! Upload your retail CSV file from the sidebar "
        "to start the analysis."
    )

    st.subheader("What this application offers")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 Sales Analysis")
        st.write(
            "Explore revenue, transactions, sales trends "
            "and store performance."
        )

    with col2:
        st.markdown("### 🧹 Data Quality")
        st.write(
            "Inspect missing values, duplicates, data types "
            "and unusual transaction values."
        )

    with col3:
        st.markdown("### 🔎 Business Insights")
        st.write(
            "Compare products, payment methods, locations "
            "and cashier activity."
        )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:
    uploaded_file.seek(0)

    df_original = pd.read_csv(uploaded_file)

except Exception as error:
    st.error(f"Unable to read the CSV file: {error}")
    st.stop()


if df_original.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()


# Remove spaces from column names.
# Keep duplicate column names visible so users can inspect them.

df_original.columns = [
    str(column).strip()
    for column in df_original.columns
]

df = df_original.copy()


# ============================================================
# COLUMN DETECTION
# ============================================================

date_col = find_column(
    df,
    ["Date", "TransactionDate", "SaleDate"]
)

time_col = find_column(
    df,
    ["Time", "TransactionTime", "SaleTime"]
)

store_col = find_column(
    df,
    ["StoreID", "Store_ID", "Store"]
)

location_col = find_column(
    df,
    ["Location", "StoreLocation"]
)

product_col = find_column(
    df,
    ["Product", "ProductName", "Item"]
)

quantity_col = find_column(
    df,
    ["Quantity", "Qty", "UnitsSold"]
)

unit_price_col = find_column(
    df,
    ["UnitPrice", "Unit_Price", "PricePerUnit"]
)

payment_col = find_column(
    df,
    ["PaymentType", "PaymentMethod", "Payment_Type"]
)

transaction_col = find_column(
    df,
    ["TransactionID", "Transaction_ID", "TranID"]
)

cashier_col = find_column(
    df,
    ["Cashier", "CashierName"]
)

manager_col = find_column(
    df,
    ["StoreManager", "Store_Manager", "Manager"]
)

time_of_day_col = find_column(
    df,
    ["TimeOfDay", "Time_Of_Day"]
)

day_of_week_col = find_column(
    df,
    ["DayOfWeek", "Day_Of_Week"]
)

total_price_col = find_column(
    df,
    ["TotalPrice", "Total_Price", "TotalSales", "Sales"]
)


# ============================================================
# DATA PREPARATION
# ============================================================

# Convert date values when a date column is available.

if date_col is not None:
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )


# Convert numeric fields.

if quantity_col is not None:
    df[quantity_col] = to_numeric_series(
        df[quantity_col]
    )

if unit_price_col is not None:
    df[unit_price_col] = to_numeric_series(
        df[unit_price_col]
    )

if total_price_col is not None:
    df[total_price_col] = to_numeric_series(
        df[total_price_col]
    )


# Create a calculated transaction value only when
# an existing total-price column is not available.

if total_price_col is not None:
    sales_col = total_price_col

elif quantity_col is not None and unit_price_col is not None:
    df["_CalculatedTotalPrice"] = (
        df[quantity_col] * df[unit_price_col]
    )
    sales_col = "_CalculatedTotalPrice"

else:
    sales_col = None


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

initial_rows = len(df)

duplicate_count = int(df.duplicated().sum())

missing_total = int(df.isnull().sum().sum())


# ============================================================
# SIDEBAR: FILTERS
# ============================================================

st.sidebar.divider()
st.sidebar.header("🔍 Filters")

filtered_df = df.copy()


def apply_multiselect_filter(data, column, label):
    if column is None:
        return data

    values = (
        data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    values = sorted(values)

    if not values:
        return data

    selected_values = st.sidebar.multiselect(
        label,
        options=values,
        default=values
    )

    if not selected_values:
        return data.iloc[0:0]

    return data[
        data[column].astype(str).isin(selected_values)
    ]


filtered_df = apply_multiselect_filter(
    filtered_df, store_col, "Select Store"
)

filtered_df = apply_multiselect_filter(
    filtered_df, location_col, "Select Location"
)

filtered_df = apply_multiselect_filter(
    filtered_df, product_col, "Select Product"
)

filtered_df = apply_multiselect_filter(
    filtered_df, payment_col, "Select Payment Type"
)

filtered_df = apply_multiselect_filter(
    filtered_df, time_of_day_col, "Select Time of Day"
)

filtered_df = apply_multiselect_filter(
    filtered_df, day_of_week_col, "Select Day of Week"
)


# Date filter

if date_col is not None:

    valid_dates = filtered_df[date_col].dropna()

    if not valid_dates.empty:

        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(date_range, (tuple, list)):

            if len(date_range) == 2:
                start_date, end_date = date_range

                filtered_df = filtered_df[
                    filtered_df[date_col].dt.date.between(
                        start_date,
                        end_date
                    )
                ]

            elif len(date_range) == 1:
                selected_date = date_range[0]

                filtered_df = filtered_df[
                    filtered_df[date_col].dt.date
                    == selected_date
                ]


# ============================================================
# HANDLE EMPTY FILTER RESULTS
# ============================================================

if filtered_df.empty:
    st.warning(
        "No records match the selected filters. "
        "Change the filters in the sidebar."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

if sales_col is not None:

    total_revenue = safe_sum(
        filtered_df[sales_col].dropna()
    )

    average_sale = (
        filtered_df[sales_col].mean()
        if filtered_df[sales_col].notna().any()
        else 0
    )

else:
    total_revenue = 0
    average_sale = 0


if transaction_col is not None:

    total_transactions = int(
        filtered_df[transaction_col].nunique()
    )

else:
    # Row count is a proxy when transaction IDs are unavailable.
    total_transactions = len(filtered_df)


if quantity_col is not None:

    total_units = safe_sum(
        filtered_df[quantity_col].dropna()
    )

else:
    total_units = 0


if total_transactions > 0:
    average_transaction_value = (
        total_revenue / total_transactions
    )
else:
    average_transaction_value = 0


# ============================================================
# NAVIGATION TABS
# ============================================================

tabs = st.tabs(
    [
        "🏠 Overview",
        "🏪 Store Analysis",
        "📦 Product Analysis",
        "👨‍💼 Cashier Analysis",
        "🧹 Data Quality",
        "🚨 Anomaly Detection",
        "🗃️ SQL Queries",
        "📋 Data Explorer"
    ]
)


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tabs[0]:

    st.header("Retail Performance Overview")

    st.caption(
        "The metrics below reflect the currently selected filters."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Revenue",
        format_number(total_revenue)
    )

    c2.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    c3.metric(
        "Units Sold",
        format_number(total_units)
    )

    c4.metric(
        "Average Transaction Value",
        format_number(average_transaction_value)
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Sales by Payment Type")

        if payment_col is not None and sales_col is not None:

            payment_data = (
                filtered_df.groupby(
                    payment_col,
                    dropna=False
                )[sales_col]
                .sum()
                .reset_index()
            )

            fig = px.pie(
                payment_data,
                names=payment_col,
                values=sales_col,
                hole=0.45,
                title="Revenue Distribution by Payment Type"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info(
                "Payment type or sales column not available."
            )

    with right:

        st.subheader("Sales by Store")

        if store_col is not None and sales_col is not None:

            store_data = (
                filtered_df.groupby(
                    store_col,
                    dropna=False
                )[sales_col]
                .sum()
                .reset_index()
            )

            store_data = store_data.sort_values(
                sales_col,
                ascending=False
            )

            fig = px.bar(
                store_data,
                x=store_col,
                y=sales_col,
                title="Total Revenue by Store",
                text_auto=".2s"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info(
                "Store ID or sales column not available."
            )

    st.subheader("Sales Trend")

    if date_col is not None and sales_col is not None:

        trend_data = filtered_df.dropna(
            subset=[date_col, sales_col]
        ).copy()

        if not trend_data.empty:

            trend_data["_Month"] = (
                trend_data[date_col].dt.to_period("M")
                .astype(str)
            )

            monthly_sales = (
                trend_data.groupby("_Month")[sales_col]
                .sum()
                .reset_index()
            )

            fig = px.line(
                monthly_sales,
                x="_Month",
                y=sales_col,
                markers=True,
                title="Monthly Revenue Trend"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("No valid dates or sales values found.")

    else:
        st.info(
            "A valid date and sales column are required "
            "for the sales trend."
        )


# ============================================================
# TAB 2: STORE ANALYSIS
# ============================================================

with tabs[1]:

    st.header("Store Performance Analysis")

    if store_col is None:

        st.warning("No StoreID column was detected.")

    else:

        if sales_col is not None:

            store_summary = (
                filtered_df.groupby(
                    store_col,
                    dropna=False
                )
                .agg(
                    Total_Revenue=(sales_col, "sum"),
                    Average_Sale=(sales_col, "mean"),
                    Sales_Records=(sales_col, "count")
                )
                .reset_index()
            )

        else:

            store_summary = (
                filtered_df.groupby(
                    store_col,
                    dropna=False
                )
                .size()
                .reset_index(name="Sales_Records")
            )

        if transaction_col is not None:

            transaction_summary = (
                filtered_df.groupby(
                    store_col,
                    dropna=False
                )[transaction_col]
                .nunique()
                .reset_index(name="Unique_Transactions")
            )

            store_summary = store_summary.merge(
                transaction_summary,
                on=store_col,
                how="left"
            )

        st.subheader("Store Summary")

        st.dataframe(
            store_summary,
            use_container_width=True,
            hide_index=True
        )

        if sales_col is not None and not store_summary.empty:

            st.subheader("Store Revenue Ranking")

            ranking = store_summary.sort_values(
                "Total_Revenue",
                ascending=False
            )

            fig = px.bar(
                ranking,
                x=store_col,
                y="Total_Revenue",
                color="Total_Revenue",
                title="Revenue Ranking by Store"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            best_store = ranking.iloc[0]

            st.success(
                f"Highest revenue in the filtered data: "
                f"{best_store[store_col]} "
                f"({format_number(best_store['Total_Revenue'])})"
            )


# ============================================================
# TAB 3: PRODUCT ANALYSIS
# ============================================================

with tabs[2]:

    st.header("Product Performance Analysis")

    if product_col is None:

        st.warning("No Product column was detected.")

    else:

        aggregation = {}

        if quantity_col is not None:
            aggregation["Total_Quantity"] = (
                quantity_col, "sum"
            )

        if sales_col is not None:
            aggregation["Total_Revenue"] = (
                sales_col, "sum"
            )

            aggregation["Average_Sale"] = (
                sales_col, "mean"
            )

        if aggregation:

            product_summary = (
                filtered_df.groupby(
                    product_col,
                    dropna=False
                )
                .agg(**aggregation)
                .reset_index()
            )

            if "Total_Revenue" in product_summary.columns:

                product_summary = product_summary.sort_values(
                    "Total_Revenue",
                    ascending=False
                )

                st.subheader("Product Revenue Ranking")

                st.dataframe(
                    product_summary,
                    use_container_width=True,
                    hide_index=True
                )

                top_products = product_summary.head(10)

                fig = px.bar(
                    top_products,
                    x="Total_Revenue",
                    y=product_col,
                    orientation="h",
                    title="Top 10 Products by Revenue"
                )

                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.dataframe(
                    product_summary,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.info(
                "Quantity and sales fields are not available "
                "for product aggregation."
            )


# ============================================================
# TAB 4: CASHIER ANALYSIS
# ============================================================

with tabs[3]:

    st.header("Cashier Activity Analysis")

    if cashier_col is None:

        st.warning("No Cashier column was detected.")

    else:

        if sales_col is not None:

            cashier_summary = (
                filtered_df.groupby(
                    cashier_col,
                    dropna=False
                )
                .agg(
                    Total_Revenue=(sales_col, "sum"),
                    Average_Sale=(sales_col, "mean"),
                    Sales_Records=(sales_col, "count")
                )
                .reset_index()
            )

        else:

            cashier_summary = (
                filtered_df.groupby(
                    cashier_col,
                    dropna=False
                )
                .size()
                .reset_index(name="Sales_Records")
            )

        if transaction_col is not None:

            cashier_transactions = (
                filtered_df.groupby(
                    cashier_col,
                    dropna=False
                )[transaction_col]
                .nunique()
                .reset_index(name="Unique_Transactions")
            )

            cashier_summary = cashier_summary.merge(
                cashier_transactions,
                on=cashier_col,
                how="left"
            )

        st.dataframe(
            cashier_summary.sort_values(
                "Sales_Records",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These metrics describe transaction activity. "
            "They should not be used alone to judge employee performance."
        )

        if sales_col is not None:

            fig = px.bar(
                cashier_summary.sort_values(
                    "Total_Revenue",
                    ascending=False
                ).head(10),
                x=cashier_col,
                y="Total_Revenue",
                title="Revenue Associated with Each Cashier"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# TAB 5: DATA QUALITY
# ============================================================

with tabs[4]:

    st.header("Data Quality & Cleaning")

    st.write(
        "Review missing values, duplicate rows and column data types "
        "before performing further analysis."
    )

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    q2.metric(
        "Exact Duplicate Rows",
        f"{duplicate_count:,}"
    )

    q3.metric(
        "Missing Cells",
        f"{missing_total:,}"
    )

    st.subheader("Missing Values by Column")

    missing_report = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": df.isnull().sum().values,
        "Missing_Percentage": (
            df.isnull().mean().values * 100
        ).round(2),
        "Data_Type": [
            str(dtype) for dtype in df.dtypes
        ]
    })

    st.dataframe(
        missing_report,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Duplicate Row Handling")

    remove_duplicates = st.checkbox(
        "Remove exact duplicate rows from the downloadable cleaned data",
        value=False
    )

    cleaned_df = df.copy()

    if remove_duplicates:

        cleaned_df = cleaned_df.drop_duplicates()

        st.success(
            f"Removed {len(df) - len(cleaned_df):,} exact duplicate rows."
        )

    st.subheader("Data Type Information")

    dtype_report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype) for dtype in df.dtypes
        ],
        "Unique Values": [
            df[column].nunique(dropna=True)
            for column in df.columns
        ]
    })

    st.dataframe(
        dtype_report,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Cleaned CSV",
        data=cleaned_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="cleaned_retail_data.csv",
        mime="text/csv"
    )

    st.info(
        "Only exact duplicate rows are removed by the optional cleaning "
        "control. Missing values and unusual values are reported, "
        "not automatically deleted."
    )


# ============================================================
# TAB 6: ANOMALY DETECTION
# ============================================================

with tabs[5]:

    st.header("Transaction Anomaly Detection")

    st.write(
        "This section flags unusual records for investigation. "
        "A flagged record is not automatically an error or fraud."
    )

    if sales_col is None:

        st.warning(
            "A sales amount column or both Quantity and UnitPrice "
            "columns are required."
        )

    else:

        valid_sales = filtered_df[sales_col].dropna()

        if valid_sales.empty:

            st.info("No valid sales values available.")

        else:

            q1_value = valid_sales.quantile(0.25)
            q3_value = valid_sales.quantile(0.75)

            iqr = q3_value - q1_value

            lower_limit = q1_value - 1.5 * iqr
            upper_limit = q3_value + 1.5 * iqr

            unusual_sales = filtered_df[
                (filtered_df[sales_col] < lower_limit)
                | (filtered_df[sales_col] > upper_limit)
            ].copy()

            st.subheader("IQR-Based Outlier Detection")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Lower Threshold",
                format_number(lower_limit)
            )

            c2.metric(
                "Upper Threshold",
                format_number(upper_limit)
            )

            c3.metric(
                "Flagged Records",
                f"{len(unusual_sales):,}"
            )

            st.dataframe(
                unusual_sales,
                use_container_width=True,
                hide_index=True
            )

            if not unusual_sales.empty:

                st.download_button(
                    "⬇️ Download Flagged Transactions",
                    data=unusual_sales.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="flagged_transactions.csv",
                    mime="text/csv"
                )

            st.divider()

            st.subheader("Basic Invalid Value Checks")

            invalid_reports = []

            if quantity_col is not None:

                invalid_quantity_count = int(
                    (
                        filtered_df[quantity_col].isna()
                        | (filtered_df[quantity_col] <= 0)
                    ).sum()
                )

                invalid_reports.append({
                    "Check": "Missing or non-positive quantity",
                    "Flagged Records": invalid_quantity_count
                })

            if unit_price_col is not None:

                invalid_price_count = int(
                    (
                        filtered_df[unit_price_col].isna()
                        | (filtered_df[unit_price_col] < 0)
                    ).sum()
                )

                invalid_reports.append({
                    "Check": "Missing or negative unit price",
                    "Flagged Records": invalid_price_count
                })

            if sales_col is not None:

                negative_sales_count = int(
                    (filtered_df[sales_col] < 0).sum()
                )

                invalid_reports.append({
                    "Check": "Negative sales amount",
                    "Flagged Records": negative_sales_count
                })

            if invalid_reports:

                st.dataframe(
                    pd.DataFrame(invalid_reports),
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Zero values and negative values can sometimes be valid "
                "depending on refunds, returns and business rules. "
                "Review records before correcting them."
            )


# ============================================================
# TAB 7: SQL QUERIES
# ============================================================

with tabs[6]:

    st.header("SQL Analysis")

    st.write(
        "The following examples demonstrate the SQL queries "
        "you can use for retail analysis."
    )

    st.warning(
        "These queries are examples for a database table named "
        "`transactions`. They are displayed for reference and are "
        "not executed against a live SQL database."
    )

    sql_examples = {
        "Total Sales by Store": """
SELECT
    StoreID,
    SUM(TotalPrice) AS TotalRevenue
FROM transactions
GROUP BY StoreID
ORDER BY TotalRevenue DESC;
""",

        "Product Performance": """
SELECT
    Product,
    SUM(Quantity) AS TotalUnitsSold,
    SUM(TotalPrice) AS TotalRevenue
FROM transactions
GROUP BY Product
ORDER BY TotalRevenue DESC;
""",

        "Payment Type Analysis": """
SELECT
    PaymentType,
    COUNT(*) AS TransactionCount,
    SUM(TotalPrice) AS TotalRevenue
FROM transactions
GROUP BY PaymentType
ORDER BY TotalRevenue DESC;
""",

        "Cashier Analysis": """
SELECT
    Cashier,
    COUNT(DISTINCT TransactionID) AS TotalTransactions,
    SUM(TotalPrice) AS TotalRevenue
FROM transactions
GROUP BY Cashier
ORDER BY TotalRevenue DESC;
""",

        "Average Sale by Store": """
SELECT
    StoreID,
    AVG(TotalPrice) AS AverageSale
FROM transactions
GROUP BY StoreID
ORDER BY AverageSale DESC;
""",

        "Day-wise Sales": """
SELECT
    DayOfWeek,
    SUM(TotalPrice) AS TotalRevenue
FROM transactions
GROUP BY DayOfWeek
ORDER BY TotalRevenue DESC;
"""
    }

    selected_query = st.selectbox(
        "Select an SQL analysis",
        list(sql_examples.keys())
    )

    st.code(
        sql_examples[selected_query],
        language="sql"
    )

    st.info(
        "Check your actual database column names before executing "
        "these queries in MySQL Workbench. If your database uses "
        "different names, update the queries accordingly."
    )


# ============================================================
# TAB 8: DATA EXPLORER
# ============================================================

with tabs[7]:

    st.header("Data Explorer")

    st.write(
        "Explore the filtered transaction records and download "
        "the current selection."
    )

    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} total records."
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Filtered Dataset",
        data=filtered_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="filtered_retail_data.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b;">
        <p><b>Retail Analytics & Intelligence System</b></p>
        <p>Developed by Chiranjit Sahu | B.Tech CSE (AIML)</p>
        <p>Technologies: Python, Pandas, Streamlit, Plotly and SQL</p>
    </div>
    """,
    unsafe_allow_html=True
)