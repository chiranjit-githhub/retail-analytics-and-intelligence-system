import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import date

# NEW: AI extensions (forecasting, recommendations, natural-language queries)
from retail_ai import (
    render_forecast_tab,
    render_recommendation_tab,
    render_nlq_tab,
)


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
# THEME (LIGHT / DARK TOGGLE)
# ============================================================

toggle_widget = getattr(st.sidebar, "toggle", st.sidebar.checkbox)

dark_mode = toggle_widget(
    "🌙 Dark mode",
    value=False,
    key="dark_mode",
    help="Switch between light and dark appearance."
)

presentation_mode = toggle_widget(
    "🖥️ Presentation mode",
    value=True,
    key="presentation_mode",
    help="Fits the Overview dashboard on one screen so you do not need to scroll."
)

dashboard_height = 235

if presentation_mode:
    dashboard_height = st.sidebar.slider(
        "Dashboard chart height (px)",
        min_value=170, max_value=340, value=235, step=5,
        key="dashboard_height",
        help="Lower this on small screens. Browser zoom of 80% also helps."
    )

THEMES = {
    "light": {
        "bg": "#f3f5fb", "sidebar": "#ffffff", "card": "#ffffff",
        "input": "#ffffff", "border": "#e3e8f2", "text": "#0f172a",
        "muted": "#64748b", "accent": "#6366f1",
        "shadow": "0 2px 10px rgba(15,23,42,0.06)",
        "df_filter": "none",
    },
    "dark": {
        "bg": "#0b1020", "sidebar": "#0e1428", "card": "#151c33",
        "input": "#1b2340", "border": "#27314f", "text": "#e6e9f2",
        "muted": "#95a0ba", "accent": "#818cf8",
        "shadow": "0 4px 18px rgba(0,0,0,0.35)",
        "df_filter": "invert(0.92) hue-rotate(180deg)",
    },
}

theme = THEMES["dark" if dark_mode else "light"]

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: %(bg)s; --sidebar: %(sidebar)s; --card: %(card)s;
    --input: %(input)s; --border: %(border)s; --text: %(text)s;
    --muted: %(muted)s; --accent: %(accent)s; --shadow: %(shadow)s;
    --df-filter: %(df_filter)s;
}

html, body, .stApp, .stApp button, .stApp input, .stApp textarea {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1400px; }

/* ---------- Text ---------- */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp label, .stApp li, .stApp .stMarkdown {
    color: var(--text);
}
.stApp h1, .stApp h2, .stApp h3 { font-weight: 700; letter-spacing: -0.01em; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}
hr { border-color: var(--border) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
    color: var(--text);
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--input);
    border: 1px dashed var(--border);
    border-radius: 12px;
}

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #06b6d4 100%);
    border-radius: 20px;
    padding: 30px 34px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.30);
}
.hero, .hero * { color: #ffffff !important; }
.hero-title { font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; line-height: 1.2; }
.hero-sub { font-size: 1rem; opacity: 0.92; margin: 0 0 16px 0; }
.chip {
    display: inline-block; padding: 5px 12px; margin: 0 8px 6px 0;
    border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.30);
}

/* ---------- Feature cards (welcome screen) ---------- */
.feature-grid {
    display: grid; gap: 16px; margin-top: 8px;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}
.feature-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 20px; box-shadow: var(--shadow);
    transition: transform .15s ease, box-shadow .15s ease;
}
.feature-card:hover { transform: translateY(-3px); }
.feature-icon {
    width: 44px; height: 44px; border-radius: 12px; font-size: 22px;
    display: flex; align-items: center; justify-content: center; margin-bottom: 12px;
    background: linear-gradient(135deg, rgba(99,102,241,.18), rgba(6,182,212,.18));
}
.feature-title { font-weight: 650; font-size: 1.05rem; margin-bottom: 4px; color: var(--text); }
.feature-text { color: var(--muted); font-size: 0.92rem; line-height: 1.45; }

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {
    background: var(--card); border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    padding: 16px 18px; border-radius: 14px; box-shadow: var(--shadow);
    transition: transform .15s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-2px); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    color: var(--muted) !important; font-weight: 500; font-size: 0.85rem;
}
[data-testid="stMetricValue"] { color: var(--text); font-weight: 700; }

/* ---------- Charts and tables ---------- */
div[data-testid="stPlotlyChart"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 8px; box-shadow: var(--shadow);
}
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
    filter: var(--df-filter);
}
[data-testid="stCode"] { border-radius: 12px; }

/* ---------- Tabs as pills ---------- */
div[data-baseweb="tab-list"] { gap: 6px; border-bottom: none; flex-wrap: wrap; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }
button[data-baseweb="tab"] {
    border-radius: 10px; padding: 8px 14px; height: auto;
    background: var(--card); border: 1px solid var(--border);
}
button[data-baseweb="tab"] p { color: var(--muted); font-weight: 600; font-size: 0.9rem; }
button[data-baseweb="tab"]:hover { border-color: var(--accent); }
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6); border-color: transparent;
}
button[data-baseweb="tab"][aria-selected="true"] p { color: #ffffff !important; }

/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
div[data-baseweb="base-input"], .stTextInput input, .stDateInput input {
    background: var(--input) !important; color: var(--text) !important;
    border-color: var(--border) !important; border-radius: 10px;
}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] div[role="listbox"],
div[data-baseweb="calendar"] {
    background: var(--card) !important; color: var(--text) !important;
}
li[role="option"] { color: var(--text) !important; }
span[data-baseweb="tag"] { background: var(--accent) !important; border-radius: 8px; }
span[data-baseweb="tag"], span[data-baseweb="tag"] span { color: #ffffff !important; }

/* ---------- Buttons, expanders ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; border: 1px solid var(--border);
    background: var(--card); color: var(--text); font-weight: 600;
    box-shadow: var(--shadow); transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent); color: var(--accent);
}
[data-testid="stExpander"] {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
}
[data-testid="stAlert"] { border-radius: 12px; }

/* ---------- Footer ---------- */
.app-footer {
    text-align: center; color: var(--muted); font-size: 0.85rem;
    padding: 14px 0 6px 0; line-height: 1.6;
}
.app-footer b { color: var(--text); }
"""

DARK_EXTRA_CSS = """
[data-testid="stAlert"] {
    background: rgba(148, 163, 184, 0.12) !important;
    border: 1px solid var(--border);
}
[data-testid="stAlert"] * { color: var(--text) !important; }
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"])    { border-left: 4px solid #3b82f6; }
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { border-left: 4px solid #f59e0b; }
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { border-left: 4px solid #10b981; }
[data-testid="stAlert"]:has([data-testid="stAlertContentError"])   { border-left: 4px solid #ef4444; }
[data-testid="stSidebar"] button { background: var(--card); border: 1px solid var(--border); }
"""

PRESENTATION_CSS = """
.block-container { padding-top: 2.4rem !important; padding-bottom: 0.5rem !important; }
div[data-testid="stVerticalBlock"] { gap: 0.55rem; }
.hero.slim { padding: 10px 20px; margin-bottom: 6px; border-radius: 14px; }
.hero.slim .hero-title { font-size: 1.2rem; margin: 0; }
button[data-baseweb="tab"] { padding: 5px 12px; }
div[data-baseweb="tab-list"] { margin-bottom: 2px; }
div[data-testid="stPlotlyChart"] { padding: 3px; border-radius: 12px; }
"""

theme_css = BASE_CSS

for key, value in theme.items():
    theme_css = theme_css.replace(f"%({key})s", value)

st.markdown(
    "<style>" + theme_css + (DARK_EXTRA_CSS if dark_mode else "")
    + (PRESENTATION_CSS if presentation_mode else "") + "</style>",
    unsafe_allow_html=True
)


# Plotly template that follows the selected theme.

PALETTE = [
    "#6366f1", "#06b6d4", "#f59e0b", "#10b981",
    "#ef4444", "#a855f7", "#14b8a6", "#f97316"
]

chart_template = go.layout.Template(
    pio.templates["plotly_dark" if dark_mode else "plotly_white"]
)

chart_template.layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Inter, system-ui, sans-serif",
        color=theme["text"]
    ),
    colorway=PALETTE,
    margin=dict(l=30, r=20, t=60, b=30),
    title=dict(font=dict(size=16))
)

pio.templates["retail_theme"] = chart_template
pio.templates.default = "retail_theme"

px.defaults.template = "retail_theme"
px.defaults.color_continuous_scale = px.colors.sequential.Purp


# ============================================================
# HEADER
# ============================================================

if presentation_mode:

    st.markdown(
        '<div class="hero slim">'
        '<div class="hero-title">📊 Retail Analytics &amp; Intelligence System</div>'
        '</div>',
        unsafe_allow_html=True
    )

else:

    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">📊 Retail Analytics &amp; Intelligence System</div>'
        '<div class="hero-sub">Explore sales, forecast demand, get recommendations '
        'and ask questions about your data in plain English.</div>'
        '<span class="chip">📈 Live filters</span>'
        '<span class="chip">🔮 Forecasting</span>'
        '<span class="chip">💡 Recommendations</span>'
        '<span class="chip">💬 Ask your data</span>'
        '</div>',
        unsafe_allow_html=True
    )


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


def clean_dataframe(data):
    """
    Repair common CSV problems:
    - merge duplicated columns (Date, Date.1 ...) keeping any real values
    - drop columns that are completely empty
    Returns the cleaned data and a list of human-readable notes.
    """

    notes = []
    data = data.copy()

    # Merge pandas-renamed duplicate columns such as "Date.1" into "Date".
    for column in list(data.columns):
        text = str(column)
        base, dot, suffix = text.rpartition(".")

        if dot and suffix.isdigit() and base in data.columns:
            data[base] = data[base].combine_first(data[column])
            data = data.drop(columns=[column])
            notes.append(f"Merged duplicate column '{text}' into '{base}'.")

    # Drop columns with no values at all (e.g. broken headers).
    empty_columns = [
        column for column in data.columns
        if data[column].isna().all()
    ]

    if empty_columns:
        data = data.drop(columns=empty_columns)
        notes.append(
            "Dropped empty columns: "
            + ", ".join(str(column) for column in empty_columns)
        )

    return data, notes


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

    features = [
        ("📈", "Sales Analysis",
         "Explore revenue, transactions, sales trends and store performance."),
        ("🧹", "Data Quality",
         "Inspect missing values, duplicates, data types and unusual values."),
        ("🔎", "Business Insights",
         "Compare products, payment methods, locations and cashier activity."),
        ("🔮", "Forecasting",
         "Predict future daily revenue with a backtested model and confidence range."),
        ("💡", "Recommendations",
         "Get automated insights and cross-sell suggestions from basket analysis."),
        ("💬", "Ask Your Data",
         "Ask questions in plain English, optionally using a local Ollama model."),
    ]

    cards_html = "".join(
        '<div class="feature-card">'
        f'<div class="feature-icon">{icon}</div>'
        f'<div class="feature-title">{title}</div>'
        f'<div class="feature-text">{text}</div>'
        '</div>'
        for icon, title, text in features
    )

    st.markdown(
        f'<div class="feature-grid">{cards_html}</div>',
        unsafe_allow_html=True
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

df, cleaning_notes = clean_dataframe(df_original)

if cleaning_notes:
    with st.sidebar.expander("🧹 Automatic data cleaning applied"):
        for note in cleaning_notes:
            st.write("• " + note)


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

if total_price_col is not None and df[total_price_col].isna().all():
    # TotalPrice exists but holds no usable values: calculate it instead.
    total_price_col = None

if total_price_col is not None:
    sales_col = total_price_col

elif quantity_col is not None and unit_price_col is not None:
    df["_CalculatedTotalPrice"] = (
        df[quantity_col] * df[unit_price_col]
    )
    sales_col = "_CalculatedTotalPrice"

else:
    sales_col = None


# NEW: column map used by the AI extension tabs.

ai_cols = dict(
    date=date_col,
    store=store_col,
    location=location_col,
    product=product_col,
    payment=payment_col,
    cashier=cashier_col,
    time_of_day=time_of_day_col,
    day_of_week=day_of_week_col,
    transaction=transaction_col,
    quantity=quantity_col,
    sales=sales_col,
)


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
        "📋 Data Explorer",
        "🔮 Forecast",
        "💡 Recommendations",
        "💬 Ask Your Data"
    ]
)


# ============================================================
# TAB 1: OVERVIEW (DASHBOARD)
# ============================================================

OV_PAL = [
    "#14b8a6", "#f97316", "#8b5cf6", "#ec4899",
    "#eab308", "#3b82f6", "#22c55e", "#ef4444"
]
OV_SUNSET = [[0.0, "#fde68a"], [0.5, "#fb923c"], [1.0, "#be123c"]]
OV_OCEAN = [[0.0, "#ccfbf1"], [0.5, "#2dd4bf"], [1.0, "#0f766e"]]
GRID = "rgba(148, 163, 184, 0.18)"

WEEK = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]
SLOT_ORDER = ["Morning", "Afternoon", "Evening", "Night"]

OVERVIEW_CSS = """
<style>
.section-title { font-size: 1.05rem; font-weight: 650; margin: 20px 0 8px 0; color: var(--text); }
.kpi-grid { display: grid; gap: 14px; margin: 6px 0 14px 0;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
.kpi-card { position: relative; display: flex; gap: 14px; align-items: center;
            background: var(--card); border: 1px solid var(--border); border-radius: 16px;
            padding: 16px 18px; box-shadow: var(--shadow); overflow: hidden;
            transition: transform .15s ease; }
.kpi-card:hover { transform: translateY(-3px); }
.kpi-card::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0;
                    width: 5px; background: var(--c); }
.kpi-icon { width: 46px; height: 46px; border-radius: 14px; display: flex;
            align-items: center; justify-content: center; font-size: 22px;
            background: var(--input); background: color-mix(in srgb, var(--c) 20%, transparent); }
.kpi-label { font-size: .75rem; color: var(--muted); font-weight: 600;
             text-transform: uppercase; letter-spacing: .05em; }
.kpi-value { font-size: 1.6rem; font-weight: 750; color: var(--text); line-height: 1.2; }
.kpi-delta { display: inline-block; margin-top: 3px; padding: 2px 8px; border-radius: 999px;
             font-size: .75rem; font-weight: 700; }
.kpi-delta.up { background: rgba(34,197,94,.15); color: #16a34a; }
.kpi-delta.down { background: rgba(239,68,68,.15); color: #dc2626; }
.kpi-grid.compact { gap: 10px; margin: 2px 0 6px 0;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.kpi-grid.compact .kpi-card { padding: 9px 12px; gap: 10px; border-radius: 12px; }
.kpi-grid.compact .kpi-icon { width: 34px; height: 34px; font-size: 17px; border-radius: 10px; }
.kpi-grid.compact .kpi-label { font-size: .64rem; }
.kpi-grid.compact .kpi-value, .kpi-grid.compact .kpi-delta { display: inline-block; vertical-align: middle; }
.kpi-grid.compact .kpi-value { font-size: 1.2rem; }
.kpi-grid.compact .kpi-delta { margin: 0 0 0 6px; font-size: .66rem; padding: 1px 6px; }
.hl-grid { display: grid; gap: 12px; margin: 4px 0 6px 0;
           grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
.hl-card { border: 1px solid var(--border); border-radius: 14px; padding: 12px 16px;
           background: var(--card);
           background: linear-gradient(135deg, color-mix(in srgb, var(--c) 16%, var(--card)), var(--card)); }
.hl-label { font-size: .75rem; color: var(--muted); font-weight: 600; }
.hl-value { font-size: 1.02rem; font-weight: 700; color: var(--text); }
</style>
"""


def compact(value):
    """Short number format: 1.2K, 3.4M, 5.6B."""
    magnitude = abs(value)
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= divisor:
            return f"{value / divisor:.2f}{suffix}"
    return f"{value:,.0f}" if magnitude >= 100 else f"{value:,.2f}"


def kpi_card(icon, label, value, color, delta=None):
    chip = ""
    if delta is not None and not pd.isna(delta):
        direction = "up" if delta >= 0 else "down"
        arrow = "▲" if delta >= 0 else "▼"
        chip = f'<span class="kpi-delta {direction}">{arrow} {abs(delta):.1f}%</span>'

    return (
        f'<div class="kpi-card" style="--c:{color}">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{chip}</div>'
        f'</div>'
    )


def highlight_card(icon, label, value, color):
    return (
        f'<div class="hl-card" style="--c:{color}">'
        f'<div class="hl-label">{icon} {label}</div>'
        f'<div class="hl-value">{value}</div>'
        f'</div>'
    )


def ov_style(fig, title, height=360):
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        xaxis_title=None,
        yaxis_title=None,
    )
    fig.update_yaxes(gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    return fig


def pct_change(new, old):
    return (new - old) / old * 100 if old else None


def weekday_series(data):
    if day_of_week_col is not None:
        s = data[day_of_week_col].astype(str).str.strip().str.title()
        s = s.where(data[day_of_week_col].notna())
        if s.isin(WEEK).mean() > 0.8:
            return s
    if date_col is not None:
        return data[date_col].dt.day_name()
    return None


def slot_series(data):
    """Return (series, label, ordered categories) for time of day or hour."""
    if time_of_day_col is not None:
        s = data[time_of_day_col].astype(str).str.strip().str.title()
        s = s.where(data[time_of_day_col].notna())
        present = set(s.dropna())
        order = [x for x in SLOT_ORDER if x in present] + sorted(present - set(SLOT_ORDER))
        return s, "Time of day", order

    if time_col is not None:
        parsed = pd.to_datetime(
            data[time_col].astype(str), errors="coerce"
        )
        if parsed.notna().mean() > 0.5:
            hour = parsed.dt.hour
            return hour, "Hour", sorted(hour.dropna().unique().tolist())

    return None, None, []


def render_overview():

    st.markdown(OVERVIEW_CSS, unsafe_allow_html=True)

    if not presentation_mode:

        st.header("Retail Performance Overview")

        st.caption(
            "Every card and chart reflects the filters selected in the sidebar."
        )

    # ---------- Period comparison for KPI deltas ----------

    deltas = {}

    if date_col is not None and sales_col is not None:

        last_date = filtered_df[date_col].max()

        if pd.notna(last_date):

            recent_df = filtered_df[
                filtered_df[date_col] > last_date - pd.Timedelta(days=30)
            ]
            prior_df = filtered_df[
                (filtered_df[date_col] <= last_date - pd.Timedelta(days=30))
                & (filtered_df[date_col] > last_date - pd.Timedelta(days=60))
            ]

            if not recent_df.empty and not prior_df.empty:

                def period_numbers(frame):
                    revenue = float(frame[sales_col].sum())
                    txn = (
                        frame[transaction_col].nunique()
                        if transaction_col is not None else len(frame)
                    )
                    units = (
                        float(frame[quantity_col].sum())
                        if quantity_col is not None else 0.0
                    )
                    return revenue, txn, units

                r_rev, r_txn, r_units = period_numbers(recent_df)
                p_rev, p_txn, p_units = period_numbers(prior_df)

                deltas = {
                    "revenue": pct_change(r_rev, p_rev),
                    "transactions": pct_change(r_txn, p_txn),
                    "units": pct_change(r_units, p_units),
                    "avg": pct_change(
                        r_rev / r_txn if r_txn else 0,
                        p_rev / p_txn if p_txn else 0
                    ),
                }

    # ---------- KPI cards ----------

    cards = [
        kpi_card("💰", "Total Revenue", compact(total_revenue),
                 OV_PAL[0], deltas.get("revenue")),
        kpi_card("🧾", "Transactions", f"{total_transactions:,}",
                 OV_PAL[2], deltas.get("transactions")),
    ]

    if quantity_col is not None:
        cards.append(
            kpi_card("📦", "Units Sold", compact(total_units),
                     OV_PAL[1], deltas.get("units"))
        )

    cards.append(
        kpi_card("🛒", "Avg Transaction Value",
                 compact(average_transaction_value),
                 OV_PAL[3], deltas.get("avg"))
    )

    if date_col is not None and sales_col is not None:
        active_days = int(filtered_df[date_col].dt.normalize().nunique())
        if active_days > 0:
            cards.append(
                kpi_card("📅", "Revenue per Active Day",
                         compact(total_revenue / active_days), OV_PAL[5])
            )

    if quantity_col is not None and total_units > 0:
        cards.append(
            kpi_card("🏷️", "Avg Price per Unit",
                     compact(total_revenue / total_units), OV_PAL[6])
        )

    grid_class = "kpi-grid compact" if presentation_mode else "kpi-grid"

    st.markdown(
        f'<div class="{grid_class}">' + "".join(cards) + "</div>",
        unsafe_allow_html=True
    )

    if deltas and not presentation_mode:
        st.caption(
            "Arrows compare the latest 30 days with the 30 days before them."
        )

    if sales_col is None:
        st.info(
            "A sales amount (TotalPrice, or Quantity and UnitPrice) is "
            "needed to draw the dashboard charts."
        )
        return

    # ---------- Highlights ----------

    highlights = []

    if store_col is not None:
        by_store = filtered_df.groupby(store_col)[sales_col].sum()
        if not by_store.empty and by_store.sum() > 0:
            highlights.append(highlight_card(
                "🏆", "Top store",
                f"{by_store.idxmax()} · {by_store.max() / by_store.sum():.0%} of revenue",
                OV_PAL[0]))

    if product_col is not None:
        by_product = filtered_df.groupby(product_col)[sales_col].sum()
        if not by_product.empty and by_product.sum() > 0:
            highlights.append(highlight_card(
                "⭐", "Top product",
                f"{by_product.idxmax()} · {by_product.max() / by_product.sum():.0%} of revenue",
                OV_PAL[1]))

    weekday = weekday_series(filtered_df)

    if weekday is not None:
        by_day = filtered_df.groupby(weekday)[sales_col].sum()
        if not by_day.empty:
            highlights.append(highlight_card(
                "📆", "Best day", f"{by_day.idxmax()}", OV_PAL[2]))

    slot, slot_label, slot_order = slot_series(filtered_df)

    if slot is not None:
        by_slot = filtered_df.groupby(slot)[sales_col].sum()
        if not by_slot.empty:
            highlights.append(highlight_card(
                "⏰", f"Peak {slot_label.lower()}", f"{by_slot.idxmax()}",
                OV_PAL[3]))

    chart_map = {}

    # 1. Revenue by store
    if store_col is not None:

        store_rev = (
            filtered_df.groupby(store_col)[sales_col].sum()
            .sort_values(ascending=False).reset_index()
        )
        store_rev[store_col] = store_rev[store_col].astype(str)

        fig = px.bar(
            store_rev, x=store_col, y=sales_col, color=store_col,
            color_discrete_sequence=OV_PAL, text_auto=".2s"
        )
        fig.update_layout(showlegend=False)
        fig.update_traces(textposition="outside", cliponaxis=False)
        chart_map["store"] = (ov_style(fig, "Revenue by store"))

    # 2. Payment donut
    if payment_col is not None:

        pay = filtered_df.groupby(payment_col)[sales_col].sum().reset_index()

        fig = px.pie(
            pay, names=payment_col, values=sales_col, hole=0.62,
            color_discrete_sequence=OV_PAL
        )
        fig.update_traces(textinfo="percent", textposition="outside")
        fig.add_annotation(
            text=f"<b>{compact(pay[sales_col].sum())}</b><br>revenue",
            showarrow=False, font=dict(size=15)
        )
        chart_map["payment"] = (ov_style(fig, "Revenue by payment type"))

    # 3. Top products
    if product_col is not None:

        top_products = (
            filtered_df.groupby(product_col)[sales_col].sum()
            .nlargest(10).reset_index()
        )
        top_products[product_col] = top_products[product_col].astype(str)

        fig = px.bar(
            top_products, x=sales_col, y=product_col, orientation="h",
            color=sales_col, color_continuous_scale=OV_SUNSET,
            text_auto=".2s"
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False
        )
        chart_map["products"] = (ov_style(fig, "Top 10 products by revenue"))

        # 4. Pareto
        pareto = (
            filtered_df.groupby(product_col)[sales_col].sum()
            .sort_values(ascending=False)
        )

        if len(pareto) > 3 and pareto.sum() > 0:

            cumulative = pareto.cumsum() / pareto.sum() * 100
            top_n = pareto.head(15)
            cum_n = cumulative.head(15)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=top_n.index.astype(str), y=top_n.values,
                       name="Revenue", marker_color=OV_PAL[2]),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=cum_n.index.astype(str), y=cum_n.values,
                           name="Cumulative %", mode="lines+markers",
                           line=dict(color=OV_PAL[3], width=2.5)),
                secondary_y=True
            )
            fig.add_hline(
                y=80, line_dash="dot", line_color=OV_PAL[4],
                secondary_y=True
            )
            fig.update_yaxes(range=[0, 105], ticksuffix="%", secondary_y=True,
                             showgrid=False)
            chart_map["pareto"] = (ov_style(fig, "Pareto: how concentrated is revenue?"))

    # 5. Revenue by weekday
    if weekday is not None:

        day_rev = (
            filtered_df.groupby(weekday)[sales_col].sum()
        )
        days_present = [d for d in WEEK if d in day_rev.index]
        day_rev = day_rev.reindex(days_present) if days_present else day_rev

        if not day_rev.empty:

            colors = [
                OV_PAL[1] if v == day_rev.max() else "#94a3b8"
                for v in day_rev.values
            ]

            fig = go.Figure(go.Bar(
                x=day_rev.index, y=day_rev.values, marker_color=colors,
                text=[compact(v) for v in day_rev.values],
                textposition="outside", cliponaxis=False
            ))
            fig.add_hline(
                y=day_rev.mean(), line_dash="dot", line_color=OV_PAL[0],
                annotation_text="average", annotation_position="top left"
            )
            chart_map["weekday"] = (ov_style(fig, "Revenue by day of week"))

    # 6. Revenue by time of day / hour
    if slot is not None:

        slot_rev = filtered_df.groupby(slot)[sales_col].sum()
        slot_rev = slot_rev.reindex(slot_order).dropna()

        if not slot_rev.empty:

            fig = go.Figure(go.Bar(
                x=[str(i) for i in slot_rev.index], y=slot_rev.values,
                marker_color=[OV_PAL[i % len(OV_PAL)]
                              for i in range(len(slot_rev))],
                text=[compact(v) for v in slot_rev.values],
                textposition="outside", cliponaxis=False
            ))
            chart_map["slot"] = (ov_style(fig, f"Revenue by {slot_label.lower()}"))

    # 7. Heatmap weekday x slot
    if weekday is not None and slot is not None:

        heat = pd.DataFrame({
            "Day": weekday, "Slot": slot, "Revenue": filtered_df[sales_col]
        }).dropna()

        if not heat.empty:

            pivot = heat.pivot_table(
                index="Day", columns="Slot", values="Revenue", aggfunc="sum"
            )
            rows = [d for d in WEEK if d in pivot.index] or list(pivot.index)
            cols_order = [c for c in slot_order if c in pivot.columns]
            pivot = pivot.reindex(index=rows, columns=cols_order)
            pivot.columns = [str(c) for c in pivot.columns]

            fig = px.imshow(
                pivot, color_continuous_scale=OV_SUNSET, aspect="auto",
                text_auto=".2s"
            )
            fig.update_layout(coloraxis_showscale=False)
            chart_map["heatmap"] = (ov_style(
                fig, f"When do customers buy? (day × {slot_label.lower()})"))

    # 8. Sale value distribution
    values = filtered_df[sales_col].dropna()

    if len(values) > 10:

        shown = values[values <= values.quantile(0.99)]

        fig = px.histogram(
            x=shown.values, nbins=40, marginal="box",
            color_discrete_sequence=[OV_PAL[5]]
        )
        fig.update_layout(showlegend=False, bargap=0.05)
        chart_map["dist"] = (ov_style(
            fig, "Distribution of sale values (top 1% hidden)"))

    # 9. Location treemap or store x payment
    if location_col is not None:

        path = [location_col]
        if store_col is not None and store_col != location_col:
            path.append(store_col)

        tree = filtered_df.dropna(subset=path).copy()
        for p in path:
            tree[p] = tree[p].astype(str)
        tree = tree.groupby(path, as_index=False)[sales_col].sum()
        tree = tree[tree[sales_col] > 0]

        if not tree.empty:
            fig = px.treemap(
                tree, path=[px.Constant("All")] + path, values=sales_col,
                color_discrete_sequence=OV_PAL
            )
            fig.update_traces(textinfo="label+percent root")
            chart_map["geo"] = (ov_style(fig, "Revenue by location and store"))

    elif store_col is not None and payment_col is not None:

        mix = (
            filtered_df.groupby([store_col, payment_col])[sales_col]
            .sum().reset_index()
        )
        mix[store_col] = mix[store_col].astype(str)

        fig = px.bar(
            mix, x=store_col, y=sales_col, color=payment_col,
            barmode="stack", color_discrete_sequence=OV_PAL
        )
        chart_map["geo"] = (ov_style(fig, "Payment mix by store"))

    # 10. Product bubble chart
    if product_col is not None:

        agg = {"Revenue": (sales_col, "sum")}
        agg["Transactions"] = (
            (transaction_col, "nunique") if transaction_col is not None
            else (sales_col, "count")
        )
        if quantity_col is not None:
            agg["Units"] = (quantity_col, "sum")

        bubble = (
            filtered_df.groupby(product_col).agg(**agg)
            .nlargest(25, "Revenue").reset_index()
        )
        bubble[product_col] = bubble[product_col].astype(str)
        x_field = "Units" if "Units" in bubble.columns else "Transactions"

        if len(bubble) > 2:
            fig = px.scatter(
                bubble, x=x_field, y="Revenue", size="Transactions",
                color="Revenue", hover_name=product_col,
                color_continuous_scale=OV_OCEAN, size_max=38
            )
            fig.update_layout(coloraxis_showscale=False)
            chart_map["bubble"] = (ov_style(
                fig, f"Product performance: {x_field.lower()} vs revenue"))

    # ---------- One-screen dashboard (presentation mode) ----------

    shown_keys = set()

    def compact_fig(fig, height):
        small = go.Figure(fig)
        small.update_layout(
            height=height,
            margin=dict(l=8, r=8, t=34, b=8),
            font=dict(size=11),
            title=dict(font=dict(size=13)),
            legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        )
        if any(trace.type == "pie" for trace in small.data):
            small.update_traces(textposition="inside", textinfo="percent")
            small.update_layout(
                legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=10))
            )
            small.update_layout(
                annotations=[
                    dict(note.to_plotly_json(), font=dict(size=12))
                    for note in small.layout.annotations
                ]
            )
        return small

    def auto_trend_figure():
        if date_col is None:
            return None

        dated = filtered_df.dropna(subset=[date_col])
        if dated.empty:
            return None

        span = (dated[date_col].max() - dated[date_col].min()).days

        if span <= 60:
            grain_name, key = "daily", dated[date_col].dt.normalize()
        elif span <= 400:
            grain_name = "weekly"
            key = dated[date_col].dt.to_period("W").dt.start_time
        else:
            grain_name = "monthly"
            key = dated[date_col].dt.to_period("M").dt.start_time

        series = dated.groupby(key)[sales_col].sum().sort_index()

        if len(series) < 2:
            return None

        fig = go.Figure(go.Scatter(
            x=series.index, y=series.values, mode="lines", fill="tozeroy",
            line=dict(color=OV_PAL[0], width=2.5),
            fillcolor="rgba(20, 184, 166, 0.18)", name="Revenue"
        ))
        fig.add_trace(go.Scatter(
            x=[series.idxmax()], y=[series.max()], mode="markers+text",
            marker=dict(color=OV_PAL[1], size=9),
            text=[f"Peak {compact(series.max())}"], textposition="top center",
            showlegend=False
        ))
        fig.update_layout(showlegend=False, hovermode="x unified")
        return ov_style(fig, f"Revenue trend ({grain_name})")

    plot_config = {"displayModeBar": False}

    if presentation_mode:

        trend_fig = auto_trend_figure()

        pool_keys = [
            k for k in ("payment", "store", "products", "weekday",
                        "slot", "heatmap", "geo")
            if k in chart_map
        ]

        row_one = []
        if trend_fig is not None:
            row_one.append(compact_fig(trend_fig, dashboard_height))
        take = 1 if row_one else 2
        for k in pool_keys[:take]:
            row_one.append(compact_fig(chart_map[k], dashboard_height))
            shown_keys.add(k)

        row_two_keys = [k for k in pool_keys if k not in shown_keys][:3]
        row_two = [compact_fig(chart_map[k], dashboard_height) for k in row_two_keys]
        shown_keys.update(row_two_keys)

        widths = [2, 1] if (len(row_one) == 2 and trend_fig is not None) \
            else [1] * max(len(row_one), 1)

        if row_one:
            for column, figure in zip(st.columns(widths), row_one):
                column.plotly_chart(figure, use_container_width=True,
                                    config=plot_config)

        if row_two:
            for column, figure in zip(st.columns(len(row_two)), row_two):
                column.plotly_chart(figure, use_container_width=True,
                                    config=plot_config)

        st.caption(
            "Scroll down for highlights, an interactive trend and more charts. "
            "Arrows on the cards compare the latest 30 days with the 30 days before."
        )

    # ---------- Highlights ----------

    if highlights:
        st.markdown(
            '<div class="hl-grid">' + "".join(highlights) + "</div>",
            unsafe_allow_html=True
        )

    # ---------- Trend (full width) ----------

    st.markdown('<div class="section-title">📈 Trend over time</div>',
                unsafe_allow_html=True)

    if date_col is not None:

        metric_options = ["Revenue", "Transactions"]
        if quantity_col is not None:
            metric_options.append("Units")

        control_a, control_b, _ = st.columns([1.2, 1.2, 2])

        grain = control_a.radio(
            "Group by", ["Daily", "Weekly", "Monthly"],
            horizontal=True, key="ov_grain"
        )
        metric_name = control_b.radio(
            "Measure", metric_options,
            horizontal=True, key="ov_metric"
        )

        dated = filtered_df.dropna(subset=[date_col]).copy()

        if grain == "Daily":
            dated["_Period"] = dated[date_col].dt.normalize()
            window = 7
        elif grain == "Weekly":
            dated["_Period"] = dated[date_col].dt.to_period("W").dt.start_time
            window = 4
        else:
            dated["_Period"] = dated[date_col].dt.to_period("M").dt.start_time
            window = 3

        if metric_name == "Revenue":
            trend = dated.groupby("_Period")[sales_col].sum()
        elif metric_name == "Transactions":
            trend = (
                dated.groupby("_Period")[transaction_col].nunique()
                if transaction_col is not None
                else dated.groupby("_Period").size()
            )
        else:
            trend = dated.groupby("_Period")[quantity_col].sum()

        trend = trend.sort_index()

        if len(trend) > 1:

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=trend.index, y=trend.values, name=metric_name,
                mode="lines", fill="tozeroy",
                line=dict(color=OV_PAL[0], width=2.5),
                fillcolor="rgba(20, 184, 166, 0.18)"
            ))

            fig.add_trace(go.Scatter(
                x=trend.index,
                y=trend.rolling(window, min_periods=1).mean().values,
                name=f"{window}-period average",
                mode="lines",
                line=dict(color=OV_PAL[1], width=2, dash="dash")
            ))

            peak_x = trend.idxmax()

            fig.add_annotation(
                x=peak_x, y=trend.max(),
                text=f"Peak {compact(trend.max())}",
                showarrow=True, arrowhead=2, ay=-40,
                font=dict(color=OV_PAL[1])
            )

            fig.update_layout(hovermode="x unified")

            st.plotly_chart(
                ov_style(fig, f"{grain} {metric_name.lower()}", height=400),
                use_container_width=True,
                config={"displayModeBar": False}
            )

        else:
            st.info("Not enough dated records to draw a trend.")

    else:
        st.info("A valid date column is required for the trend chart.")

    # ---------- Remaining charts (two per row) ----------

    remaining = [
        fig for key, fig in chart_map.items() if key not in shown_keys
    ]

    if remaining:

        st.markdown('<div class="section-title">📊 Performance breakdown</div>',
                    unsafe_allow_html=True)

    for i in range(0, len(remaining), 2):

        pair = remaining[i:i + 2]

        if len(pair) == 2:
            left, right = st.columns(2)
            left.plotly_chart(pair[0], use_container_width=True,
                              config=plot_config)
            right.plotly_chart(pair[1], use_container_width=True,
                               config=plot_config)
        else:
            st.plotly_chart(pair[0], use_container_width=True,
                            config=plot_config)

    if not chart_map:
        st.info("Add store, product or payment columns to see more charts.")


with tabs[0]:
    render_overview()


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
# TAB 9: FORECAST (NEW)
# ============================================================

with tabs[8]:

    render_forecast_tab(
        filtered_df,
        date_col,
        sales_col,
        store_col
    )


# ============================================================
# TAB 10: RECOMMENDATIONS (NEW)
# ============================================================

with tabs[9]:

    render_recommendation_tab(
        filtered_df,
        ai_cols
    )


# ============================================================
# TAB 11: ASK YOUR DATA (NEW)
# ============================================================

with tabs[10]:

    render_nlq_tab(
        filtered_df,
        ai_cols
    )


# ============================================================
# FOOTER
# ============================================================

if not presentation_mode:

    st.divider()

    st.markdown(
        '<div class="app-footer">'
        '<b>Retail Analytics &amp; Intelligence System</b><br>'
        'Developed by Chiranjit Sahu | B.Tech CSE (AIML)<br>'
        'Python · Pandas · Streamlit · Plotly · scikit-learn · Ollama · SQL'
        '</div>',
        unsafe_allow_html=True
    )