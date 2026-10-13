"""
retail_ai.py
Forecasting, recommendations and natural-language Q&A for the
Retail Analytics & Intelligence System (Streamlit).

Usage (in your main app):
    from retail_ai import render_forecast_tab, render_recommendation_tab, render_nlq_tab
"""

from __future__ import annotations

import calendar
import json
import os
import re

import numpy as np
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

OLLAMA_URL = "http://localhost:11434"   # default Ollama address

CATEGORICAL_ROLES = [
    "store", "location", "product", "payment",
    "cashier", "time_of_day", "day_of_week",
]


# ============================================================
# 1. SALES FORECASTING
# ============================================================

def _daily_series(df, date_col, sales_col):
    data = df.dropna(subset=[date_col, sales_col])
    if data.empty:
        return None
    daily = data.groupby(data[date_col].dt.normalize())[sales_col].sum().sort_index()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_index, fill_value=0.0)


def _features(dates, t0):
    """Trend + day-of-week + yearly seasonality features."""
    days = np.asarray((dates - t0).days, dtype=float) / 365.0
    dow = np.asarray(dates.dayofweek)
    doy = np.asarray(dates.dayofyear, dtype=float)
    cols = [days]
    cols += [(dow == d).astype(float) for d in range(7)]
    cols += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    return np.column_stack(cols)


def _backtest(series, holdout):
    """Compare the model with a naive 'recent average' baseline on held-out days."""
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    t0 = train.index[0]
    model = Ridge(alpha=1.0).fit(_features(train.index, t0), train.values)
    pred = np.clip(model.predict(_features(test.index, t0)), 0, None)
    baseline = np.full(len(test), train.tail(28).mean())
    denom = float(np.abs(test.values).sum()) or 1.0
    model_err = float(np.abs(test.values - pred).sum()) / denom
    base_err = float(np.abs(test.values - baseline).sum()) / denom
    return model_err, base_err


def render_forecast_tab(df, date_col, sales_col, store_col=None):
    st.header("🔮 Sales Forecast")

    if date_col is None or sales_col is None:
        st.warning("A date column and a sales amount are required for forecasting.")
        return

    data, scope = df, "All stores"
    if store_col is not None:
        options = ["All stores"] + sorted(df[store_col].dropna().astype(str).unique())
        scope = st.selectbox("Forecast scope", options)
        if scope != "All stores":
            data = df[df[store_col].astype(str) == scope]

    horizon = st.slider("Days to forecast", 7, 90, 30)

    series = _daily_series(data, date_col, sales_col)
    if series is None or len(series) < 42:
        st.info("At least 6 weeks of daily history is needed for a reliable forecast.")
        return

    t0 = series.index[0]
    model = Ridge(alpha=1.0).fit(_features(series.index, t0), series.values)
    sigma = float((series.values - model.predict(_features(series.index, t0))).std())

    future = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    pred = np.clip(model.predict(_features(future, t0)), 0, None)
    upper = pred + 1.96 * sigma
    lower = np.clip(pred - 1.96 * sigma, 0, None)

    holdout = min(28, len(series) // 4)
    model_err, base_err = _backtest(series, holdout)

    recent_total = float(series.tail(horizon).sum())
    forecast_total = float(pred.sum())
    change = (forecast_total - recent_total) / recent_total * 100 if recent_total else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Forecast revenue ({horizon}d)", f"{forecast_total:,.0f}",
              f"{change:+.1f}% vs last {horizon}d")
    m2.metric("Backtest error (model)", f"{model_err:.1%}")
    m3.metric("Backtest error (naive baseline)", f"{base_err:.1%}")

    history = series.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="Actual"))
    fig.add_trace(go.Scatter(x=future, y=upper, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future, y=lower, fill="tonexty",
                             line=dict(width=0), name="95% range",
                             fillcolor="rgba(99,102,241,0.2)"))
    fig.add_trace(go.Scatter(x=future, y=pred, name="Forecast",
                             line=dict(dash="dash", color="#6366f1")))
    fig.update_layout(title=f"Daily revenue forecast: {scope}", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    weekday = (
        pd.DataFrame({"Day": series.index.day_name(), "Revenue": series.values})
        .groupby("Day")["Revenue"].mean()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"])
        .reset_index()
    )
    st.plotly_chart(px.bar(weekday, x="Day", y="Revenue",
                           title="Average revenue by weekday"),
                    use_container_width=True)

    if model_err < base_err:
        st.success("The model beat the naive baseline on held-out days.")
    else:
        st.warning("The model did not beat the naive baseline. Treat this forecast with caution.")
    st.caption(
        "Method: ridge regression on trend, day-of-week and yearly seasonality. "
        "It cannot anticipate promotions, holidays or stock-outs unless they are "
        "visible in the history."
    )


# ============================================================
# 2. RECOMMENDATIONS
# ============================================================

def _basket_rules(df, txn_col, product_col, min_support):
    """Pairwise association rules computed with plain pandas."""
    b = df[[txn_col, product_col]].dropna().drop_duplicates()
    b.columns = ["t", "p"]
    n_txn = b["t"].nunique()
    if n_txn == 0:
        return pd.DataFrame()

    item_count = b.groupby("p")["t"].nunique()
    pairs = b.merge(b, on="t", suffixes=("_a", "_b"))
    pairs = pairs[pairs["p_a"] != pairs["p_b"]]
    pair_count = pairs.groupby(["p_a", "p_b"])["t"].nunique().reset_index(name="count")

    pair_count["support"] = pair_count["count"] / n_txn
    pair_count["confidence"] = pair_count["count"] / pair_count["p_a"].map(item_count)
    pair_count["lift"] = pair_count["confidence"] / (pair_count["p_b"].map(item_count) / n_txn)

    rules = pair_count[pair_count["support"] >= min_support]
    return rules.rename(columns={"p_a": "If customer buys", "p_b": "Recommend"}) \
                .sort_values(["lift", "confidence"], ascending=False)


def _insights(df, c):
    """Rule-based business insights. Returns a list of (level, text)."""
    out = []
    sales = c.get("sales")
    if sales is None:
        return out

    # Best and worst weekday
    if c.get("day_of_week"):
        dow = df[c["day_of_week"]].astype(str)
    elif c.get("date"):
        dow = df[c["date"]].dt.day_name()
    else:
        dow = None
    if dow is not None:
        by_day = df.groupby(dow)[sales].sum().sort_values()
        if len(by_day) > 1:
            out.append(("info",
                        f"**{by_day.index[-1]}** has the highest total revenue and "
                        f"**{by_day.index[0]}** the lowest. Consider a targeted promotion "
                        f"or lighter staffing on {by_day.index[0]}."))

    # Underperforming stores
    if c.get("store"):
        by_store = df.groupby(c["store"])[sales].sum()
        if len(by_store) > 2:
            avg = by_store.mean()
            weak = by_store[by_store < 0.8 * avg]
            if not weak.empty:
                names = ", ".join(map(str, weak.index[:5]))
                out.append(("warning",
                            f"Stores more than 20% below the store average: **{names}**. "
                            f"Compare their product mix and footfall with top stores."))

    # Pareto concentration
    if c.get("product"):
        ps = df.groupby(c["product"])[sales].sum().sort_values(ascending=False)
        if len(ps) > 4 and ps.sum() > 0:
            cum = ps.cumsum() / ps.sum()
            n80 = int((cum < 0.8).sum()) + 1
            out.append(("info",
                        f"**{n80} of {len(ps)} products ({n80 / len(ps):.0%})** generate 80% "
                        f"of revenue. Protect stock for these and review the long tail."))

    # Product momentum (last 30 days vs previous 30)
    if c.get("product") and c.get("date"):
        last = df[c["date"]].max()
        if pd.notna(last):
            recent = df[df[c["date"]] > last - pd.Timedelta(days=30)]
            prior = df[(df[c["date"]] <= last - pd.Timedelta(days=30))
                       & (df[c["date"]] > last - pd.Timedelta(days=60))]
            a = recent.groupby(c["product"])[sales].sum()
            b = prior.groupby(c["product"])[sales].sum()
            b = b[b > 0]
            if len(b) > 1:
                chg = (a.reindex(b.index).fillna(0) - b) / b
                out.append(("success",
                            f"Fastest-growing product (last 30 vs previous 30 days): "
                            f"**{chg.idxmax()}** ({chg.max():+.0%})."))
                if chg.min() < -0.1:
                    out.append(("warning",
                                f"Sharpest decline: **{chg.idxmin()}** ({chg.min():+.0%}). "
                                f"Check pricing, stock-outs or seasonality."))

    # Payment mix
    if c.get("payment"):
        share = df.groupby(c["payment"])[sales].sum()
        if share.sum() > 0 and len(share) > 1:
            share = (share / share.sum()).sort_values(ascending=False)
            out.append(("info",
                        f"**{share.index[0]}** accounts for {share.iloc[0]:.0%} of revenue. "
                        f"Make sure that payment channel is reliable at every store."))
    return out


# ---------- ML: item-based collaborative filtering ----------

def _prepare_baskets(df, cols, mode):
    """Add a '_basket' key. A basket is a transaction, or a store-day as a fallback."""
    data = df.copy()
    if mode.startswith("Same TransactionID"):
        data["_basket"] = data[cols["transaction"]].astype(str)
    else:
        key = data[cols["date"]].dt.date.astype(str)
        if cols.get("store"):
            key = key + "|" + data[cols["store"]].astype(str)
        data["_basket"] = key
    return data


def _item_similarity(df, product_col):
    """
    Item-item cosine similarity on the basket x product matrix.
    Two products are similar when they appear in the same baskets.
    """
    b = df[["_basket", product_col]].dropna().drop_duplicates()
    b.columns = ["basket", "product"]
    if b["product"].nunique() < 2:
        return None

    basket_codes = b["basket"].astype("category").cat.codes.values
    products = b["product"].astype("category")
    product_codes = products.cat.codes.values

    matrix = csr_matrix(
        (np.ones(len(b)), (basket_codes, product_codes)),
        shape=(basket_codes.max() + 1, len(products.cat.categories)),
    )
    sim = cosine_similarity(matrix.T, dense_output=True)
    np.fill_diagonal(sim, 0.0)

    names = products.cat.categories.astype(str)
    return pd.DataFrame(sim, index=names, columns=names)


def _cross_sell_table(rules, sim, product, top=5):
    scores = sim.loc[product].sort_values(ascending=False)
    scores = scores[scores > 0].head(top * 2)
    table = scores.rename("Similarity").to_frame()
    table.index.name = "Recommend"
    table = table.reset_index()

    if not rules.empty:
        r = rules[rules["If customer buys"].astype(str) == product][
            ["Recommend", "support", "confidence", "lift"]
        ].copy()
        r["Recommend"] = r["Recommend"].astype(str)
        table = table.merge(r, on="Recommend", how="left")
    else:
        table[["support", "confidence", "lift"]] = np.nan
    return table.head(top)


def _render_cross_sell(df, cols):
    prod = cols.get("product")
    if prod is None:
        st.info("A Product column is required for cross-sell recommendations.")
        return

    modes = []
    if cols.get("transaction"):
        modes.append("Same TransactionID")
    if cols.get("date"):
        modes.append("Same store and day" if cols.get("store") else "Same day")
    if not modes:
        st.info("A TransactionID or Date column is required for basket analysis.")
        return

    mode = st.radio(
        "What counts as one basket?", modes, horizontal=True,
        help="Use 'Same store and day' when each TransactionID holds only one product."
    )
    data = _prepare_baskets(df, cols, mode)
    items_per_basket = data.groupby("_basket")[prod].nunique().mean()

    if items_per_basket < 1.2:
        st.warning(
            f"Baskets hold only {items_per_basket:.2f} distinct products on average, so "
            "there is little to learn from. Try the 'Same store and day' option above."
        )

    min_support = st.slider("Minimum support (share of baskets)",
                            0.001, 0.30, 0.01, step=0.001, format="%.3f")

    rules = _basket_rules(data, "_basket", prod, min_support)
    sim = _item_similarity(data, prod)
    if sim is None:
        st.info("At least two products are needed.")
        return

    chosen = st.selectbox("Pick a product to get cross-sell suggestions", sorted(sim.index))
    table = _cross_sell_table(rules, sim, chosen)

    if table.empty:
        st.info("No related products found for this product.")
    else:
        st.plotly_chart(
            px.bar(table.sort_values("Similarity"), x="Similarity", y="Recommend",
                   orientation="h", title=f"Products most similar to {chosen}"),
            use_container_width=True)
        st.dataframe(
            table.rename(columns={"support": "Support", "confidence": "Confidence",
                                  "lift": "Lift"})
            .style.format({"Similarity": "{:.3f}", "Support": "{:.2%}",
                           "Confidence": "{:.1%}", "Lift": "{:.2f}"}, na_rep="-"),
            use_container_width=True, hide_index=True)

    with st.expander("All association rules"):
        if rules.empty:
            st.write("No product pairs meet this support level.")
        else:
            st.dataframe(rules.head(100), use_container_width=True, hide_index=True)

    with st.expander("How these recommendations are calculated"):
        st.markdown(
            "**Similarity (item-based collaborative filtering):** each product becomes a vector "
            "of the baskets it appears in. Cosine similarity measures how alike two vectors are, "
            "from 0 (never together) to 1 (always together).\n\n"
            "**Support, confidence and lift (association rules):** support is how often both "
            "products appear together. Confidence is how often B is bought when A is bought. "
            "Lift above 1 means the pair is bought together more than chance."
        )


# ---------- ML: K-Means segmentation ----------

def segment_entities(df, group_col, cols, k=None, entity="Products"):
    """Cluster products or stores with K-Means on revenue, volume, price and growth."""
    sales = cols["sales"]
    txn, qty, date_col = cols.get("transaction"), cols.get("quantity"), cols.get("date")

    g = df.groupby(group_col)
    feats = pd.DataFrame({
        "Revenue": g[sales].sum(),
        "Avg sale": g[sales].mean(),
        "Transactions": g[txn].nunique() if txn else g.size(),
    })
    if qty:
        feats["Units"] = g[qty].sum()

    growth_series = None
    if date_col:
        last = df[date_col].max()
        recent = df[df[date_col] > last - pd.Timedelta(days=30)].groupby(group_col)[sales].sum()
        prior = df[(df[date_col] <= last - pd.Timedelta(days=30))
                   & (df[date_col] > last - pd.Timedelta(days=60))].groupby(group_col)[sales].sum()
        prior = prior[prior > 0]
        if len(prior) > 1:
            growth = (recent.reindex(prior.index).fillna(0) - prior) / prior
            growth_series = growth.reindex(feats.index).fillna(0).clip(-1, 2)

    feats = feats.dropna()
    n = len(feats)
    if n < 4:
        return None

    X = feats.copy()
    for c in ("Revenue", "Avg sale", "Transactions", "Units"):
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    Xs = StandardScaler().fit_transform(X)

    # keep clusters meaningful: roughly 3+ members each on average
    max_k = min(5, max(2, n // 3), n - 1)
    scores = {}
    for kk in range(2, max_k + 1):
        labels = KMeans(n_clusters=kk, n_init=10, random_state=42).fit_predict(Xs)
        if len(set(labels)) > 1:
            scores[kk] = silhouette_score(Xs, labels)
    if not scores:
        return None

    best_k = max(scores, key=scores.get)
    used_k = k if k in scores else best_k
    labels = KMeans(n_clusters=used_k, n_init=10, random_state=42).fit_predict(Xs)
    feats["Cluster"] = labels
    if growth_series is not None:
        feats["Growth"] = growth_series.reindex(feats.index)   # context only, not clustered

    order = feats.groupby("Cluster")["Revenue"].mean().sort_values(ascending=False).index.tolist()
    is_product = entity == "Products"
    text_top = ("Protect availability, feature in promotions and avoid heavy discounting."
                if is_product else
                "Use as a benchmark: share its practices with other stores and protect its stock and staffing.")
    text_mid = ("Maintain stock. Test bundles or cross-sell to move these upward."
                if is_product else
                "Maintain. Test local promotions to move these toward the top tier.")
    text_low = ("Review pricing, placement or range. Test a promotion before considering delisting."
                if is_product else
                "Investigate footfall, product mix and staffing. Compare with top stores before acting.")

    names, actions = {}, {}
    for pos, cluster in enumerate(order):
        if pos == 0:
            names[cluster], actions[cluster] = "Top performers", text_top
        elif pos == len(order) - 1:
            names[cluster], actions[cluster] = "Low performers", text_low
        else:
            names[cluster] = "Mid tier" if len(order) == 3 else f"Mid tier {pos}"
            actions[cluster] = text_mid
    feats["Segment"] = feats["Cluster"].map(names)

    return {"data": feats, "k": used_k, "best_k": best_k, "scores": scores,
            "silhouette": scores[used_k], "actions": {names[c]: actions[c] for c in order},
            "order": [names[c] for c in order], "max_k": max_k}


def _render_segmentation(df, cols):
    if cols.get("sales") is None:
        st.info("A sales amount is required for segmentation.")
        return

    options = {}
    if cols.get("product"):
        options["Products"] = cols["product"]
    if cols.get("store"):
        options["Stores"] = cols["store"]
    if cols.get("location"):
        options["Locations"] = cols["location"]
    if not options:
        st.info("A Product, Store or Location column is required.")
        return

    c1, c2 = st.columns(2)
    label = c1.selectbox("What to segment", list(options.keys()))
    group_col = options[label]

    base = segment_entities(df, group_col, cols, entity=label)
    if base is None:
        st.info(f"Need at least 4 {label.lower()} with valid data to build segments.")
        return

    choice = c2.selectbox("Number of segments", ["Auto (best silhouette)"] + list(range(2, base["max_k"] + 1)))
    res = base if choice == "Auto (best silhouette)" else segment_entities(
        df, group_col, cols, k=int(choice), entity=label) or base

    feats = res["data"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Segments", res["k"])
    m2.metric("Silhouette score", f"{res['silhouette']:.2f}",
              help="Closer to 1 means clusters are well separated.")
    m3.metric(f"{label} analysed", len(feats))

    if res["silhouette"] < 0.25:
        st.warning("The segments overlap a lot (silhouette below 0.25), so the differences between "
                   "them are small. Treat these groups as a rough guide, or try fewer segments.")
    if res["k"] >= len(feats) / 2:
        st.warning("There are very few members per segment. Fewer segments will be more reliable.")

    total_rev = feats["Revenue"].sum()
    rows = []
    for seg in res["order"]:
        part = feats[feats["Segment"] == seg]
        row = {"Segment": seg, label: len(part),
               "Revenue share": part["Revenue"].sum() / total_rev if total_rev else 0,
               "Avg revenue each": part["Revenue"].mean()}
        if "Growth" in part:
            row["Avg growth (30d)"] = part["Growth"].mean()
        action = res["actions"][seg]
        if "Growth" in part:
            g = part["Growth"].mean()
            if seg == res["order"][0] and g <= -0.15:
                action += " Growth is falling, so investigate early."
            elif seg == res["order"][-1] and g >= 0.15:
                action += " Growth is rising, so monitor before cutting back."
        row["Recommended action"] = action
        rows.append(row)
    summary = pd.DataFrame(rows)

    fmt = {"Revenue share": "{:.1%}", "Avg revenue each": "{:,.0f}"}
    if "Avg growth (30d)" in summary:
        fmt["Avg growth (30d)"] = "{:+.0%}"
    st.dataframe(summary.style.format(fmt), use_container_width=True, hide_index=True)

    plot = feats.reset_index().rename(columns={group_col: label[:-1]})
    st.plotly_chart(
        px.scatter(plot, x="Transactions", y="Revenue", color="Segment", size="Avg sale",
                   hover_name=label[:-1], log_x=True, log_y=True,
                   category_orders={"Segment": res["order"]},
                   title=f"{label} segments (K-Means, log scale)"),
        use_container_width=True)

    for seg in res["order"]:
        with st.expander(f"{seg}: members"):
            members = plot[plot["Segment"] == seg].sort_values("Revenue", ascending=False)
            st.dataframe(members.drop(columns=["Cluster"]), use_container_width=True,
                         hide_index=True)

    st.caption("Segments are built from revenue, average sale, transactions and units (log-scaled "
               "and standardised). 30-day growth is shown for context only, is capped at -100% to "
               "+200%, and is noisy for small groups. Review segments alongside stock, pricing "
               "and seasonality.")


def render_recommendation_tab(df, cols):
    st.header("💡 Recommendations & Insights")

    tab_insights, tab_cross, tab_segments = st.tabs(
        ["📌 Business insights", "🛒 Cross-sell (ML)", "🧩 Segmentation (K-Means)"])

    with tab_insights:
        insights = _insights(df, cols)
        if not insights:
            st.info("Not enough columns detected to generate insights.")
        for level, text in insights:
            {"info": st.info, "warning": st.warning, "success": st.success}[level](text)

    with tab_cross:
        _render_cross_sell(df, cols)

    with tab_segments:
        _render_segmentation(df, cols)


# ============================================================
# 3. NATURAL-LANGUAGE QUERIES
# ============================================================

ROLE_WORDS = {
    "store": ["store", "stores", "shop", "shops", "branch", "branches"],
    "location": ["location", "locations", "city", "cities", "area", "region"],
    "product": ["product", "products", "item", "items"],
    "payment": ["payment", "payments", "payment type", "payment method"],
    "cashier": ["cashier", "cashiers", "employee", "employees"],
    "time_of_day": ["time of day", "shift", "shifts"],
    "day_of_week": ["weekday", "weekdays", "day of week", "day of the week", "day"],
    "month": ["month", "monthly", "months"],
    "date": ["daily", "date", "dates"],
}

EXAMPLES = [
    "Total revenue by store",
    "Top 5 products by revenue",
    "Monthly revenue trend",
    "Average sale by payment type",
    "How many transactions per cashier?",
    "Why did revenue drop in August compared to July?",
]


def _rule_based_spec(question, cols, df):
    q = question.lower()

    if re.search(r"\b(average|avg|mean)\b", q):
        metric = "avg_sale"
    elif re.search(r"\b(units|quantity|qty|sold)\b", q):
        metric = "units"
    elif re.search(r"\b(transactions?|orders?|bills?|how many)\b", q):
        metric = "transactions"
    else:
        metric = "revenue"

    group_by = []
    for role, words in ROLE_WORDS.items():
        alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        pattern = (rf"\b(?:by|per|each|across|for each|wise)\s+(?:the\s+)?(?:{alt})\b"
                   rf"|\b(?:top|best|worst|lowest|highest|bottom)\s+(?:\d+\s+)?(?:{alt})\b")
        if re.search(pattern, q) or (role in ("month", "date") and re.search(rf"\b(?:{alt})\b", q)):
            group_by.append(role)
    group_by = group_by[:2]

    top_n = None
    m = re.search(r"\b(?:top|bottom|best|worst|highest|lowest)\s+(\d+)", q)
    if m:
        top_n = int(m.group(1))
    elif re.search(r"\b(top|bottom|best|worst)\b", q):
        top_n = 5
    sort = "asc" if re.search(r"\b(bottom|worst|lowest|least)\b", q) else "desc"

    filters = []
    for role in CATEGORICAL_ROLES:
        col = cols.get(role)
        if not col:
            continue
        uniques = df[col].dropna().astype(str).unique()
        if len(uniques) > 60:
            continue
        hits = [u for u in uniques
                if len(u) >= 2 and re.search(rf"\b{re.escape(u.lower())}\b", q)]
        if len(hits) == 1:
            filters.append({"column": role, "op": "eq", "value": hits[0]})
        elif len(hits) > 1:
            filters.append({"column": role, "op": "in", "value": hits})

    year = re.search(r"\b(20\d{2})\b", q)
    if year and cols.get("date"):
        y = year.group(1)
        filters += [{"column": "date", "op": "gte", "value": f"{y}-01-01"},
                    {"column": "date", "op": "lte", "value": f"{y}-12-31"}]

    return {"metric": metric, "group_by": group_by, "filters": filters,
            "sort": sort, "top_n": top_n}


def list_ollama_models(base_url):
    """Return installed Ollama model names, or [] if Ollama is not reachable."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _ollama_chat(base_url, model, system, user, json_mode=False):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0},
    }
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _llm_spec(question, cols, df, base_url, model):
    catalog = {
        role: sorted(df[col].dropna().astype(str).unique())[:40]
        for role, col in cols.items()
        if col and role in CATEGORICAL_ROLES
    }
    date_info = ""
    if cols.get("date"):
        date_info = (f"Data covers {df[cols['date']].min().date()} "
                     f"to {df[cols['date']].max().date()}.")

    system = f"""You translate retail analytics questions into a JSON query spec.
Return ONLY one JSON object, no prose.

Schema:
{{
  "metric": "revenue" | "units" | "transactions" | "avg_sale" | "records",
  "group_by": [up to 2 of: "store","location","product","payment","cashier",
               "time_of_day","day_of_week","month","date"],
  "filters": [{{"column": <role or "date","sales","quantity">,
               "op": "eq"|"in"|"gt"|"gte"|"lt"|"lte"|"contains",
               "value": <string, number or list>}}],
  "sort": "asc" | "desc",
  "top_n": integer or null
}}

Examples:
Q: top 3 products by revenue
A: {{"metric":"revenue","group_by":["product"],"filters":[],"sort":"desc","top_n":3}}
Q: average sale per store paid by UPI
A: {{"metric":"avg_sale","group_by":["store"],"filters":[{{"column":"payment","op":"eq","value":"UPI"}}],"sort":"desc","top_n":null}}
Q: monthly units sold in 2024
A: {{"metric":"units","group_by":["month"],"filters":[{{"column":"date","op":"gte","value":"2024-01-01"}},{{"column":"date","op":"lte","value":"2024-12-31"}}],"sort":"desc","top_n":null}}

Dates in filters must be ISO format (YYYY-MM-DD).
Available columns and their values (sample): {json.dumps(catalog)}
Available roles with data: {[r for r, c in cols.items() if c]}
{date_info}
If the question cannot be answered with this schema, return {{"error": "<short reason>"}}."""

    text = _ollama_chat(base_url, model, system, question, json_mode=True)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("The model did not return a valid query.")
    return json.loads(match.group(0))


def _llm_explain(question, result, base_url, model):
    return _ollama_chat(
        base_url, model,
        "You are a retail analyst. Explain the result in 2-3 plain sentences. "
        "Use only the numbers provided.",
        f"Question: {question}\n\nResult:\n{result.head(20).to_csv(index=False)}",
    )


def _apply_filter(df, f, cols):
    role, op, val = f.get("column"), f.get("op", "eq"), f.get("value")
    col = cols.get(role)
    if col is None:
        raise ValueError(f"Cannot filter on '{role}': column not found in your data.")
    s = df[col]

    if role == "date":
        v = pd.to_datetime(val, errors="coerce")
        if pd.isna(v):
            raise ValueError(f"Invalid date in filter: {val}")
        if op in ("gte", "gt"):
            return df[s >= v]
        if op in ("lte", "lt"):
            return df[s <= v]
        return df[s.dt.normalize() == v.normalize()]

    if role in ("sales", "quantity"):
        v = float(val)
        ops = {"gt": s > v, "gte": s >= v, "lt": s < v, "lte": s <= v, "eq": s == v}
        if op not in ops:
            raise ValueError(f"Unsupported operator '{op}' for numeric filter.")
        return df[ops[op]]

    sl = s.astype(str).str.lower()
    if op == "in":
        vals = [str(x).lower() for x in (val if isinstance(val, list) else [val])]
        return df[sl.isin(vals)]
    if op == "contains":
        return df[sl.str.contains(re.escape(str(val).lower()), na=False)]
    return df[sl == str(val).lower()]


def run_spec(df, spec, cols):
    """Execute a validated spec with pandas. No generated code is ever executed."""
    labels = {"revenue": "Revenue", "units": "Units", "transactions": "Transactions",
              "avg_sale": "Average Sale", "records": "Records"}
    metric = spec.get("metric", "revenue")
    if metric not in labels:
        raise ValueError(f"Unsupported metric '{metric}'.")
    label = labels[metric]

    required = {"revenue": "sales", "avg_sale": "sales", "units": "quantity",
                "transactions": "transaction"}
    if metric in required and not cols.get(required[metric]):
        raise ValueError(f"This question needs a '{required[metric]}' column, "
                         f"which was not detected in your data.")

    work = df
    for f in spec.get("filters") or []:
        work = _apply_filter(work, f, cols)
    if work.empty:
        raise ValueError("No records match the filters in this question.")
    work = work.copy()

    keys = []
    for g in (spec.get("group_by") or [])[:2]:
        if g == "month" and cols.get("date"):
            work["Month"] = work[cols["date"]].dt.to_period("M").astype(str)
            keys.append("Month")
        elif g == "date" and cols.get("date"):
            work["Date"] = work[cols["date"]].dt.date
            keys.append("Date")
        elif g in CATEGORICAL_ROLES and cols.get(g):
            keys.append(cols[g])
        else:
            raise ValueError(f"Cannot group by '{g}': column not found in your data.")

    group_keys = keys or ["_all"]
    if not keys:
        work["_all"] = "All"
    grp = work.groupby(group_keys, dropna=False)

    if metric == "revenue":
        r = grp[cols["sales"]].sum()
    elif metric == "avg_sale":
        r = grp[cols["sales"]].mean()
    elif metric == "units":
        r = grp[cols["quantity"]].sum()
    elif metric == "transactions":
        r = grp[cols["transaction"]].nunique()
    else:
        r = grp.size()

    res = r.reset_index(name=label)
    if not keys:
        res = res.drop(columns="_all")
        return res, keys, label

    if "Month" in keys or "Date" in keys:
        res = res.sort_values(keys)
    else:
        res = res.sort_values(label, ascending=(spec.get("sort") == "asc"))

    top_n = spec.get("top_n")
    if isinstance(top_n, (int, float)) and top_n:
        res = res.head(int(max(1, min(top_n, 100))))
    return res, keys, label


def _show_result(res, keys, label):
    if not keys:
        st.metric(label, f"{res.iloc[0, 0]:,.2f}")
        return
    if "Month" in keys or "Date" in keys:
        time_key = "Month" if "Month" in keys else "Date"
        other = [k for k in keys if k != time_key]
        fig = px.line(res, x=time_key, y=label, markers=True,
                      color=other[0] if other else None)
    else:
        fig = px.bar(res, x=keys[0], y=label,
                     color=keys[1] if len(keys) > 1 else None, barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(res, use_container_width=True, hide_index=True)


# ============================================================
# 4. "WHY DID IT CHANGE?" DRIVER ANALYSIS
# ============================================================

_MONTH_NAMES = {n.lower(): i for i, n in enumerate(calendar.month_name) if n}
_MONTH_NAMES.update({n.lower(): i for i, n in enumerate(calendar.month_abbr) if n})
_MONTH_NAMES["sept"] = 9

_CHANGE_WORDS = (r"\b(why|reason|reasons|cause|causes|decline|declined|drop|dropped|fall|fell|"
                 r"decrease|decreased|increase|increased|rise|rose|growth|compare|compared|"
                 r"comparison|versus|vs|difference)\b")
_WHY_WORDS = (r"\b(why|reason|reasons|cause|causes|decline|declined|drop|dropped|fall|fell|"
              r"decrease|decreased|increase|increased|rise|rose|growth)\b")

DRIVER_DIMENSIONS = [
    ("store", "Store"), ("product", "Product"), ("payment", "Payment type"),
    ("location", "Location"), ("cashier", "Cashier"),
    ("time_of_day", "Time of day"), ("day_of_week", "Day of week"),
]


def _plabel(period):
    year, month = period
    return f"{calendar.month_name[month]} {year}"


def _detect_change_question(question, df, cols):
    """Return {'base':(y,m),'cur':(y,m)}, {'error':...}, or None if not a change question."""
    date_col = cols.get("date")
    if date_col is None or cols.get("sales") is None:
        return None
    q = question.lower()
    if not re.search(_CHANGE_WORDS, q):
        return None

    months = []
    for word in re.findall(r"\b[a-z]+\b", q):
        number = _MONTH_NAMES.get(word)
        if number and number not in months:
            months.append(number)
    if not months:
        return None
    if len(months) == 1 and not re.search(_WHY_WORDS, q):
        return None

    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]
    dates = df[date_col].dropna()

    def resolve(month, index):
        if years:
            return (years[index] if index < len(years) else years[0], month)
        available = sorted(dates[dates.dt.month == month].dt.year.unique())
        return (int(available[-1]), month) if len(available) else None

    if len(months) >= 2:
        periods = [resolve(months[0], 0), resolve(months[1], 1)]
        if None in periods:
            return {"error": "I couldn't find data for one of those months."}
        base, cur = sorted(periods)
    else:
        cur = resolve(months[0], 0)
        if cur is None:
            return {"error": f"No data found for {calendar.month_name[months[0]]}."}
        base = (cur[0] - 1, 12) if cur[1] == 1 else (cur[0], cur[1] - 1)
    return {"base": base, "cur": cur}


def _period_stats(frame, period, cols):
    date_col, sales = cols["date"], cols["sales"]
    revenue = float(frame[sales].sum())
    txn = int(frame[cols["transaction"]].nunique()) if cols.get("transaction") else len(frame)
    days = int(frame[date_col].dt.normalize().nunique())
    return {
        "revenue": revenue,
        "transactions": txn,
        "units": float(frame[cols["quantity"]].sum()) if cols.get("quantity") else None,
        "active_days": days,
        "days_in_month": calendar.monthrange(*period)[1],
        "first": frame[date_col].min(),
        "last": frame[date_col].max(),
        "avg_txn": revenue / txn if txn else 0.0,
        "per_day": revenue / days if days else 0.0,
    }


def _pct(new, old):
    return (new - old) / old * 100 if old else float("nan")


def analyze_change(df, cols, base, cur):
    date_col, sales = cols["date"], cols["sales"]
    in_period = lambda p: df[(df[date_col].dt.year == p[0]) & (df[date_col].dt.month == p[1])]
    a, b = in_period(base), in_period(cur)
    if a.empty:
        raise ValueError(f"No records found for {_plabel(base)}.")
    if b.empty:
        raise ValueError(f"No records found for {_plabel(cur)}.")

    sa, sb = _period_stats(a, base, cols), _period_stats(b, cur, cols)
    bl, cl = _plabel(base), _plabel(cur)
    findings = []

    # headline
    chg = _pct(sb["revenue"], sa["revenue"])
    verb = "fell" if sb["revenue"] < sa["revenue"] else "rose"
    findings.append(("info", f"Revenue **{verb} {abs(chg):.1f}%**, from {sa['revenue']:,.0f} in {bl} "
                             f"to {sb['revenue']:,.0f} in {cl}."))

    # data coverage (often the real cause of a huge drop)
    coverage_issue = False
    for label, st_ in ((bl, sa), (cl, sb)):
        if st_["active_days"] < 0.7 * st_["days_in_month"]:
            coverage_issue = True
            findings.append((
                "warning",
                f"**Incomplete data in {label}:** records exist for only {st_['active_days']} of "
                f"{st_['days_in_month']} days (from {st_['first']:%d %b} to {st_['last']:%d %b}). "
                f"Part of the change may simply be missing days, not weaker sales."))
    if sa["active_days"] and sb["active_days"]:
        findings.append(("info",
                         f"Revenue per active day went from {sa['per_day']:,.0f} to {sb['per_day']:,.0f} "
                         f"({_pct(sb['per_day'], sa['per_day']):+.1f}%). This removes the effect of "
                         f"having fewer or more trading days."))

    # volume vs value
    volume_effect = (sb["transactions"] - sa["transactions"]) * sa["avg_txn"]
    value_effect = (sb["avg_txn"] - sa["avg_txn"]) * sb["transactions"]
    dominant = "number of transactions" if abs(volume_effect) >= abs(value_effect) else "average transaction value"
    findings.append((
        "info",
        f"Transactions changed {_pct(sb['transactions'], sa['transactions']):+.1f}% "
        f"({sa['transactions']:,} to {sb['transactions']:,}) and average transaction value changed "
        f"{_pct(sb['avg_txn'], sa['avg_txn']):+.1f}% ({sa['avg_txn']:,.0f} to {sb['avg_txn']:,.0f}). "
        f"The **{dominant}** explains most of the revenue change "
        f"(volume effect {volume_effect:+,.0f}, value effect {value_effect:+,.0f})."))

    # contribution by dimension
    tables, broad = {}, True
    for role, name in DRIVER_DIMENSIONS:
        col = cols.get(role)
        if col is None or df[col].nunique() < 2:
            continue
        t = pd.concat([a.groupby(col)[sales].sum().rename("Base"),
                       b.groupby(col)[sales].sum().rename("Current")], axis=1).fillna(0.0)
        t["Change"] = t["Current"] - t["Base"]
        total = t["Change"].sum()
        t["Share of change"] = t["Change"] / total if total else np.nan
        t = t.reindex(t["Change"].abs().sort_values(ascending=False).index).reset_index()
        t = t.rename(columns={t.columns[0]: name})
        tables[name] = t

        top = t.iloc[0]
        share = top["Share of change"]
        if pd.notna(share) and abs(share) >= 0.5:
            broad = False
        if len(findings) < 12:
            findings.append((
                "info",
                f"**{name}:** {top[name]} is the largest contributor "
                f"({top['Change']:+,.0f}, {share:.0%} of the net change)."
                if pd.notna(share) else
                f"**{name}:** {top[name]} changed by {top['Change']:+,.0f}."))

        lost = t[(t["Base"] > 0) & (t["Current"] == 0)][name].astype(str).tolist()
        if lost and role in ("store", "product", "payment"):
            findings.append(("warning", f"**{name} with sales in {bl} but none in {cl}:** "
                                        + ", ".join(lost[:8])))

    if tables and broad:
        findings.append(("info", "The change is **broad-based**: no single store, product or payment "
                                 "type explains most of it. That pattern fits a data-coverage or "
                                 "overall demand effect better than a single failing item."))

    return {"a": a, "b": b, "sa": sa, "sb": sb, "bl": bl, "cl": cl,
            "findings": findings, "tables": tables, "coverage_issue": coverage_issue}


def _llm_narrate(findings, question, base_url, model):
    facts = "\n".join(f"- {text}" for _, text in findings)
    return _ollama_chat(
        base_url, model,
        "You are a retail analyst. Using ONLY the facts provided, write a short explanation "
        "(4 to 6 sentences) of why revenue changed. Mention data-coverage problems first if present. "
        "Do not invent causes that are not supported by the facts.",
        f"Question: {question}\n\nFacts:\n{facts}",
    )


def render_change_analysis(df, cols, intent, base_url=None, model=None, explain=False, question=""):
    try:
        res = analyze_change(df, cols, intent["base"], intent["cur"])
    except Exception as error:
        st.error(f"{error}")
        return

    sa, sb, bl, cl = res["sa"], res["sb"], res["bl"], res["cl"]
    st.subheader(f"Why did revenue change? {bl} → {cl}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Revenue", f"{sb['revenue']:,.0f}", f"{_pct(sb['revenue'], sa['revenue']):+.1f}%")
    m2.metric("Transactions", f"{sb['transactions']:,}",
              f"{_pct(sb['transactions'], sa['transactions']):+.1f}%")
    m3.metric("Avg transaction value", f"{sb['avg_txn']:,.0f}",
              f"{_pct(sb['avg_txn'], sa['avg_txn']):+.1f}%")
    m4.metric("Days with data", f"{sb['active_days']}",
              f"{sb['active_days'] - sa['active_days']:+d} vs {bl}")

    st.markdown("#### Key findings")
    for level, text in res["findings"]:
        {"info": st.info, "warning": st.warning, "success": st.success}[level](text)

    if explain and model:
        try:
            st.markdown("#### Plain-English explanation")
            st.write(_llm_narrate(res["findings"], question, base_url, model))
        except Exception as error:
            st.warning(f"Could not generate explanation: {error}")

    # daily revenue for both months
    date_col, sales = cols["date"], cols["sales"]
    daily = []
    for label, frame in ((bl, res["a"]), (cl, res["b"])):
        g = frame.groupby(frame[date_col].dt.day)[sales].sum().reset_index()
        g.columns = ["Day of month", "Revenue"]
        g["Period"] = label
        daily.append(g)
    st.plotly_chart(
        px.line(pd.concat(daily), x="Day of month", y="Revenue", color="Period", markers=True,
                title="Daily revenue by day of month"),
        use_container_width=True)

    if res["tables"]:
        st.markdown("#### What drove the change")
        tabs = st.tabs(list(res["tables"].keys()))
        for tab, (name, table) in zip(tabs, res["tables"].items()):
            with tab:
                top = table.head(8)
                st.plotly_chart(px.bar(top, x=name, y="Change", title=f"Revenue change by {name.lower()}"),
                                use_container_width=True)
                st.dataframe(table.style.format({"Base": "{:,.0f}", "Current": "{:,.0f}",
                                                 "Change": "{:+,.0f}", "Share of change": "{:.0%}"}),
                             use_container_width=True, hide_index=True)
    st.caption("This analysis shows where the change happened. It can't prove why customers "
               "behaved differently (promotions, holidays or stock-outs aren't in the data).")


def render_nlq_tab(df, cols):
    st.header("💬 Ask Your Data")
    st.write("Ask a question in plain English and get a table and chart back.")

    mode = st.radio("Engine",
                    ["Built-in parser (instant)", "Local LLM via Ollama (flexible)"],
                    horizontal=True)

    base_url, model = None, None
    if mode.startswith("Local"):
        base_url = st.text_input("Ollama URL", value=OLLAMA_URL)
        models = list_ollama_models(base_url)
        if models:
            model = st.selectbox("Model", models)
        else:
            st.warning("Could not reach Ollama or no models are installed. "
                       "Start it with `ollama serve` and run `ollama pull llama3.1:8b`.")
        st.caption("Everything runs on your machine. Nothing is sent to the internet.")

    example = st.selectbox("Try an example", [""] + EXAMPLES)
    question = st.text_input("Your question", value=example,
                             placeholder="e.g. Top 3 products by revenue in 2024")
    explain = st.checkbox("Explain the result in plain English",
                          value=False, disabled=model is None)

    if not question:
        return

    # "Why did revenue drop in August vs July?" -> driver analysis
    intent = _detect_change_question(question, df, cols)
    if intent:
        if "error" in intent:
            st.warning(intent["error"])
        else:
            render_change_analysis(df, cols, intent, base_url, model, explain, question)
        return

    spec = None
    if model:
        try:
            spec = _llm_spec(question, cols, df, base_url, model)
            if "error" in spec:
                st.warning(f"I couldn't answer that: {spec['error']}")
                return
            run_spec(df, spec, cols)          # validate before accepting
        except Exception as error:
            st.info(f"The local model's answer could not be used ({error}). "
                    "Falling back to the built-in parser.")
            spec = None

    try:
        if spec is None:
            spec = _rule_based_spec(question, cols, df)
        result, keys, label = run_spec(df, spec, cols)
    except Exception as error:
        st.error(f"{error}")
        return

    with st.expander("How I interpreted your question"):
        st.json(spec)

    _show_result(result, keys, label)

    if explain and model:
        try:
            st.info(_llm_explain(question, result, base_url, model))
        except Exception as error:
            st.warning(f"Could not generate explanation: {error}")