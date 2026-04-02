import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="E-commerce Sales EDA", page_icon="📊", layout="wide")

# --- 2. SIDEBAR & PORTFOLIO BRANDING ---
st.sidebar.header("👨‍💻 About the Developer")
st.sidebar.markdown("**Mohammad Saizan Ansari**")
st.sidebar.markdown("B.Tech in AI & Data Science (2025)")
st.sidebar.markdown("[LinkedIn](https://linkedin.com/in/saizan-ansari) | [GitHub](https://github.com/MoSaizanCoder)")
st.sidebar.divider()

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(current_dir, 'Superstore.csv')
    df = pd.read_csv(file_path, encoding='windows-1252')
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    return df

st.title("📊 Comprehensive E-commerce Data Analysis")
st.markdown("A deep dive into sales patterns using Univariate, Bivariate, and Multivariate analysis.")

try:
    df = load_data()
    
    # --- 4. SIDEBAR FILTERS ---
    st.sidebar.header("Filter Data")
    year_filter = st.sidebar.multiselect(
        "Select Year", 
        options=df['Order Date'].dt.year.dropna().unique(), 
        default=df['Order Date'].dt.year.dropna().unique()
    )
    
    # Apply filter if years are selected, otherwise use all data
    if year_filter:
        filtered_df = df[df['Order Date'].dt.year.isin(year_filter)]
    else:
        filtered_df = df

    # --- 5. EDA TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Univariate Analysis", "📉 Bivariate Analysis", "📊 Multivariate Analysis"])

    # ==========================================
    # TAB 1: UNIVARIATE ANALYSIS
    # ==========================================
    with tab1:
        st.header("Univariate Analysis")
        st.markdown("Understanding the distribution of individual variables.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Distribution of Sales")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.histplot(filtered_df['Sales'][filtered_df['Sales'] < 1000], bins=30, kde=True, color='blue', ax=ax1)
            ax1.set_title("Sales Distribution (Values < $1000)")
            st.pyplot(fig1)
            with st.expander("Explanation"):
                st.markdown("""
                - **Right-Skewed Distribution:** The majority of transactions are of lower value (under $200).
                - **Insight:** The business relies heavily on high-volume, low-value orders rather than a few massive enterprise deals.
                """)

            st.subheader("2. Orders by Category")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.countplot(data=filtered_df, x='Category', hue='Category', palette='Set2', ax=ax2, legend=False)
            st.pyplot(fig2)
            with st.expander("Explanation"):
                st.markdown("""
                - **Volume Drivers:** Office Supplies overwhelmingly generate the most individual orders.
                - **Insight:** While Office Supplies drive footfall/web traffic, we must analyze if they drive actual profitability.
                """)

        with col2:
            st.subheader("3. Distribution of Profit")
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.histplot(filtered_df['Profit'][(filtered_df['Profit'] > -500) & (filtered_df['Profit'] < 500)], bins=30, kde=True, color='green', ax=ax3)
            ax3.set_title("Profit Distribution (-$500 to $500)")
            st.pyplot(fig3)
            with st.expander("Explanation"):
                st.markdown("""
                - **Zero-Centric Spread:** Most orders yield a very small profit margin, clustering tightly around $0.
                - **Insight:** There is a noticeable left tail indicating loss-making transactions that require immediate leakage optimization.
                """)

            st.subheader("4. Orders by Region")
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            sns.countplot(data=filtered_df, x='Region', hue='Region', palette='viridis', ax=ax4, legend=False)
            st.pyplot(fig4)
            with st.expander("Explanation"):
                st.markdown("""
                - **Geographic Dominance:** The West and East regions generate the highest order volumes.
                - **Insight:** Supply chain and inventory distribution should be heavily prioritized in these coastal hubs.
                """)

    # ==========================================
    # TAB 2: BIVARIATE ANALYSIS
    # ==========================================
    with tab2:
        st.header("Bivariate Analysis")
        st.markdown("Exploring the relationship between two different variables.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("1. Sales vs. Profit")
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=filtered_df, x='Sales', y='Profit', alpha=0.5, color='purple', ax=ax5)
            st.pyplot(fig5)
            with st.expander("Explanation"):
                st.markdown("""
                - **Variance at Scale:** While higher sales generally increase profit, there is a massive cluster of high-revenue sales resulting in severe losses (points dropping below the zero line).
                - **Insight:** High revenue does not guarantee high profit; cost structures on expensive items are flawed.
                """)
            
            st.subheader("2. Total Sales by Sub-Category")
            sales_subcat = filtered_df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False)
            fig6, ax6 = plt.subplots(figsize=(8, 4))
            sns.barplot(x=sales_subcat.values, y=sales_subcat.index, hue=sales_subcat.index, palette='Blues_r', ax=ax6, legend=False)
            st.pyplot(fig6)
            with st.expander("Explanation"):
                st.markdown("""
                - **Revenue Leaders:** Phones and Chairs generate the highest total gross revenue.
                - **Insight:** Marketing budgets should maintain focus on these high-ticket categories to sustain top-line growth.
                """)

        with col4:
            st.subheader("3. Impact of Discount on Profit")
            fig7, ax7 = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=filtered_df, x='Discount', y='Profit', color='red', ax=ax7)
            st.pyplot(fig7)
            with st.expander("Explanation"):
                st.markdown("""
                - **The Discount Trap:** Profitability plummets aggressively once discounts exceed 20%.
                - **Insight:** Discounts above 20% are directly cannibalizing net margins and should be strictly regulated by business rules.
                """)

            st.subheader("4. Total Profit by Sub-Category")
            profit_subcat = filtered_df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False)
            fig8, ax8 = plt.subplots(figsize=(8, 4))
            sns.barplot(x=profit_subcat.values, y=profit_subcat.index, hue=profit_subcat.index, palette='RdYlGn', ax=ax8, legend=False)
            st.pyplot(fig8)
            with st.expander("Explanation"):
                st.markdown("""
                - **Profit Drivers vs. Bleeders:** Copiers and Phones are highly profitable, while Tables and Bookcases are operating at a net loss.
                - **Insight:** We need to re-evaluate vendor pricing or discontinue heavily loss-making furniture lines.
                """)

    # ==========================================
    # TAB 3: MULTIVARIATE ANALYSIS
    # ==========================================
    with tab3:
        st.header("Multivariate Analysis")
        st.markdown("Analyzing complex interactions between multiple variables simultaneously.")
        
        st.subheader("1. Correlation Matrix")
        fig9, ax9 = plt.subplots(figsize=(10, 5))
        numeric_cols = filtered_df[['Sales', 'Quantity', 'Discount', 'Profit']]
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax9)
        st.pyplot(fig9)
        with st.expander("Explanation"):
            st.markdown("""
            - **Negative Correlation:** There is a distinct negative correlation (-0.22) between Discount and Profit.
            - **Weak Volume Impact:** Quantity has surprisingly little correlation with Profit (0.07), meaning simply selling more items doesn't guarantee better margins.
            """)

        col5, col6 = st.columns(2)

        with col5:
            st.subheader("2. Sales vs Profit (by Category)")
            fig10, ax10 = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=filtered_df, x='Sales', y='Profit', hue='Category', size='Discount', sizes=(20, 200), alpha=0.6, palette='deep', ax=ax10)
            st.pyplot(fig10)
            with st.expander("Explanation"):
                st.markdown("""
                - **Category Behavior:** Furniture (often the larger bubbles representing higher discounts) frequently drops into the negative profit zone despite decent sales figures.
                - **Insight:** Technology items tend to stay consistently above the profit baseline.
                """)

        with col6:
            st.subheader("3. Profit by Region and Category")
            fig11, ax11 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=filtered_df, x='Region', y='Profit', hue='Category', palette='muted', errorbar=None, ax=ax11)
            st.pyplot(fig11)
            with st.expander("Explanation"):
                st.markdown("""
                - **Regional Anomalies:** Furniture is severely unprofitable in the Central region.
                - **Insight:** Logistical costs or regional competitive pricing for Furniture in the Central area needs immediate strategic review.
                """)
            
        st.subheader("4. Monthly Sales Trend by Category")
        time_df = filtered_df.copy()
        time_df['Month-Year'] = time_df['Order Date'].dt.to_period('M')
        monthly_sales = time_df.groupby(['Month-Year', 'Category'])['Sales'].sum().reset_index()
        monthly_sales['Month-Year'] = monthly_sales['Month-Year'].dt.to_timestamp()
        
        fig12, ax12 = plt.subplots(figsize=(14, 5))
        sns.lineplot(data=monthly_sales, x='Month-Year', y='Sales', hue='Category', linewidth=2, ax=ax12)
        plt.xticks(rotation=45)
        st.pyplot(fig12)
        with st.expander("Explanation"):
            st.markdown("""
            - **Seasonality Patterns:** Significant spikes in sales are visible towards the end of the year (Q4) across all categories.
            - **Insight:** Inventory scaling and targeted ad spending should be heavily front-loaded going into the holiday months.
            """)

except FileNotFoundError:
    st.error("⚠️ Dataset file not found! Please ensure 'Superstore.csv' is exactly matching the file name in your folder.")
except Exception as e:
    st.error(f"An error occurred: {e}")