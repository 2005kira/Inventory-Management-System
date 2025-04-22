import pandas as pd                                  # For data manipulation and analysis
import numpy as np                                   # For numerical operations
import streamlit as st                               # To build the interactive Streamlit app
import matplotlib.pyplot as plt                      # For basic plotting
import seaborn as sns                                # For advanced statistical plots
import plotly.express as px                          # For interactive visualizations
import plotly.graph_objects as go                    # For detailed plot customization

from sklearn.ensemble import IsolationForest         # For anomaly detection in datasets
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # For encoding categories and scaling data

from keras.models import Sequential                  # Keras model architecture (Sequential)
from keras.layers import LSTM, Dense                 # LSTM layer for time series, Dense for output layer

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator  # For creating time series data windows

from mlxtend.frequent_patterns import apriori, association_rules  # For market basket analysis

import sqlite3                                       # To interact with SQLite databases
import requests                                      # For making API requests
from datetime import datetime, timedelta           
import time                                    
import re                                            # pattern matching using regular expressions

# Page configuration
st.set_page_config(
    page_title="Smart Inventory Management", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom embedded CSS
st.markdown("""
<style>
    body {
        background-color: #1e293b;
        color: #e2e8f0;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #93c5fd;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #334155;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #93c5fd;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    .section-card {
        background-color: #334155;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(255, 255, 255, 0.05);
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: -0.5rem;
    }
    .insight-box {
        background-color: #1e40af;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #e0f2fe;
    }
    .success-box {
        background-color: #14532d;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #d1fae5;
    }
    .warning-box {
        background-color: #78350f;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #fef3c7;
    }
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #2563eb;
    }
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #334155;
        background-color: #0f172a;
        color: #e2e8f0;
    }
    div[data-testid="stDataFrameResizable"] {
        overflow: hidden;
        border-radius: 8px;
    }
    div[data-testid="stForm"] {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: #e2e8f0;
    }
    .stProgress .st-bo {
        background-color: #38bdf8;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .association-rule {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(255, 255, 255, 0.05);
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #334155;
    }
</style>

""", unsafe_allow_html=True)

# Display header
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<div class="main-header">Smart Inventory Management System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2821/2821637.png", width=100)
    st.markdown("### Navigation")
    page = st.radio("", [
        "📊 Dashboard",
        "📈 Demand Forecasting",
        "🔗 Association Rules",
        "🚨 Anomaly Detection",
        "📝 Manage Sales",
    ])
    
    st.markdown("---")
    st.markdown("### System Info")
    st.info(f"Last data update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
# Connect to SQLite DB
@st.cache_resource
def get_connection():
    return sqlite3.connect("sales_data.db", check_same_thread=False)

conn = get_connection()
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS sales (
    Date TEXT,
    StockCode TEXT,
    Quantity INTEGER
)''')

# Load Excel Dataset
@st.cache_data
def load_data():
    df = pd.read_excel("data.xlsx", sheet_name=0)
    df = df.dropna(subset=['CustomerID'])                   # Drops rows where the 'CustomerID' column has missing values
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])   # Converts 'InvoiceDate' column to datetime format
    return df

data = load_data()

@st.cache_resource
def insert_historical_data_once():
    cursor = conn.cursor()
    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM sales")
    count = cursor.fetchone()[0]                           # Fetch the count of rows in the sales table
    
    if count == 0:  # Only insert if table is empty
        st.info("Initializing database with historical data...")
        historical = data[['InvoiceDate', 'StockCode', 'Quantity']].copy()
        historical['Date'] = historical['InvoiceDate'].dt.strftime('%Y-%m-%d')
        historical = historical[['Date', 'StockCode', 'Quantity']]
        
        # Batch insertion for better performance
        batch_size = 1000
        for i in range(0, len(historical), batch_size):
            batch = historical.iloc[i:i+batch_size]
            records = batch.to_records(index=False)
            cursor.executemany(
                "INSERT INTO sales (Date, StockCode, Quantity) VALUES (?, ?, ?)",
                list(records)
            )
        conn.commit()
        st.success("Database initialized successfully!")

# Only initialize DB once
if page == "📊 Dashboard":
    insert_historical_data_once()

# Main page content based on sidebar selection
if page == "📊 Dashboard":
    st.markdown('<div class="sub-header">Overview Dashboard</div>', unsafe_allow_html=True)
    
    # 4 Key metrics - Total Units Sold, Unique Products, Unique Customers, Average Order Size
    col1, col2, col3, col4 = st.columns(4)
    
    # Total sales
    total_sales = data['Quantity'].sum()
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_sales:,.0f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Units Sold</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Unique products
    unique_products = data['StockCode'].nunique()
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{unique_products:,.0f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Unique Products</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Unique customers
    unique_customers = data['CustomerID'].nunique()
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{unique_customers:,.0f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Unique Customers</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Average order size
    avg_order = data.groupby('InvoiceNo')['Quantity'].sum().mean()
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_order:.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg. Order Size</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sales trends
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Sales Trends")
    
    # Prepare time series data / total quantity sold per day
    daily_sales = data.groupby(data['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()  # Group data by date and sum the quantity sold each day
    daily_sales.columns = ['Date', 'Quantity']
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    
    # Create interactive time series chart with Plotly
    fig = px.line(
        daily_sales, 
        x='Date', 
        y='Quantity',
        title='Daily Sales Volume',
        template='plotly_white'
    )
    fig.update_traces(line=dict(color='#4338CA', width=3))
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Units Sold',
        hovermode='x unified',
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Product insights section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Top-Selling Products")
        
        # Group by product, then sum its quantities and return top - selling products
        top_products = data.groupby('Description')['Quantity'].sum().reset_index().sort_values('Quantity', ascending=False).head(10)
        
        fig = px.bar(
            top_products,
            x='Quantity',
            y='Description',
            orientation='h',
            title='Best Performers',
            color_discrete_sequence=['#4F46E5'],
            template='plotly_white',
            text='Quantity'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Sales by Country")
        
        # Group by Country, then sum its quantities and return sales by country
        country_sales = data.groupby('Country')['Quantity'].sum().reset_index().sort_values('Quantity', ascending=False)
        
        fig = px.pie(
            country_sales, 
            values='Quantity', 
            names='Country',
            title='Geographic Distribution',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template='plotly_white'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample data (in collapsible section)
    with st.expander("View Sample Data"):
        st.dataframe(data.head(10), use_container_width=True)

elif page == "📈 Demand Forecasting":
    st.markdown('<div class="sub-header">Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Predict future sales using LSTM neural networks trained on historical data.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        product_list = data['StockCode'].unique()
        product_descriptions = data[['StockCode', 'Description']].drop_duplicates()
        product_dict = {f"{row['StockCode']} - {row['Description'][:30]}...": row['StockCode'] 
                         for _, row in product_descriptions.iterrows()}
        
        selected_product_label = st.selectbox("Select a Product for Forecasting", list(product_dict.keys()))
        selected_product = product_dict[selected_product_label]
    
    with col2:
        forecast_days = st.number_input("Forecast Days", min_value=1, max_value=30, value=7)
    
    # Progress indicator for model training
    progress_placeholder = st.empty()
    
    # Filter and aggregate
    forecast_df = data[data['StockCode'] == selected_product]
    daily_qty = forecast_df.groupby(forecast_df['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()
    daily_qty.columns = ['Date', 'Quantity']
    
    if len(daily_qty) < 20:
        st.markdown('<div class="warning-box">⚠ Not enough data for this product to build a reliable forecast model.</div>', unsafe_allow_html=True)
    else:
        with progress_placeholder:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preparing data...")
            progress_bar.progress(10)
            time.sleep(0.3)  # Simulating work
            
            daily_qty['Date'] = pd.to_datetime(daily_qty['Date'])
            daily_qty.set_index('Date', inplace=True)
            
            status_text.text("Scaling data...")
            progress_bar.progress(20)
            time.sleep(0.3)  # Simulating work
            
            scaler = MinMaxScaler()
            qty_scaled = scaler.fit_transform(daily_qty[['Quantity']])
            
            status_text.text("Building LSTM model...")
            progress_bar.progress(40)
            time.sleep(0.3)  # Simulating work
            
            generator = TimeseriesGenerator(qty_scaled, qty_scaled, length=10, batch_size=1)
            
            model = Sequential()
            model.add(LSTM(64, activation='relu', input_shape=(10, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            
            status_text.text("Training model... (This might take a moment)")
            progress_bar.progress(60)
            
            model.fit(generator, epochs=10, verbose=0)
            
            status_text.text("Generating forecasts...")
            progress_bar.progress(80)
            time.sleep(0.3)  # Simulating work
            
            # Generate multiple days forecast
            last_sequence = qty_scaled[-10:].reshape((1, 10, 1))
            forecast_values = []
            
            for _ in range(forecast_days):
                next_pred = model.predict(last_sequence)[0]
                forecast_values.append(next_pred[0])
                last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)
            
            forecast_values_rescaled = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1)).flatten()
            forecast_values_rounded = [max(0, round(x)) for x in forecast_values_rescaled]
            
            # Create forecast dates
            last_date = daily_qty.index[-1]
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            
            progress_bar.progress(100)
            status_text.text("Forecast complete!")
            time.sleep(0.5)  # Simulating work
            progress_placeholder.empty()  # Clear the progress elements
        
        # Show forecasted values
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted Quantity': forecast_values_rounded
        })
        
        # Visualization
        col1, col2 = st.columns([7, 3])
        
        with col1:
            # Combine historical and forecast for visualization
            historical = daily_qty.reset_index().tail(60)
            
            # Create interactive chart
            fig = go.Figure()
            
            # Add historical line
            fig.add_trace(go.Scatter(
                x=historical['Date'],
                y=historical['Quantity'],
                mode='lines',
                name='Historical',
                line=dict(color='#3B82F6', width=2)
            ))
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Forecasted Quantity'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#10B981', width=3, dash='dot'),
                marker=dict(size=8)
            ))
            
            # Add confidence interval (as an example, using +/- 15%)
            upper_bound = [x * 1.15 for x in forecast_df['Forecasted Quantity']]
            lower_bound = [max(0, x * 0.85) for x in forecast_df['Forecasted Quantity']]
            
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Upper Bound'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=lower_bound,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(16, 185, 129, 0.2)',
                showlegend=False,
                name='Lower Bound'
            ))
            
            fig.update_layout(
                title='Historical Data and Forecast',
                xaxis_title='Date',
                yaxis_title='Quantity',
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Summary statistics
            st.markdown(f"""
            <div class="success-box">
                <h4>Forecast Summary</h4>
                <p>Total forecasted demand: <strong>{sum(forecast_values_rounded):,}</strong> units</p>
                <p>Average daily demand: <strong>{sum(forecast_values_rounded)/len(forecast_values_rounded):.1f}</strong> units</p>
                <p>Peak demand day: <strong>{forecast_df.loc[forecast_df['Forecasted Quantity'].idxmax(), 'Date'].strftime('%Y-%m-%d')}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tabular view of forecast
            st.markdown("#### Daily Forecast")
            forecast_display = forecast_df.copy()
            forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(forecast_display, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations based on forecast
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Inventory Recommendations")
    
    if len(daily_qty) >= 20:
        avg_forecast = sum(forecast_values_rounded) / len(forecast_values_rounded)
        historical_avg = daily_qty['Quantity'].mean()
        
        if avg_forecast > historical_avg * 1.2:
            st.markdown('<div class="insight-box">📈 <strong>Stock Up:</strong> Forecasted demand is trending higher than historical average. Consider increasing inventory by 20-30%.</div>', unsafe_allow_html=True)
        elif avg_forecast < historical_avg * 0.8:
            st.markdown('<div class="insight-box">📉 <strong>Reduce Stock:</strong> Forecasted demand is trending lower than historical average. Consider reducing inventory to avoid excess.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">⚖ <strong>Maintain Current Levels:</strong> Forecasted demand is consistent with historical patterns.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# To be Explained by Prabhmeet Singh (2022UCM2305)
elif page == "🔗 Association Rules":
    st.markdown('<div class="sub-header">Association Rule Mining</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Discover which products are frequently purchased together to optimize product placement and cross-selling opportunities.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_support = st.slider("Minimum Support", 0.01, 0.1, 0.02, 0.005)   # Slider Range from 0.01 to 0.1, default is 0.02, step size is 0.005
    with col2:
        min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.0, 0.1)             # Slider Range from 1.0 to 5.0, default is 1.0, step size is 0.1
    with col3:
        metric = st.selectbox("Sorting Metric", ["lift", "confidence", "support"])   # Select box/ DropDown menu for choosing sorting metric for Association rules
    
    # Processing indicator
    with st.spinner("Mining association rules..."):
        # Create a basket view
        basket = (data[data['Quantity'] > 0]
                  .groupby(['InvoiceNo', 'Description'])['Quantity']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('InvoiceNo'))
        
        basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        # Filter products that meet the minimum support threshold
        item_frequencies = basket_sets.sum()
        frequent_items = item_frequencies[item_frequencies >= min_support * len(basket_sets)].index
        basket_sets_filtered = basket_sets[frequent_items]

        # applying Apriori algorithm to find association rules 
        freq_items = apriori(basket_sets_filtered, min_support=min_support, use_colnames=True)
        # freq_items = apriori(basket_sets, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Top Association Rules")
        
        if rules.empty:
            st.warning("No rules found with the current parameters. Try lowering the thresholds.")
        else:
            # top 10 rules based on chosen metric 
            top_rules = rules.sort_values(by=metric, ascending=False).head(10)
            
            for idx, row in top_rules.iterrows():
                antecedents = ', '.join(list(row['antecedents']))
                consequents = ', '.join(list(row['consequents']))
                
                st.markdown(f"""
                <div class="association-rule">
                    <h4>🧩 If customers buy: <span style="color:#4338CA">{antecedents}</span></h4>
                    <h4>👉 They also likely buy: <span style="color:#10B981">{consequents}</span></h4>
                    <div style="margin-top: 10px; display: flex; gap: 20px;">
                        <div>
                            <span style="color:#6B7280; font-size: 0.9rem;">Support</span><br>
                            <span style="font-weight: 600; font-size: 1.2rem;">{row['support']:.3f}</span>
                        </div>
                        <div>
                            <span style="color:#6B7280; font-size: 0.9rem;">Confidence</span><br>
                            <span style="font-weight: 600; font-size: 1.2rem;">{row['confidence']:.3f}</span>
                        </div>
                        <div>
                            <span style="color:#6B7280; font-size: 0.9rem;">Lift</span><br>
                            <span style="font-weight: 600; font-size: 1.2rem;">{row['lift']:.3f}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Top Frequent Itemsets")
        
        if freq_items.empty:
            st.warning("No frequent itemsets found with the current parameters.")
        else:
            # Sort and display top 10 frequent itemsets
            top_itemsets = freq_items.sort_values(by='support', ascending=False).head(10)
            
            for idx, row in top_itemsets.iterrows():
                items = ', '.join(list(row['itemsets']))
                st.markdown(f"""
                <div style="background-color: grey; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <div style="color:white; font-size: 0.8rem;">Items</div>
                    <div style="font-weight: 500;">{items}</div>
                    <div style="color:white; font-size: 0.8rem; margin-top: 5px;">Support</div>
                    <div style="font-weight: 600; color:#4338CA;">{row['support']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🚨 Anomaly Detection":
    st.markdown('<div class="sub-header">Anomaly Detection in Sales</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Identify unusual sales patterns that may indicate issues or opportunities in your inventory.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    # Filter options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get unique products with descriptions
        product_descriptions = data[['StockCode', 'Description']].drop_duplicates()
        product_dict = {f"{row['StockCode']} - {row['Description'][:30]}...": row['StockCode'] 
                     for _, row in product_descriptions.iterrows()}
        
        selected_product_label = st.selectbox("Select a Product for Anomaly Detection", list(product_dict.keys()))
        selected_anomaly_product = product_dict[selected_product_label]
    
    with col2:
        contamination = st.slider("Anomaly Sensitivity", 0.01, 0.15, 0.05, 0.01,
                               help="Lower values detect fewer but more significant anomalies")
    
    # Prepare data
    daily_sales = data.groupby([data['InvoiceDate'].dt.date, 'StockCode'])['Quantity'].sum().reset_index()
    daily_sales.columns = ['Date', 'StockCode', 'Quantity']
    
    # Filter to selected product
    product_df = daily_sales[daily_sales['StockCode'] == selected_anomaly_product].copy()
    
    if len(product_df) < 5:
        st.warning("Not enough data points for this product to perform anomaly detection.")
    else:
        # Run anomaly detection
        product_df['Date'] = pd.to_datetime(product_df['Date'])
        model_iforest = IsolationForest(contamination=contamination, random_state=42)
        product_df['anomaly'] = model_iforest.fit_predict(product_df[['Quantity']])
        
        # Get anomalies
        anomalies = product_df[product_df['anomaly'] == -1].copy()
        
        # Create interactive visualization
        fig = go.Figure()
        
        # Add normal points
        fig.add_trace(go.Scatter(
            x=product_df[product_df['anomaly'] == 1]['Date'],
            y=product_df[product_df['anomaly'] == 1]['Quantity'],
            mode='lines+markers',
            name='Normal',
            line=dict(color='#3B82F6', width=2),
            marker=dict(size=6)
        ))
        
        # Add anomaly points
        fig.add_trace(go.Scatter(
            x=anomalies['Date'],
            y=anomalies['Quantity'],
            mode='markers',
            name='Anomaly',
            marker=dict(
                color='#DC2626',
                size=12,
                symbol='circle',
                line=dict(
                    color='#7F1D1D',
                    width=2
                )
            )
        ))
        
        fig.update_layout(
            title=f"Sales Pattern with Detected Anomalies",
            xaxis_title="Date",
            yaxis_title="Quantity",
            hovermode="closest",
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Table of anomalies
        if not anomalies.empty:
            st.markdown("#### Detected Anomalies")
            
            # Format for display
            anomalies_display = anomalies.copy()
            anomalies_display['Date'] = anomalies_display['Date'].dt.strftime('%Y-%m-%d')
            anomalies_display['Average Quantity'] = product_df['Quantity'].mean()
            anomalies_display['Deviation (%)'] = ((anomalies_display['Quantity'] - anomalies_display['Average Quantity']) 
                                             / anomalies_display['Average Quantity'] * 100).round(1)
            anomalies_display = anomalies_display[['Date', 'Quantity', 'Average Quantity', 'Deviation (%)']]
            
            st.dataframe(anomalies_display, use_container_width=True)
            
            # Analysis
            high_anomalies = anomalies[anomalies['Quantity'] > product_df['Quantity'].mean()]
            low_anomalies = anomalies[anomalies['Quantity'] < product_df['Quantity'].mean()]
            
            if not high_anomalies.empty:
                st.markdown(f"""
                <div class="insight-box">
                    <h4>📈 Unusual Demand Spikes</h4>
                    <p>{len(high_anomalies)} unusual spike(s) in demand detected, averaging 
                    {high_anomalies['Quantity'].mean():.1f} units compared to normal {product_df['Quantity'].mean():.1f} units.</p>
                    <p>Consider investigating these dates for special events or promotions that drove higher sales.</p>
                </div>
                """, unsafe_allow_html=True)
            
            if not low_anomalies.empty:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>📉 Unusual Low Sales</h4>
                    <p>{len(low_anomalies)} unusual drop(s) in demand detected, averaging 
                    {low_anomalies['Quantity'].mean():.1f} units compared to normal {product_df['Quantity'].mean():.1f} units.</p>
                    <p>Check for potential inventory stockouts or issues during these periods.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No anomalies detected for this product with the current sensitivity setting.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Systemwide anomaly detection
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### System-wide Anomaly Detection")
    
    # Aggregate data by date across all products
    system_sales = data.groupby(data['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()
    system_sales.columns = ['Date', 'Quantity']
    system_sales['Date'] = pd.to_datetime(system_sales['Date'])
    
    if len(system_sales) >= 30:  # Ensure enough data for meaningful detection
        # Apply anomaly detection
        system_model = IsolationForest(contamination=0.05, random_state=42)
        system_sales['anomaly'] = system_model.fit_predict(system_sales[['Quantity']])
        
        # Get system anomalies
        system_anomalies = system_sales[system_sales['anomaly'] == -1]
        
        # Create visualization
        fig = go.Figure()
        
        # Add normal points
        fig.add_trace(go.Scatter(
            x=system_sales[system_sales['anomaly'] == 1]['Date'],
            y=system_sales[system_sales['anomaly'] == 1]['Quantity'],
            mode='lines+markers',
            name='Normal',
            line=dict(color='#3B82F6', width=2),
            marker=dict(size=5)
        ))
        
        # Add anomaly points
        fig.add_trace(go.Scatter(
            x=system_anomalies['Date'],
            y=system_anomalies['Quantity'],
            mode='markers',
            name='Anomaly',
            marker=dict(
                color='#DC2626',
                size=12,
                symbol='circle',
                line=dict(
                    color='#7F1D1D',
                    width=2
                )
            )
        ))
        
        fig.update_layout(
            title="Total Sales Volume with System-wide Anomalies",
            xaxis_title="Date",
            yaxis_title="Total Quantity",
            hovermode="closest",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if not system_anomalies.empty:
            # Show system anomalies summary
            st.markdown(f"Detected {len(system_anomalies)} system-wide anomalies across all products.")
            
            # Format table
            system_anomalies_display = system_anomalies.copy()
            system_anomalies_display['Date'] = system_anomalies_display['Date'].dt.strftime('%Y-%m-%d')
            system_anomalies_display['Average Daily Sales'] = system_sales['Quantity'].mean()
            system_anomalies_display['Deviation (%)'] = ((system_anomalies_display['Quantity'] - system_anomalies_display['Average Daily Sales']) 
                                                  / system_anomalies_display['Average Daily Sales'] * 100).round(1)
            
            # Show table
            st.dataframe(system_anomalies_display[['Date', 'Quantity', 'Average Daily Sales', 'Deviation (%)']], 
                      use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📝 Manage Sales":
    st.markdown('<div class="sub-header">Sales Data Management</div>', unsafe_allow_html=True)
    
    # Add new sales record
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Add New Sale Record")
    
    with st.form("add_sale"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            add_date = st.date_input("Sale Date")
        
        with col2:
            # Get unique products with descriptions for better selection
            product_descriptions = data[['StockCode', 'Description']].drop_duplicates()
            product_dict = {f"{row['StockCode']} - {row['Description'][:30]}...": row['StockCode'] 
                         for _, row in product_descriptions.iterrows()}
            
            selected_product_label = st.selectbox("Product", list(product_dict.keys()))
            add_product = product_dict[selected_product_label]
        
        with col3:
            add_quantity = st.number_input("Quantity", min_value=1, step=1)
        
        submit_btn = st.form_submit_button("Add Sale Record")
        
        if submit_btn:
            cursor.execute("INSERT INTO sales (Date, StockCode, Quantity) VALUES (?, ?, ?)",
                       (add_date.strftime('%Y-%m-%d'), add_product, int(add_quantity)))
            conn.commit()
            st.markdown('<div class="success-box">✅ Sale record successfully added to database!</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # View and filter database records
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Database Records")
    
    # Query parameters
    col1, col2 = st.columns(2)
    
    with col1:
        days_filter = st.slider("Show records from last N days", 1, 365, 30)
    
    with col2:
        filter_product = st.checkbox("Filter by product")
        if filter_product:
            filter_product_label = st.selectbox("Select product", list(product_dict.keys()), key="filter_product")
            filter_product_code = product_dict[filter_product_label]
    
    # Build query
    filter_date = (datetime.now() - timedelta(days=days_filter)).strftime('%Y-%m-%d')
    
    if filter_product:
        query = f"SELECT * FROM sales WHERE Date >= '{filter_date}' AND StockCode = '{filter_product_code}' ORDER BY Date DESC"
    else:
        query = f"SELECT * FROM sales WHERE Date >= '{filter_date}' ORDER BY Date DESC"
    
    # Execute query
    df_sales = pd.read_sql_query(query, conn)
    
    # Display records
    if not df_sales.empty:
        # Add product descriptions
        product_lookup = dict(zip(data['StockCode'], data['Description']))
        df_sales['Description'] = df_sales['StockCode'].map(lambda x: product_lookup.get(x, 'Unknown'))
        
        # Format for display
        df_display = df_sales.copy()
        df_display = df_display[['Date', 'StockCode', 'Description', 'Quantity']]
        
        st.dataframe(df_display, use_container_width=True)
        
        # Summary statistics
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 Summary</h4>
            <p>Total Records: {len(df_sales)}</p>
            <p>Total Quantity: {df_sales['Quantity'].sum():,} units</p>
            <p>Unique Products: {df_sales['StockCode'].nunique()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Export option
        export_col1, export_col2 = st.columns([3, 1])
        with export_col1:
            export_format = st.selectbox("Export Format", ["CSV", "Excel"])
        with export_col2:
            if st.button("Export Data"):
                if export_format == "CSV":
                    df_sales.to_csv("sales_export.csv", index=False)
                    st.success("Data exported as CSV!")
                else:
                    df_sales.to_excel("sales_export.xlsx", index=False)
                    st.success("Data exported as Excel!")
    else:
        st.info("No records found with the selected filters.")
    
    st.markdown('</div>', unsafe_allow_html=True)