# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Price Pattern Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'show_raw_data' not in st.session_state:
    st.session_state.show_raw_data = False
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
if 'features_engineered' not in st.session_state:
    st.session_state.features_engineered = False

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #1565C0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .stSidebar {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTitle {
        color: #1E88E5;
        font-size: 2.5em;
        text-align: center;
        padding: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .stSuccess {
        background-color: #1B5E20;
        color: #E8F5E9;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    .stInfo {
        background-color: #0D47A1;
        color: #E3F2FD;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
    }
    .stError {
        background-color: #B71C1C;
        color: #FFEBEE;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #F44336;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <h2>📈 Market Data</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload dataset
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    st.markdown("---")
    
    # Fetch stock data
    st.markdown("### 📊 Fetch Stock Data")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    date_range = st.date_input("Date Range", [])
    fetch_data = st.button("🔍 Fetch Data", use_container_width=True)

# Add this function at the top of the file, after imports
def standardize_column_names(df):
    """Standardize column names to ensure required columns are present."""
    # Debug print
    print("Original columns:", df.columns)
    
    # Handle MultiIndex columns (from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex columns
        df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in df.columns]
        print("After flattening MultiIndex:", df.columns)
    
    # Create a mapping of possible column name variations
    column_mapping = {
        'close': 'Close',
        'closing': 'Close',
        'closing_price': 'Close',
        'adj close': 'Close',
        'adj_close': 'Close',
        'adjclose': 'Close',
        'close_aapl': 'Close',
        'close_': 'Close',
        'high': 'High',
        'highest': 'High',
        'high_price': 'High',
        'high_aapl': 'High',
        'high_': 'High',
        'low': 'Low',
        'lowest': 'Low',
        'low_price': 'Low',
        'low_aapl': 'Low',
        'low_': 'Low',
        'open': 'Open',
        'opening': 'Open',
        'opening_price': 'Open',
        'open_aapl': 'Open',
        'open_': 'Open',
        'volume': 'Volume',
        'vol': 'Volume',
        'volume_aapl': 'Volume',
        'volume_': 'Volume',
        'date': 'Date',
        'timestamp': 'Date',
        'time': 'Date',
        'date_': 'Date'
    }
    
    # Convert all column names to lowercase for matching
    df.columns = [str(col).lower().strip() for col in df.columns]
    print("After lowercase conversion:", df.columns)
    
    # Rename columns according to mapping
    df = df.rename(columns=column_mapping)
    print("After renaming:", df.columns)
    
    return df

# Main content
if not st.session_state.data_loaded:
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            # Standardize column names
            st.session_state.df = standardize_column_names(st.session_state.df)
            
            # Check if we have the required columns
            required_columns = ['Close', 'High', 'Low']
            missing_columns = [col for col in required_columns if col not in st.session_state.df.columns]
            if missing_columns:
                st.error(f"❌ Missing required columns in uploaded file: {missing_columns}")
                st.error("Please ensure your CSV file contains columns for closing, high, and low prices")
                st.session_state.data_loaded = False
            else:
                st.session_state.data_loaded = True
                st.success("✅ Dataset uploaded successfully!")
                st.write("Available columns:", st.session_state.df.columns.tolist())
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.session_state.data_loaded = False
    elif fetch_data and len(date_range) == 2:
        with st.spinner('Fetching market data...'):
            try:
                # Download data
                data = yf.download(ticker, start=date_range[0], end=date_range[1])
                
                # Debug print
                st.write("Raw data columns:", data.columns.tolist())
                
                # Reset index to make Date a column
                data.reset_index(inplace=True)
                
                # Standardize column names
                data = standardize_column_names(data)
                
                # Debug print
                st.write("After standardization:", data.columns.tolist())
                
                # Store in session state
                st.session_state.df = data
                st.session_state.data_loaded = True
                
                st.success(f"✅ {ticker} data fetched successfully!")
                st.write("Available columns:", st.session_state.df.columns.tolist())
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>📊 Price Pattern Analyzer</h1>
            <p style='font-size: 1.2em;'>Advanced Market Regime Analysis Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://media.giphy.com/media/3o7buirYof8fILTkU0/giphy.gif' width='400'>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h2>🚀 Get Started</h2>
            <p>Upload your dataset or fetch real-time market data to begin your analysis!</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("### 📈 Analysis Pipeline")
    
    # Show Raw Data
    if st.button("🔍 View Market Data", use_container_width=True):
        st.session_state.show_raw_data = True
    
    if st.session_state.show_raw_data:
        st.markdown("### Market Data Overview")
        st.dataframe(st.session_state.df.head().style.set_properties(**{
            'background-color': '#1E1E1E',
            'color': '#FAFAFA',
            'border': '1px solid #333'
        }))
        
        # Show price chart
        if 'Close' in st.session_state.df.columns:
            try:
                fig = go.Figure()
                
                # Add candlestick chart if we have all required columns
                if all(col in st.session_state.df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    fig.add_trace(go.Candlestick(
                        x=st.session_state.df['Date'] if 'Date' in st.session_state.df.columns else st.session_state.df.index,
                        open=st.session_state.df['Open'],
                        high=st.session_state.df['High'],
                        low=st.session_state.df['Low'],
                        close=st.session_state.df['Close'],
                        name='Price'
                    ))
                else:
                    # If we don't have all OHLC data, just plot the close price
                    fig.add_trace(go.Scatter(
                        x=st.session_state.df['Date'] if 'Date' in st.session_state.df.columns else st.session_state.df.index,
                        y=st.session_state.df['Close'],
                        mode='lines',
                        name='Close Price'
                    ))
                
                fig.update_layout(
                    title=f'{ticker if "ticker" in locals() else "Price"} Chart',
                    yaxis_title='Price',
                    template='plotly_dark',
                    xaxis_rangeslider_visible=False,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating price chart: {str(e)}")
                st.write("Available columns:", st.session_state.df.columns.tolist())

    # Preprocess Data
    if st.button("⚙️ Clean Data", use_container_width=True):
        if not st.session_state.data_loaded:
            st.error("❌ Please load data first!")
            st.stop()
        
        try:
            # Get data from session state
            df = st.session_state['df'].copy()
            
            # Debug information
            st.write("Before cleaning - Available columns:", df.columns.tolist())
            
            # Standardize column names again to be safe
            df = standardize_column_names(df)
            
            # Debug information
            st.write("After standardization - Available columns:", df.columns.tolist())
            
            # Check if required columns exist
            required_columns = ['Close', 'High', 'Low']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                st.error("Please ensure your data contains columns for closing, high, and low prices")
                st.error(f"Current columns: {df.columns.tolist()}")
                st.stop()
            
            # Clean the data
            original_rows = len(df)
            df = df.dropna()
            
            # Update session state
            st.session_state['df'] = df
            st.session_state.preprocessed = True
            
            st.success(f"✅ Data cleaned successfully! Removed {original_rows - len(df)} rows with missing values.")
            st.write("After cleaning - Available columns:", df.columns.tolist())
            
        except Exception as e:
            st.error(f"Error during data cleaning: {str(e)}")
            st.session_state.preprocessed = False

    # Show data quality metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(st.session_state.df))
    with col2:
        st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
    with col3:
        st.metric("Data Types", len(st.session_state.df.dtypes.unique()))

    # Feature Engineering
    if st.button("2️⃣ Engineer Features", key="engineer_features"):
        try:
            if not st.session_state.preprocessed:
                st.error("❌ Please clean the data first!")
                st.stop()
                
            # Get data from session state
            df = st.session_state['df'].copy()
            
            # Standardize column names again to be safe
            df = standardize_column_names(df)
            
            # Debug information
            st.write("Current dataframe columns:", df.columns.tolist())
            
            # Check if required columns exist
            required_columns = ['Close', 'High', 'Low']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                st.error("Please ensure your data contains columns for closing, high, and low prices")
                st.stop()
            
            # Flatten MultiIndex columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in df.columns]
                st.write("After flattening MultiIndex, columns are:", df.columns.tolist())
            
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Returns_5d'] = df['Close'].pct_change(periods=5)
            df['Returns_20d'] = df['Close'].pct_change(periods=20)
            
            # Calculate volatility (20-day rolling standard deviation of returns)
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Calculate moving averages
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # Calculate price to moving average ratios
            df['Price_to_MA20'] = df['Close'] / df['MA20']
            df['Price_to_MA50'] = df['Close'] / df['MA50']
            
            # Calculate momentum indicators
            df['Momentum'] = df['Close'].pct_change(periods=10)
            
            # Calculate RSI
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            df['RSI'] = calculate_rsi(df['Close'])
            
            # Calculate Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_std'] = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
            df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
            
            # Calculate price ranges
            df['Daily_Range'] = df['High'] - df['Low']
            df['Range_to_Close'] = df['Daily_Range'] / df['Close']
            
            # Add volume analysis if available
            if 'Volume' in df.columns:
                df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
            
            # Drop NaN values
            df = df.dropna()
            
            # Update session state
            st.session_state['df'] = df
            st.session_state['features_engineered'] = True
            
            st.success("✅ Features engineered successfully!")
            
            # Show feature summary
            st.subheader("Feature Summary")
            feature_summary = pd.DataFrame({
                'Feature': df.columns,
                'Mean': df.mean(),
                'Std': df.std(),
                'Min': df.min(),
                'Max': df.max()
            })
            st.dataframe(feature_summary)
            
            # Visualize feature distributions
            st.subheader("Feature Distributions")
            selected_features = st.multiselect(
                "Select features to visualize",
                df.columns,
                default=['Returns', 'Volatility', 'Price_to_MA20', 'Momentum']
            )
            
            if selected_features:
                fig = go.Figure()
                for feature in selected_features:
                    fig.add_trace(go.Histogram(
                        x=df[feature],
                        name=feature,
                        opacity=0.7
                    ))
                fig.update_layout(
                    title="Feature Distributions",
                    xaxis_title="Value",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation matrix
            st.subheader("Feature Correlations")
            corr_matrix = df.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            fig.update_layout(
                title="Feature Correlation Matrix",
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during feature engineering: {str(e)}")
            st.error("Please ensure you have loaded and preprocessed the data first.")
            st.session_state['features_engineered'] = False

    # Clustering Analysis
    if st.button("🔮 Analyze Patterns", use_container_width=True):
        if not st.session_state.get('features_engineered', False):
            st.error("❌ Please create features first!")
            st.stop()
            
        try:
            # Prepare features for clustering
            feature_cols = ['Returns', 'Returns_5d', 'Returns_20d', 'Volatility', 
                          'Price_to_MA20']
            if 'Volume' in st.session_state.df.columns:
                feature_cols.append('Volume_Ratio')
            
            # Check if all required features exist
            missing_features = [col for col in feature_cols if col not in st.session_state.df.columns]
            if missing_features:
                st.error(f"Missing required features: {missing_features}")
                st.error("Please run feature engineering again.")
                st.stop()
            
            X = st.session_state.df[feature_cols].dropna()
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.session_state.scaler = scaler
            
            # Find optimal number of clusters
            max_clusters = min(10, len(X_scaled))
            silhouette_scores = []
            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(score)
            
            # Plot silhouette scores
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(2, max_clusters + 1)),
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score'
            ))
            fig.update_layout(
                title='Optimal Number of Clusters',
                xaxis_title='Number of Clusters',
                yaxis_title='Silhouette Score',
                template='plotly_dark'
            )
            st.plotly_chart(fig)
            
            # Get optimal number of clusters
            optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
            
            # Perform clustering with optimal number of clusters
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            st.session_state.model = kmeans
            
            # Add cluster labels to dataframe
            st.session_state.df.loc[X.index, 'Cluster'] = cluster_labels
            
            st.success(f"🎯 Identified {optimal_clusters} distinct price patterns!")
            
            # Show cluster characteristics
            st.markdown("### Pattern Characteristics")
            cluster_stats = st.session_state.df.groupby('Cluster')[feature_cols].mean()
            st.dataframe(cluster_stats.style.background_gradient(cmap='RdYlBu'))
            
            # Visualize clusters
            fig = go.Figure()
            for cluster in range(optimal_clusters):
                cluster_data = st.session_state.df[st.session_state.df['Cluster'] == cluster]
                fig.add_trace(go.Scatter(
                    x=cluster_data['Date'] if 'Date' in cluster_data.columns else cluster_data.index,
                    y=cluster_data['Close'],
                    mode='lines',
                    name=f'Pattern {cluster + 1}',
                    line=dict(width=2)
                ))
            fig.update_layout(
                title='Price Patterns Over Time',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark'
            )
            st.plotly_chart(fig)
            
            # Show pattern transitions
            st.markdown("### Pattern Transitions")
            transitions = st.session_state.df['Cluster'].diff().abs()
            transition_dates = st.session_state.df[transitions > 0]['Date'] if 'Date' in st.session_state.df.columns else st.session_state.df[transitions > 0].index
            st.write(f"Number of pattern transitions: {len(transition_dates)}")
            
            if len(transition_dates) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.df['Date'] if 'Date' in st.session_state.df.columns else st.session_state.df.index,
                    y=st.session_state.df['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#1E88E5')
                ))
                fig.add_trace(go.Scatter(
                    x=transition_dates,
                    y=st.session_state.df.loc[transition_dates.index, 'Close'],
                    mode='markers',
                    name='Pattern Change',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='diamond'
                    )
                ))
                fig.update_layout(
                    title='Pattern Transitions',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_dark'
                )
                st.plotly_chart(fig)
                
        except Exception as e:
            st.error(f"Error during pattern analysis: {str(e)}")
            st.error("Please ensure you have properly engineered features first.")

    # Download Results
    if st.button("📥 Download Results", use_container_width=True):
        if st.session_state.model is None:
            st.error("❌ Please analyze patterns first!")
        else:
            # Prepare results
            results_df = st.session_state.df.copy()
            
            # Download button
            st.download_button(
                "📊 Download Analysis",
                results_df.to_csv(index=False),
                file_name="pattern_analysis.csv",
                mime="text/csv"
            )
            st.success("✅ Results ready for download!")
