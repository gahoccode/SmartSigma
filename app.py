import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import HRPOpt, expected_returns, risk_models, CLA
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return

st.set_page_config(page_title="SmartSigma Portfolio Optimizer", layout="wide")

def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
                df.set_index('Date', inplace=True)
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty.")
            return None
        except pd.errors.ParserError:
            st.error("Error parsing the uploaded file.")
            return None
    else:
        # Default data path
        default_file = "myport2.csv"
        try:
            df = pd.read_csv(default_file)
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            df.set_index('Date', inplace=True)
        except FileNotFoundError:
            st.error("Default data file not found.")
            return None
        except pd.errors.EmptyDataError:
            st.error("Default data file is empty.")
            return None
        except pd.errors.ParserError:
            st.error("Error parsing default data file.")
            return None
    
    # Handle missing values according to rules.md
    prices = df.dropna()
    return prices

def optimize_hrp(prices):
    """Hierarchical Risk Parity optimization"""
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Create HRP optimizer
    hrp = HRPOpt(returns)
    
    # Get optimal weights
    weights = hrp.optimize()
    
    # Get portfolio performance
    perf = hrp.portfolio_performance(verbose=True)
    
    return weights, perf

def optimize_mvo(prices, weight_bounds=(0, 1)):
    """Mean-Variance Optimization with both Maximum Sharpe and Minimum Volatility"""
    # Calculate expected returns and covariance
    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    
    # Create efficient frontier object for maximum Sharpe
    ef_sharpe = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    
    # Get maximum Sharpe portfolio
    max_sharpe_weights = ef_sharpe.max_sharpe()
    max_sharpe_weights = ef_sharpe.clean_weights()
    max_sharpe_perf = ef_sharpe.portfolio_performance(verbose=True)
    
    # Create new efficient frontier object for minimum volatility
    ef_vol = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    
    # Get minimum volatility portfolio
    min_vol_weights = ef_vol.min_volatility()
    min_vol_weights = ef_vol.clean_weights()
    min_vol_perf = ef_vol.portfolio_performance(verbose=True)
    
    return max_sharpe_weights, max_sharpe_perf, (mu, S, weight_bounds, min_vol_weights, min_vol_perf)

def plot_portfolio_weights(weights):
    # Sort weights by value for better visualization
    weights_sorted = pd.Series(weights).sort_values(ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            x=weights_sorted.values,
            y=weights_sorted.index,
            orientation='h'
        )
    ])
    fig.update_layout(
        title="Portfolio Weights",
        xaxis_title="Weight",
        yaxis_title="Asset",
        yaxis_tickformat='.2%',
        height=400,
        margin=dict(l=200)  # Add margin for asset names
    )
    return fig

def plot_portfolio_pie(weights):
    """Create a pie chart of portfolio weights"""
    # Create a figure
    fig = go.Figure(data=[go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        textinfo='label+percent',
        hovertemplate="Asset: %{label}<br>Weight: %{value:.2%}<br><extra></extra>",
    )])
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(t=0, b=0, l=0, r=0),
    )
    
    return fig

def plot_cumulative_returns(prices):
    # Handle returns according to rules.md
    returns = prices.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod()
    
    fig = go.Figure()
    for col in cumulative_returns.columns:
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[col],
            name=col,
            mode='lines'
        ))
    
    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        yaxis_tickformat='.2%',
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    return fig

def plot_efficient_frontier(mu, S, weight_bounds, min_vol_weights, min_vol_perf):
    """Plot the efficient frontier using CLA with optimal portfolios"""
    # Calculate efficient frontier points
    cla = CLA(mu, S, weight_bounds=weight_bounds)
    returns, risks, _ = cla.efficient_frontier(points=100)
    
    # Create the efficient frontier plot
    fig = go.Figure()
    
    # Plot efficient frontier
    fig.add_trace(go.Scatter(
        x=risks,
        y=returns,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=2)
    ))
    
    # Add individual assets
    asset_returns = mu
    asset_risks = np.sqrt(np.diag(S))
    
    fig.add_trace(go.Scatter(
        x=asset_risks,
        y=asset_returns,
        mode='markers',
        name='Individual Assets',
        marker=dict(
            size=10,
            color='red',
        ),
        text=mu.index,
        hovertemplate=
        "Asset: %{text}<br>" +
        "Return: %{y:.2%}<br>" +
        "Risk: %{x:.2%}<br>" +
        "<extra></extra>"
    ))
    
    # Add minimum volatility portfolio point
    min_vol_ret = min_vol_perf[0]
    min_vol_risk = min_vol_perf[1]
    
    fig.add_trace(go.Scatter(
        x=[min_vol_risk],
        y=[min_vol_ret],
        mode='markers',
        name='Minimum Volatility',
        marker=dict(
            size=15,
            color='green',
            symbol='star'
        ),
        hovertemplate=
        "Portfolio: Minimum Volatility<br>" +
        "Return: %{y:.2%}<br>" +
        "Risk: %{x:.2%}<br>" +
        "Sharpe: " + f"{min_vol_perf[2]:.2f}" +
        "<extra></extra>"
    ))
    
    # Add maximum Sharpe ratio portfolio point
    cla.max_sharpe()
    max_sharpe_ret, max_sharpe_risk, max_sharpe_ratio = cla.portfolio_performance()
    
    fig.add_trace(go.Scatter(
        x=[max_sharpe_risk],
        y=[max_sharpe_ret],
        mode='markers',
        name='Maximum Sharpe',
        marker=dict(
            size=15,
            color='yellow',
            symbol='star'
        ),
        hovertemplate=
        "Portfolio: Maximum Sharpe<br>" +
        "Return: %{y:.2%}<br>" +
        "Risk: %{x:.2%}<br>" +
        "Sharpe: " + f"{max_sharpe_ratio:.2f}" +
        "<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title="Efficient Frontier with Optimal Portfolios",
        xaxis_title="Expected Risk (Standard Deviation)",
        yaxis_title="Expected Return",
        xaxis_tickformat='.2%',
        yaxis_tickformat='.2%',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_covariance_heatmap(prices, plot_correlation=False):
    """Create an interactive heatmap of the covariance/correlation matrix"""
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Calculate covariance matrix
    if plot_correlation:
        matrix = returns.corr()
        title = "Correlation Matrix"
    else:
        matrix = returns.cov()
        title = "Covariance Matrix"
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.index,
        y=matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(matrix.values, 4),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate="Asset 1: %{y}<br>Asset 2: %{x}<br>Value: %{z:.4f}<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        width=800,
        xaxis_title="Assets",
        yaxis_title="Assets",
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'},
    )
    
    return fig

def display_portfolio_metrics(perf):
    expected_return, volatility, sharpe = perf
    metrics = pd.DataFrame({
        'Metric': ['Expected Return', 'Annual Volatility', 'Sharpe Ratio'],
        'Value': [expected_return, volatility, sharpe]
    })
    st.dataframe(metrics.style.format({
        'Value': lambda x: f'{x:.2%}' if isinstance(x, float) else x
    }))

def main():
    st.title("SmartSigma Portfolio Optimizer")
    
    st.sidebar.header("Settings")
    optimization_method = st.sidebar.selectbox(
        "Select Optimization Method",
        ["HRP (Hierarchical Risk Parity)", "MVO (Mean-Variance Optimization)"]
    )
    
    # Add risk-free rate note
    st.sidebar.markdown("""
    **Note:** All Sharpe ratio calculations assume a risk-free rate of 2%.
    This is used for both Maximum Sharpe and Minimum Volatility optimizations.
    """)
    
    # Add weight bounds option for MVO
    if "MVO" in optimization_method:
        allow_shorting = st.sidebar.checkbox(
            "Allow Shorting",
            help="If checked, allows negative weights (short positions)"
        )
        weight_bounds = (-1, 1) if allow_shorting else (0, 1)
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your own price data (CSV)",
        type=['csv'],
        help="CSV should have dates as index and asset prices in columns"
    )
    
    if uploaded_file is not None:
        st.sidebar.info("Using uploaded data")
    else:
        st.sidebar.info("Using default stock data")
    
    prices = load_data(uploaded_file)
    
    if prices is None:
        st.error("Failed to load price data. Please check the file format.")
        return
        
    if prices.empty:
        st.error("No valid data found after removing missing values.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Asset Prices")
        st.line_chart(prices)
    
    with col2:
        st.subheader("Cumulative Returns")
        fig_returns = plot_cumulative_returns(prices)
        st.plotly_chart(fig_returns, use_container_width=True)
    
    try:
        if "HRP" in optimization_method:
            weights, perf = optimize_hrp(prices)
            has_efficient_frontier = False
            
            # Show covariance and correlation heatmaps for HRP
            st.subheader("Asset Relationships")
            st.markdown("""
            These heatmaps show how assets move together:
            - Covariance Matrix: Shows the absolute relationship between asset returns
            - Correlation Matrix: Shows the standardized relationship (-1 to 1)
            - Red indicates positive relationship
            - Blue indicates negative relationship
            - Darker colors mean stronger relationships
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Covariance Matrix")
                fig_cov = plot_covariance_heatmap(prices, plot_correlation=False)
                st.plotly_chart(fig_cov, use_container_width=True)
            
            with col2:
                st.subheader("Correlation Matrix")
                fig_corr = plot_covariance_heatmap(prices, plot_correlation=True)
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            weights, perf, frontier_params = optimize_mvo(prices, weight_bounds)
            has_efficient_frontier = True
        
        method = "HRP" if "HRP" in optimization_method else "Maximum Sharpe"
        st.subheader(f"Optimal Portfolio Weights ({method})")
        fig_weights = plot_portfolio_weights(weights)
        st.plotly_chart(fig_weights, use_container_width=True)
        
        # Show portfolio statistics and metrics in two columns
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Portfolio Statistics")
            # Add pie chart
            fig_pie = plot_portfolio_pie(weights)
            st.plotly_chart(fig_pie, use_container_width=True)
            # Show weights table below pie chart
            stats = pd.DataFrame({
                'Weight': pd.Series(weights)
            })
            st.dataframe(stats.style.format({'Weight': '{:.2%}'}))
        
        with col4:
            st.subheader("Portfolio Metrics")
            display_portfolio_metrics(perf)
        
        # Show efficient frontier plot and minimum volatility portfolio for MVO
        if has_efficient_frontier:
            st.subheader("Efficient Frontier")
            fig_frontier = plot_efficient_frontier(*frontier_params)
            st.plotly_chart(fig_frontier, use_container_width=True)
            
            # Display minimum volatility portfolio
            st.subheader("Minimum Volatility Portfolio")
            min_vol_weights = frontier_params[3]
            min_vol_perf = frontier_params[4]
            
            col5, col6 = st.columns(2)
            
            with col5:
                st.subheader("Portfolio Statistics")
                # Add pie chart for min vol portfolio
                fig_pie_min_vol = plot_portfolio_pie(min_vol_weights)
                st.plotly_chart(fig_pie_min_vol, use_container_width=True)
                # Show weights table below pie chart
                min_vol_stats = pd.DataFrame({
                    'Weight': pd.Series(min_vol_weights)
                })
                st.dataframe(min_vol_stats.style.format({'Weight': '{:.2%}'}))
            
            with col6:
                st.subheader("Portfolio Metrics")
                display_portfolio_metrics(min_vol_perf)
            
    except Exception as e:
        st.error(f"Error in portfolio optimization: {str(e)}")
        st.info("This might happen if there's not enough data or if the data is too volatile.")

if __name__ == "__main__":
    main()
