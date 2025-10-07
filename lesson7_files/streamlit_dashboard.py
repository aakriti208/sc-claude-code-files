"""
E-commerce Business Analytics Dashboard

A professional Streamlit dashboard for analyzing e-commerce sales data
with interactive filters and comprehensive business metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import EcommerceDataLoader, load_and_process_data, categorize_delivery_speed
from business_metrics import BusinessMetricsCalculator

# Page configuration
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main content area */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    div[data-testid="metric-container"] > label {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #495057 !important;
    }

    div[data-testid="metric-container"] > div {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #212529 !important;
    }

    /* Trend indicators */
    div[data-testid="metric-container"] > div > div {
        font-size: 14px !important;
    }

    /* Headers */
    h1 {
        color: #212529;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    h3 {
        color: #495057;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Chart containers */
    .plotly-graph-div {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(data_path='ecommerce_data/'):
    """Load and cache the e-commerce data."""
    loader, processed_data = load_and_process_data(data_path)
    return loader, processed_data


def format_currency_short(value):
    """Format currency values in short form (e.g., $300K, $2M)."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:.0f}"


def format_currency_full(value):
    """Format currency values in full form with commas."""
    return f"${value:,.0f}"


def create_trend_indicator(current, previous):
    """Create a trend indicator with color coding."""
    if previous == 0 or pd.isna(previous):
        return None, None

    change = ((current - previous) / previous) * 100
    if change > 0:
        return f"▲ {change:.2f}%", "normal"  # Green (positive)
    elif change < 0:
        return f"▼ {abs(change):.2f}%", "inverse"  # Red (negative)
    else:
        return "— 0.00%", "off"


def plot_revenue_trend(current_data, previous_data, current_year, previous_year):
    """Create revenue trend line chart comparing two periods."""
    fig = go.Figure()

    # Current period line (solid)
    if len(current_data) > 0:
        fig.add_trace(go.Scatter(
            x=current_data['month'],
            y=current_data['revenue'],
            mode='lines+markers',
            name=str(current_year),
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8),
            hovertemplate='Month: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
        ))

    # Previous period line (dashed)
    if previous_data is not None and len(previous_data) > 0:
        fig.add_trace(go.Scatter(
            x=previous_data['month'],
            y=previous_data['revenue'],
            mode='lines+markers',
            name=str(previous_year),
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8),
            hovertemplate='Month: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
        ))

    fig.update_layout(
        title='Revenue Trend by Month',
        xaxis_title='Month',
        yaxis_title='Revenue',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            showgrid=True,
            gridcolor='#e9ecef'
        ),
        yaxis=dict(
            tickformat='$,.0f',
            showgrid=True,
            gridcolor='#e9ecef'
        ),
        hovermode='x unified'
    )

    # Format y-axis with short currency format
    fig.update_yaxes(tickformat='$,.0s')

    return fig


def plot_top_categories(category_data):
    """Create horizontal bar chart for top 10 categories."""
    # Sort descending by revenue and take top 10
    top_10 = category_data.nlargest(10, 'total_revenue').sort_values('total_revenue', ascending=True)

    # Create blue gradient colors
    max_revenue = top_10['total_revenue'].max()
    colors = [f'rgba(31, 119, 180, {0.4 + 0.6 * (rev / max_revenue)})'
              for rev in top_10['total_revenue']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_10['product_category_name'],
        x=top_10['total_revenue'],
        orientation='h',
        marker=dict(color=colors),
        text=[format_currency_short(val) for val in top_10['total_revenue']],
        textposition='outside',
        hovertemplate='%{y}<br>Revenue: $%{x:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='Top 10 Product Categories by Revenue',
        xaxis_title='Revenue',
        yaxis_title='',
        template='plotly_white',
        height=400,
        showlegend=False,
        xaxis=dict(
            tickformat='$,.0s',
            showgrid=True,
            gridcolor='#e9ecef'
        ),
        margin=dict(l=150, r=50, t=50, b=50)
    )

    return fig


def plot_state_map(geo_data):
    """Create US choropleth map showing revenue by state."""
    fig = px.choropleth(
        geo_data,
        locations='state',
        color='revenue',
        locationmode='USA-states',
        scope='usa',
        color_continuous_scale='Blues',
        labels={'revenue': 'Revenue'},
        hover_data={'state': True, 'revenue': ':$,.0f'}
    )

    fig.update_layout(
        title='Revenue by State',
        template='plotly_white',
        height=400,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='albers usa'
        ),
        coloraxis_colorbar=dict(
            title="Revenue",
            tickformat='$,.0s'
        )
    )

    return fig


def plot_satisfaction_by_delivery(sales_data):
    """Create bar chart showing satisfaction vs delivery time."""
    # Create delivery buckets
    sales_data_copy = sales_data.copy()
    sales_data_copy['delivery_category'] = sales_data_copy['delivery_days'].apply(categorize_delivery_speed)

    # Calculate average review score by delivery bucket
    delivery_satisfaction = sales_data_copy.groupby('delivery_category')['review_score'].mean().reset_index()

    # Define order for categories
    category_order = ['1-3 days', '4-7 days', '8+ days']
    delivery_satisfaction['delivery_category'] = pd.Categorical(
        delivery_satisfaction['delivery_category'],
        categories=category_order,
        ordered=True
    )
    delivery_satisfaction = delivery_satisfaction.sort_values('delivery_category')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=delivery_satisfaction['delivery_category'],
        y=delivery_satisfaction['review_score'],
        marker=dict(color='#1f77b4'),
        text=[f"{score:.2f}" for score in delivery_satisfaction['review_score']],
        textposition='outside',
        hovertemplate='Delivery Time: %{x}<br>Avg Review Score: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Average Review Score by Delivery Time',
        xaxis_title='Delivery Time',
        yaxis_title='Average Review Score',
        template='plotly_white',
        height=400,
        showlegend=False,
        yaxis=dict(
            range=[0, 5],
            showgrid=True,
            gridcolor='#e9ecef'
        )
    )

    return fig


def main():
    """Main dashboard application."""

    # Header
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("📊 E-commerce Business Analytics Dashboard")

    # Load data
    loader, processed_data = load_data()

    # Get available date range
    orders_data = processed_data['orders']
    min_date = orders_data['order_purchase_timestamp'].min().date()
    max_date = orders_data['order_purchase_timestamp'].max().date()

    with col2:
        # Year filter - default to 2023
        st.markdown("<br>", unsafe_allow_html=True)
        available_years = sorted(orders_data['purchase_year'].dropna().unique(), reverse=True)
        default_year = 2023 if 2023 in available_years else available_years[0]
        selected_year = st.selectbox(
            "Year",
            options=available_years,
            index=available_years.index(default_year),
            key="year_filter"
        )

    with col3:
        # Month filter
        st.markdown("<br>", unsafe_allow_html=True)
        month_options = ["All Months"] + [f"{i:02d} - {pd.to_datetime(f'2000-{i:02d}-01').strftime('%B')}"
                                          for i in range(1, 13)]
        selected_month = st.selectbox(
            "Month",
            options=month_options,
            index=0,
            key="month_filter"
        )

    # Parse month filter
    if selected_month == "All Months":
        month_filter = None
    else:
        month_filter = int(selected_month.split(" - ")[0])

    # Create filtered sales dataset
    current_year = selected_year
    previous_year = current_year - 1

    sales_data_current = loader.create_sales_dataset(
        year_filter=current_year,
        month_filter=month_filter,
        status_filter='delivered'
    )

    # Get previous year data for comparison
    sales_data_previous = loader.create_sales_dataset(
        year_filter=previous_year,
        month_filter=month_filter,
        status_filter='delivered'
    )

    # Calculate metrics
    combined_data = pd.concat([sales_data_current, sales_data_previous], ignore_index=True)
    metrics_calc = BusinessMetricsCalculator(combined_data)

    # Generate reports
    current_report = metrics_calc.generate_comprehensive_report(current_year, previous_year)
    revenue_metrics = current_report['revenue_metrics']

    st.markdown("---")

    # KPI Row - 4 cards
    st.markdown("### Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        trend_text, trend_color = create_trend_indicator(
            revenue_metrics['total_revenue'],
            revenue_metrics.get('previous_year_revenue', 0)
        )
        st.metric(
            label="Total Revenue",
            value=format_currency_full(revenue_metrics['total_revenue']),
            delta=trend_text,
            delta_color=trend_color
        )

    with kpi_col2:
        # Calculate monthly growth rate
        monthly_data = current_report['monthly_trends']
        if len(monthly_data) > 0:
            avg_monthly_growth = monthly_data['revenue_growth'].mean()
            growth_text = f"▲ {avg_monthly_growth:.2f}%" if avg_monthly_growth > 0 else f"▼ {abs(avg_monthly_growth):.2f}%"
            growth_color = "normal" if avg_monthly_growth > 0 else "inverse"
        else:
            growth_text = None
            growth_color = "off"

        st.metric(
            label="Avg Monthly Growth",
            value=f"{avg_monthly_growth:.2f}%" if len(monthly_data) > 0 else "N/A",
            delta=growth_text,
            delta_color=growth_color
        )

    with kpi_col3:
        trend_text, trend_color = create_trend_indicator(
            revenue_metrics['average_order_value'],
            revenue_metrics.get('previous_year_aov', 0)
        )
        st.metric(
            label="Average Order Value",
            value=format_currency_full(revenue_metrics['average_order_value']),
            delta=trend_text,
            delta_color=trend_color
        )

    with kpi_col4:
        trend_text, trend_color = create_trend_indicator(
            revenue_metrics['total_orders'],
            revenue_metrics.get('previous_year_orders', 0)
        )
        st.metric(
            label="Total Orders",
            value=f"{revenue_metrics['total_orders']:,}",
            delta=trend_text,
            delta_color=trend_color
        )

    st.markdown("---")

    # Charts Grid - 2x2 layout
    st.markdown("### Business Performance Analysis")

    # Row 1
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Revenue trend chart
        monthly_current = current_report['monthly_trends']

        # Get previous year monthly data
        previous_report = metrics_calc.generate_comprehensive_report(previous_year, None)
        monthly_previous = previous_report['monthly_trends']

        fig_revenue = plot_revenue_trend(
            monthly_current,
            monthly_previous,
            current_year,
            previous_year
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

    with chart_col2:
        # Top categories chart
        product_data = current_report['product_performance']
        if 'error' not in product_data and len(product_data['all_categories']) > 0:
            fig_categories = plot_top_categories(product_data['all_categories'])
            st.plotly_chart(fig_categories, use_container_width=True)

    # Row 2
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        # State map
        geo_data = current_report['geographic_performance']
        if 'error' not in geo_data.columns and len(geo_data) > 0:
            fig_map = plot_state_map(geo_data)
            st.plotly_chart(fig_map, use_container_width=True)

    with chart_col4:
        # Satisfaction by delivery time
        if len(sales_data_current) > 0 and 'review_score' in sales_data_current.columns and 'delivery_days' in sales_data_current.columns:
            fig_satisfaction = plot_satisfaction_by_delivery(sales_data_current)
            st.plotly_chart(fig_satisfaction, use_container_width=True)

    st.markdown("---")

    # Bottom Row - 2 cards
    st.markdown("### Customer Experience Metrics")
    bottom_col1, bottom_col2 = st.columns(2)

    with bottom_col1:
        # Average delivery time card
        delivery_metrics = current_report['delivery_performance']
        if 'error' not in delivery_metrics:
            # Calculate previous year delivery for trend
            prev_delivery = metrics_calc.analyze_delivery_performance(previous_year)

            if 'error' not in prev_delivery:
                trend_text, trend_color = create_trend_indicator(
                    delivery_metrics['avg_delivery_days'],
                    prev_delivery['avg_delivery_days']
                )
                # Invert colors for delivery time (lower is better)
                if trend_color == "normal":
                    trend_color = "inverse"
                elif trend_color == "inverse":
                    trend_color = "normal"
            else:
                trend_text = None
                trend_color = "off"

            st.metric(
                label="Average Delivery Time",
                value=f"{delivery_metrics['avg_delivery_days']:.1f} days",
                delta=trend_text,
                delta_color=trend_color
            )

    with bottom_col2:
        # Review score card
        satisfaction_metrics = current_report['customer_satisfaction']
        if 'error' not in satisfaction_metrics:
            avg_score = satisfaction_metrics['avg_review_score']
            stars = "⭐" * int(round(avg_score))

            st.markdown(f"""
            <div style='background-color: #f8f9fa;
                        border: 1px solid #e9ecef;
                        padding: 15px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                <div style='font-size: 14px; font-weight: 600; color: #495057; margin-bottom: 8px;'>
                    Average Review Score
                </div>
                <div style='font-size: 40px; font-weight: 700; color: #212529; margin-bottom: 8px;'>
                    {avg_score:.2f} {stars}
                </div>
                <div style='font-size: 14px; color: #6c757d;'>
                    Based on {satisfaction_metrics['total_reviews']:,} reviews
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
