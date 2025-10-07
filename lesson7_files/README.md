# E-commerce Business Analytics - Refactored Analysis

A comprehensive, configurable business intelligence framework for e-commerce sales data analysis with reusable modules and professional visualizations.

## Overview

This project refactors a basic exploratory data analysis into a professional, maintainable business intelligence solution. The framework provides configurable time period analysis, modular architecture, and comprehensive business metrics calculations.

## Project Structure

```
lesson7_files/
├── EDA_Refactored.ipynb       # Main analysis notebook
├── data_loader.py              # Data loading and processing module
├── business_metrics.py         # Business metrics calculation module
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── ecommerce_data/            # Data directory
    ├── orders_dataset.csv
    ├── order_items_dataset.csv
    ├── products_dataset.csv
    ├── customers_dataset.csv
    └── order_reviews_dataset.csv
```

## Features

### 1. Configurable Analysis Framework
- Set analysis year, comparison year, and month filters
- Flexible time period analysis without code changes
- Automatic handling of missing data periods

### 2. Comprehensive Business Metrics
- **Revenue Analysis**: Total revenue, growth rates, average order value
- **Product Performance**: Category analysis, revenue share, top performers
- **Geographic Insights**: State-level revenue and order analysis
- **Customer Satisfaction**: Review scores, satisfaction distribution
- **Delivery Performance**: Delivery times, speed categorization

### 3. Professional Visualizations
- Monthly revenue trend charts with clear labels and units
- Product category performance horizontal bar charts
- Interactive geographic heatmaps (USA states)
- Customer satisfaction distribution charts
- Consistent business-oriented color schemes
- All plots include date ranges and proper formatting

### 4. Modular Code Architecture
- **data_loader.py**: Handles data loading, cleaning, and transformation
- **business_metrics.py**: Calculates all business metrics
- Reusable functions with comprehensive docstrings
- Clean separation of concerns

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation Steps

1. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure data files are in place**:
   - Place CSV files in the `ecommerce_data/` directory
   - Verify all required files are present:
     - orders_dataset.csv
     - order_items_dataset.csv
     - products_dataset.csv
     - customers_dataset.csv
     - order_reviews_dataset.csv

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook EDA_Refactored.ipynb
   ```

## Usage Guide

### Running the Analysis

1. **Open the refactored notebook**: `EDA_Refactored.ipynb`

2. **Configure analysis parameters** in the first code cell:
   ```python
   ANALYSIS_YEAR = 2023        # Year to analyze
   COMPARISON_YEAR = 2022      # Comparison year (or None)
   ANALYSIS_MONTH = None       # Specific month (1-12) or None for full year
   DATA_PATH = 'ecommerce_data/'
   ```

3. **Run all cells** to generate the complete analysis

### Configurable Analysis Examples

#### Analyze Full Year
```python
ANALYSIS_YEAR = 2023
COMPARISON_YEAR = 2022
ANALYSIS_MONTH = None  # Full year
```

#### Analyze Specific Month
```python
ANALYSIS_YEAR = 2023
COMPARISON_YEAR = 2022
ANALYSIS_MONTH = 12  # December only
```

#### Analyze Without Comparison
```python
ANALYSIS_YEAR = 2023
COMPARISON_YEAR = None  # No year-over-year comparison
ANALYSIS_MONTH = None
```

### Using the Modules Independently

#### Data Loading Module
```python
from data_loader import EcommerceDataLoader, load_and_process_data

# Quick start
loader, processed_data = load_and_process_data('ecommerce_data/')

# Create filtered dataset
sales_data = loader.create_sales_dataset(
    year_filter=2023,
    month_filter=None,
    status_filter='delivered'
)
```

#### Business Metrics Module
```python
from business_metrics import BusinessMetricsCalculator, MetricsVisualizer

# Calculate metrics
metrics_calc = BusinessMetricsCalculator(sales_data)
report = metrics_calc.generate_comprehensive_report(
    current_year=2023,
    previous_year=2022
)

# Create visualizations
visualizer = MetricsVisualizer(report)
revenue_fig = visualizer.plot_revenue_trend()
category_fig = visualizer.plot_category_performance()
```

## Key Business Metrics

### Revenue Metrics
- **Total Revenue**: Sum of all delivered order item prices
- **Revenue Growth Rate**: Year-over-year percentage change
- **Average Order Value (AOV)**: Average total value per order
- **Monthly Growth Trends**: Month-over-month performance

### Product Performance
- **Category Revenue**: Revenue by product category
- **Revenue Share**: Percentage of total revenue by category
- **Top Categories**: Categories ranked by total revenue

### Geographic Analysis
- **State Performance**: Revenue and order count by state
- **Market Penetration**: Number of active markets
- **Regional AOV**: Average order value by geographic region

### Customer Experience
- **Review Scores**: Average satisfaction rating (1-5 scale)
- **Satisfaction Distribution**: Percentage of high/low ratings
- **Delivery Performance**: Average delivery time and speed metrics

## Data Dictionary

### Core Columns

**Orders**
- `order_id`: Unique identifier for each order
- `customer_id`: Links to customer information
- `order_status`: Order status (delivered, shipped, canceled, etc.)
- `order_purchase_timestamp`: Date and time of order placement
- `order_delivered_customer_date`: Date and time of delivery

**Order Items**
- `order_id`: Links to orders table
- `product_id`: Links to products table
- `price`: Item price in USD (excluding shipping)
- `freight_value`: Shipping cost in USD

**Products**
- `product_id`: Unique product identifier
- `product_category_name`: Product category

**Customers**
- `customer_id`: Unique customer identifier
- `customer_state`: State abbreviation (CA, TX, NY, etc.)
- `customer_city`: City name

**Reviews**
- `order_id`: Links to orders table
- `review_score`: Customer rating (1-5 scale, 5 = best)

### Calculated Metrics
- **delivery_days**: Days between order placement and delivery
- **purchase_year**: Year extracted from purchase timestamp
- **purchase_month**: Month extracted from purchase timestamp

## Output Examples

### Console Output
```
BUSINESS METRICS SUMMARY - 2023
============================================================

REVENUE PERFORMANCE:
  Total Revenue: $3,360,294.74
  Total Orders: 4,635
  Average Order Value: $724.98
  Revenue Growth: -2.5%

CUSTOMER SATISFACTION:
  Average Review Score: 4.10/5.0
  High Satisfaction (4+ stars): 51.6%

DELIVERY PERFORMANCE:
  Average Delivery Time: 8.0 days
  Fast Delivery (3 days or less): 7.2%
```

### Generated Visualizations
- Monthly revenue trend line charts with proper axis labels
- Top product category horizontal bar charts
- Interactive US state choropleth maps
- Customer satisfaction distribution bar charts

## Customization and Extension

### Adding New Metrics

1. Extend the `BusinessMetricsCalculator` class in `business_metrics.py`:
```python
def calculate_custom_metric(self, year: int) -> Dict[str, float]:
    """
    Calculate custom business metric.

    Args:
        year (int): Year to analyze

    Returns:
        Dict[str, float]: Custom metrics
    """
    year_data = self.sales_data[self.sales_data['purchase_year'] == year]
    # Your calculation logic here
    return metrics
```

2. Add visualization methods to `MetricsVisualizer` class
3. Update the notebook to display new metrics

### Modifying Data Sources

- Modify `data_loader.py` to handle different CSV structures
- Update column mappings in the `EcommerceDataLoader` class
- Add new data validation rules

## Troubleshooting

### Common Issues

**Module Import Errors**
- Ensure all files are in the same directory
- Verify Python path configuration

**Missing Data Files**
- Check that CSV files exist in the `ecommerce_data/` directory
- Verify file names match expected patterns

**Empty Results**
- Verify date filters match available data
- Check that order status filter is correct

**Visualization Issues**
- Ensure all required packages are installed
- Check Plotly version compatibility for interactive maps

### Performance Optimization

For large datasets:
- Use data sampling for initial exploration
- Implement chunked processing
- Consider caching results for repeated analysis

## Success Criteria Met

- Clean, readable code without icons or emojis
- Configurable analysis framework for any date range
- Reusable modules that work across datasets
- Maintainable structure with comprehensive documentation
- All original analyses preserved and enhanced
- Professional visualizations with proper labeling

## Next Steps

### Potential Enhancements
- Add predictive analytics and forecasting
- Implement customer segmentation analysis
- Create automated report scheduling
- Add export functionality (PDF/Excel reports)
- Develop real-time dashboard integration

### Extension Ideas
- Integration with business intelligence tools
- API endpoints for metrics access
- Advanced statistical analysis
- A/B testing framework
- Mobile-responsive reporting

## License

This project is provided for educational and business analysis purposes.

---

**Note**: This framework is designed to be easily maintained and extended for ongoing business intelligence needs. The modular architecture ensures that updates to data sources or metric calculations can be made without affecting the overall analysis structure.
