import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Inventory Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class InventoryAnalyzer:
    def __init__(self):
        self.data = None
        self.processed_data = None

    def load_data(self, uploaded_file):
        """Load data from uploaded CSV or Excel file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            else:
                self.data = pd.read_excel(uploaded_file)
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False

    def preprocess_data(self):
        """Preprocess and clean the data"""
        if self.data is None:
            return False

        # Make a copy for processing
        df = self.data.copy()

        # Convert column names to lowercase and replace spaces with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Try to identify date column
        date_columns = [col for col in df.columns if 'date' in col or 'time' in col]
        if date_columns:
            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]], errors='coerce')
            df = df.dropna(subset=[date_columns[0]])
            df['date'] = df[date_columns[0]]
        else:
            st.warning("No date column found. Please ensure your data has a date column.")
            return False

        # Try to identify key columns
        quantity_cols = [col for col in df.columns if any(word in col for word in ['qty', 'quantity', 'sold', 'units'])]
        price_cols = [col for col in df.columns if any(word in col for word in ['price', 'cost', 'revenue', 'sales', 'amount'])]
        item_cols = [col for col in df.columns if any(word in col for word in ['item', 'product', 'name', 'sku'])]
        category_cols = [col for col in df.columns if any(word in col for word in ['category', 'type', 'group', 'class'])]

        # Assign columns
        if quantity_cols:
            df['quantity_sold'] = pd.to_numeric(df[quantity_cols[0]], errors='coerce')
        if price_cols:
            df['sales_amount'] = pd.to_numeric(df[price_cols[0]], errors='coerce')
        if item_cols:
            df['item_name'] = df[item_cols[0]].astype(str)
        if category_cols:
            df['category'] = df[category_cols[0]].astype(str)

        # Calculate revenue if not available
        if 'sales_amount' not in df.columns and 'quantity_sold' in df.columns:
            if price_cols:
                df['sales_amount'] = df['quantity_sold'] * pd.to_numeric(df[price_cols[0]], errors='coerce')

        # Remove rows with missing essential data
        essential_columns = ['date', 'item_name']
        df = df.dropna(subset=essential_columns)

        # Add time-based columns
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year_month'] = df['date'].dt.to_period('M')
        df['quarter'] = df['date'].dt.quarter

        self.processed_data = df
        return True

    def generate_insights(self):
        """Generate key business insights"""
        if self.processed_data is None:
            return {}

        df = self.processed_data
        insights = {}

        # Overall metrics
        total_items = df['item_name'].nunique()
        total_sales = df['sales_amount'].sum() if 'sales_amount' in df.columns else 0
        total_quantity = df['quantity_sold'].sum() if 'quantity_sold' in df.columns else 0
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"

        insights['overview'] = {
            'total_items': total_items,
            'total_sales': total_sales,
            'total_quantity': total_quantity,
            'date_range': date_range
        }

        # Item-level analysis
        if 'quantity_sold' in df.columns:
            item_performance = df.groupby('item_name').agg({
                'quantity_sold': 'sum',
                'sales_amount': 'sum' if 'sales_amount' in df.columns else 'count'
            }).reset_index()

            # Best and worst performing items
            item_performance = item_performance.sort_values('quantity_sold', ascending=False)

            insights['top_items'] = item_performance.head(10)
            insights['bottom_items'] = item_performance.tail(10)

            # Items to buy more (top performers)
            insights['buy_more'] = item_performance.head(5)['item_name'].tolist()

            # Items to reduce (bottom performers)
            insights['reduce_stock'] = item_performance.tail(5)['item_name'].tolist()

        # Trend analysis - items with declining sales
        if len(df['year_month'].unique()) > 1:
            monthly_trends = df.groupby(['item_name', 'year_month'])['quantity_sold'].sum().reset_index()
            monthly_trends['year_month'] = monthly_trends['year_month'].astype(str)

            # Calculate trend for each item
            declining_items = []
            for item in df['item_name'].unique():
                item_data = monthly_trends[monthly_trends['item_name'] == item].sort_values('year_month')
                if len(item_data) >= 3:
                    recent_avg = item_data.tail(2)['quantity_sold'].mean()
                    older_avg = item_data.head(2)['quantity_sold'].mean()
                    if recent_avg < older_avg * 0.7:  # 30% decline
                        declining_items.append(item)

            insights['watch_out'] = declining_items[:5]

        # Category analysis
        if 'category' in df.columns:
            category_performance = df.groupby('category').agg({
                'quantity_sold': 'sum',
                'sales_amount': 'sum' if 'sales_amount' in df.columns else 'count'
            }).reset_index()
            insights['category_performance'] = category_performance.sort_values('quantity_sold', ascending=False)

        return insights

def create_visualizations(analyzer):
    """Create various visualizations"""
    if analyzer.processed_data is None:
        return

    df = analyzer.processed_data

    # 1. Sales Over Time
    st.subheader("📈 Sales Trends Over Time")

    col1, col2 = st.columns(2)

    with col1:
        # Monthly sales trend
        monthly_sales = df.groupby('year_month').agg({
            'quantity_sold': 'sum' if 'quantity_sold' in df.columns else 'count',
            'sales_amount': 'sum' if 'sales_amount' in df.columns else 'count'
        }).reset_index()
        monthly_sales['year_month_str'] = monthly_sales['year_month'].astype(str)

        fig_monthly = px.line(monthly_sales, x='year_month_str', y='quantity_sold',
                             title='Monthly Quantity Sold Trend',
                             labels={'year_month_str': 'Month', 'quantity_sold': 'Quantity Sold'})
        fig_monthly.update_layout(height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col2:
        if 'sales_amount' in df.columns:
            fig_revenue = px.line(monthly_sales, x='year_month_str', y='sales_amount',
                                 title='Monthly Revenue Trend',
                                 labels={'year_month_str': 'Month', 'sales_amount': 'Revenue ($)'})
            fig_revenue.update_layout(height=400)
            st.plotly_chart(fig_revenue, use_container_width=True)

    # 2. Top Performing Items
    st.subheader("🏆 Top Performing Items")

    col1, col2 = st.columns(2)

    with col1:
        # Top items by quantity
        top_items_qty = df.groupby('item_name')['quantity_sold'].sum().sort_values(ascending=False).head(10)
        fig_top_qty = px.bar(x=top_items_qty.values, y=top_items_qty.index, orientation='h',
                            title='Top 10 Items by Quantity Sold',
                            labels={'x': 'Quantity Sold', 'y': 'Item'})
        fig_top_qty.update_layout(height=500)
        st.plotly_chart(fig_top_qty, use_container_width=True)

    with col2:
        if 'sales_amount' in df.columns:
            # Top items by revenue
            top_items_revenue = df.groupby('item_name')['sales_amount'].sum().sort_values(ascending=False).head(10)
            fig_top_revenue = px.bar(x=top_items_revenue.values, y=top_items_revenue.index, orientation='h',
                                    title='Top 10 Items by Revenue',
                                    labels={'x': 'Revenue ($)', 'y': 'Item'})
            fig_top_revenue.update_layout(height=500)
            st.plotly_chart(fig_top_revenue, use_container_width=True)

    # 3. Category Analysis
    if 'category' in df.columns:
        st.subheader("📊 Category Performance")

        col1, col2 = st.columns(2)

        with col1:
            category_qty = df.groupby('category')['quantity_sold'].sum().sort_values(ascending=False)
            fig_cat_qty = px.pie(values=category_qty.values, names=category_qty.index,
                                title='Quantity Sold by Category')
            st.plotly_chart(fig_cat_qty, use_container_width=True)

        with col2:
            if 'sales_amount' in df.columns:
                category_revenue = df.groupby('category')['sales_amount'].sum().sort_values(ascending=False)
                fig_cat_revenue = px.pie(values=category_revenue.values, names=category_revenue.index,
                                        title='Revenue by Category')
                st.plotly_chart(fig_cat_revenue, use_container_width=True)

    # 4. Year-over-Year Comparison (if applicable)
    years = df['year'].unique()
    if len(years) > 1:
        st.subheader("📅 Year-over-Year Comparison")

        yearly_comparison = df.groupby(['year', 'month']).agg({
            'quantity_sold': 'sum',
            'sales_amount': 'sum' if 'sales_amount' in df.columns else 'count'
        }).reset_index()

        fig_yoy = px.line(yearly_comparison, x='month', y='quantity_sold', color='year',
                         title='Year-over-Year Monthly Comparison',
                         labels={'month': 'Month', 'quantity_sold': 'Quantity Sold'})
        st.plotly_chart(fig_yoy, use_container_width=True)

    # 5. Seasonal Analysis
    st.subheader("🗓️ Seasonal Analysis")

    seasonal_data = df.groupby(['month'])['quantity_sold'].sum().reset_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    seasonal_data['month_name'] = seasonal_data['month'].apply(lambda x: month_names[x-1])

    fig_seasonal = px.bar(seasonal_data, x='month_name', y='quantity_sold',
                         title='Sales by Month (Seasonal Trends)',
                         labels={'month_name': 'Month', 'quantity_sold': 'Quantity Sold'})
    st.plotly_chart(fig_seasonal, use_container_width=True)

def generate_pdf_report(analyzer, insights):
    """Generate PDF report with insights and recommendations"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1f77b4'),
        alignment=1  # Center alignment
    )

    story.append(Paragraph("Inventory Analysis Report", title_style))
    story.append(Spacer(1, 20))

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))

    overview = insights.get('overview', {})
    summary_text = f"""
    This report analyzes inventory data from {overview.get('date_range', 'N/A')}.
    The analysis covers {overview.get('total_items', 0)} unique items with total sales of
    ${overview.get('total_sales', 0):,.2f} and {overview.get('total_quantity', 0):,} units sold.
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))

    # Key Recommendations
    story.append(Paragraph("📈 Buy More of These Items", styles['Heading2']))
    if 'buy_more' in insights:
        buy_more_items = insights['buy_more']
        for i, item in enumerate(buy_more_items, 1):
            story.append(Paragraph(f"{i}. {item}", styles['Normal']))
    story.append(Spacer(1, 15))

    story.append(Paragraph("📉 Reduce Stock on These Items", styles['Heading2']))
    if 'reduce_stock' in insights:
        reduce_items = insights['reduce_stock']
        for i, item in enumerate(reduce_items, 1):
            story.append(Paragraph(f"{i}. {item}", styles['Normal']))
    story.append(Spacer(1, 15))

    story.append(Paragraph("⚠️ Watch Out for These Items (Sales Dropping Fast)", styles['Heading2']))
    if 'watch_out' in insights:
        watch_items = insights.get('watch_out', [])
        if watch_items:
            for i, item in enumerate(watch_items, 1):
                story.append(Paragraph(f"{i}. {item}", styles['Normal']))
        else:
            story.append(Paragraph("No items showing significant declining trends.", styles['Normal']))

    # Category Analysis
    if 'category_performance' in insights:
        story.append(PageBreak())
        story.append(Paragraph("Category Performance Analysis", styles['Heading2']))

        cat_data = insights['category_performance']
        table_data = [['Category', 'Total Quantity Sold', 'Total Revenue']]
        for _, row in cat_data.head(10).iterrows():
            table_data.append([
                str(row['category']),
                f"{row['quantity_sold']:,}",
                f"${row.get('sales_amount', 0):,.2f}"
            ])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    st.markdown("<h1 class='main-header'>📊 Inventory Analysis Dashboard</h1>", unsafe_allow_html=True)

    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = InventoryAnalyzer()

    # Sidebar
    st.sidebar.header("📁 Data Upload")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your inventory data file. Make sure it contains columns for dates, item names, quantities, and optionally prices/categories."
    )

    if uploaded_file is not None:
        if st.session_state.analyzer.load_data(uploaded_file):
            st.sidebar.success("✅ File uploaded successfully!")

            # Show data preview
            st.sidebar.subheader("Data Preview")
            st.sidebar.dataframe(st.session_state.analyzer.data.head())

            # Process data
            if st.session_state.analyzer.preprocess_data():
                df = st.session_state.analyzer.processed_data

                # Main dashboard
                st.subheader("📊 Overview")

                # Generate insights
                insights = st.session_state.analyzer.generate_insights()
                overview = insights.get('overview', {})

                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Items", f"{overview.get('total_items', 0):,}")

                with col2:
                    st.metric("Total Quantity Sold", f"{overview.get('total_quantity', 0):,}")

                with col3:
                    st.metric("Total Revenue", f"${overview.get('total_sales', 0):,.2f}")

                with col4:
                    st.metric("Date Range", len(df['date'].dt.date.unique()))

                # Display insights
                st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
                st.subheader("🎯 Key Insights")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**📈 Top Items to Buy More:**")
                    if 'buy_more' in insights:
                        for item in insights['buy_more'][:5]:
                            st.write(f"• {item}")

                with col2:
                    st.write("**📉 Items to Reduce Stock:**")
                    if 'reduce_stock' in insights:
                        for item in insights['reduce_stock'][:5]:
                            st.write(f"• {item}")

                with col3:
                    st.write("**⚠️ Items to Watch:**")
                    watch_items = insights.get('watch_out', [])
                    if watch_items:
                        for item in watch_items:
                            st.write(f"• {item}")
                    else:
                        st.write("• No concerning trends detected")

                st.markdown("</div>", unsafe_allow_html=True)

                # Create visualizations
                create_visualizations(st.session_state.analyzer)

                # PDF Export
                st.subheader("📄 Export Report")

                if st.button("Generate PDF Report", type="primary"):
                    with st.spinner("Generating PDF report..."):
                        pdf_buffer = generate_pdf_report(st.session_state.analyzer, insights)

                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"inventory_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )

                        st.success("✅ PDF report generated successfully!")

                # Data Export Options
                st.subheader("💾 Export Processed Data")

                col1, col2 = st.columns(2)

                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Processed Data (CSV)",
                        data=csv,
                        file_name=f"processed_inventory_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    # Summary statistics
                    summary_stats = df.describe()
                    summary_csv = summary_stats.to_csv()
                    st.download_button(
                        label="📈 Download Summary Statistics",
                        data=summary_csv,
                        file_name=f"inventory_summary_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("❌ Error processing data. Please check your file format and column names.")
    else:
        # Instructions
        st.markdown("""
        ### 📋 Instructions

        1. **Upload your inventory data** using the file uploader in the sidebar
        2. **Supported formats**: CSV, Excel (.xlsx, .xls)
        3. **Required columns**: Your data should include:
           - Date column (sales date, transaction date, etc.)
           - Item/Product name
           - Quantity sold
           - Price/Revenue (optional)
           - Category (optional)

        ### 📊 What You'll Get

        - **Interactive visualizations** showing sales trends and patterns
        - **Actionable insights** on which items to buy more or less
        - **Category analysis** to understand which product categories drive sales
        - **Seasonal trends** to identify peak selling periods
        - **Year-over-year comparisons** (when applicable)
        - **PDF reports** for sharing with stakeholders

        ### 📁 Sample Data Format

        Your CSV/Excel file should look something like this:

        | Date | Item Name | Quantity Sold | Price | Category |
        |------|-----------|---------------|-------|----------|
        | 2024-01-01 | Widget A | 10 | 25.00 | Electronics |
        | 2024-01-02 | Widget B | 5 | 15.00 | Home |

        Upload your file to get started! 🚀
        """)

if __name__ == "__main__":
    main()