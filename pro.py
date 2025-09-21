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

st.set_page_config(
    page_title="Panel de Análisis de Inventario",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        try:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            else:
                self.data = pd.read_excel(uploaded_file)
            return True
        except Exception as e:
            st.error(f"Error cargando archivo: {str(e)}")
            return False

    def preprocess_data(self):
        if self.data is None:
            return False

        df = self.data.copy()

        df.columns = df.columns.str.lower().str.replace(' ', '_')

        date_columns = [col for col in df.columns if 'date' in col or 'time' in col or 'fecha' in col]
        if date_columns:
            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]], errors='coerce')
            df = df.dropna(subset=[date_columns[0]])
            df['date'] = df[date_columns[0]]
        else:
            st.warning("No se encontró columna de fecha. Asegúrate de que tus datos tengan una columna de fecha.")
            return False

        quantity_cols = [col for col in df.columns if any(word in col for word in ['qty', 'quantity', 'sold', 'units', 'cantidad', 'vendido', 'unidades'])]
        price_cols = [col for col in df.columns if any(word in col for word in ['price', 'cost', 'revenue', 'sales', 'amount', 'precio', 'costo', 'venta', 'importe'])]
        item_cols = [col for col in df.columns if any(word in col for word in ['item', 'product', 'name', 'sku', 'producto', 'nombre', 'articulo'])]
        category_cols = [col for col in df.columns if any(word in col for word in ['category', 'type', 'group', 'class', 'categoria', 'tipo', 'grupo', 'clase'])]

        if quantity_cols:
            df['quantity_sold'] = pd.to_numeric(df[quantity_cols[0]], errors='coerce')
        if price_cols:
            df['sales_amount'] = pd.to_numeric(df[price_cols[0]], errors='coerce')
        if item_cols:
            df['item_name'] = df[item_cols[0]].astype(str)
        if category_cols:
            df['category'] = df[category_cols[0]].astype(str)

        if 'sales_amount' not in df.columns and 'quantity_sold' in df.columns:
            if price_cols:
                df['sales_amount'] = df['quantity_sold'] * pd.to_numeric(df[price_cols[0]], errors='coerce')

        essential_columns = ['date', 'item_name']
        df = df.dropna(subset=essential_columns)

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year_month'] = df['date'].dt.to_period('M')
        df['quarter'] = df['date'].dt.quarter

        self.processed_data = df
        return True

    def generate_insights(self):
        if self.processed_data is None:
            return {}

        df = self.processed_data
        insights = {}

        total_items = df['item_name'].nunique()
        total_sales = df['sales_amount'].sum() if 'sales_amount' in df.columns else 0
        total_quantity = df['quantity_sold'].sum() if 'quantity_sold' in df.columns else 0
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} a {df['date'].max().strftime('%Y-%m-%d')}"

        insights['overview'] = {
            'total_items': total_items,
            'total_sales': total_sales,
            'total_quantity': total_quantity,
            'date_range': date_range
        }

        if 'quantity_sold' in df.columns:
            item_performance = df.groupby('item_name').agg({
                'quantity_sold': 'sum',
                'sales_amount': 'sum' if 'sales_amount' in df.columns else 'count'
            }).reset_index()

            item_performance = item_performance.sort_values('quantity_sold', ascending=False)

            insights['top_items'] = item_performance.head(10)
            insights['bottom_items'] = item_performance.tail(10)

            insights['buy_more'] = item_performance.head(5)['item_name'].tolist()

            insights['reduce_stock'] = item_performance.tail(5)['item_name'].tolist()

        if len(df['year_month'].unique()) > 1:
            monthly_trends = df.groupby(['item_name', 'year_month'])['quantity_sold'].sum().reset_index()
            monthly_trends['year_month'] = monthly_trends['year_month'].astype(str)

            declining_items = []
            for item in df['item_name'].unique():
                item_data = monthly_trends[monthly_trends['item_name'] == item].sort_values('year_month')
                if len(item_data) >= 3:
                    recent_avg = item_data.tail(2)['quantity_sold'].mean()
                    older_avg = item_data.head(2)['quantity_sold'].mean()
                    if recent_avg < older_avg * 0.7:
                        declining_items.append(item)

            insights['watch_out'] = declining_items[:5]

        if 'category' in df.columns:
            category_performance = df.groupby('category').agg({
                'quantity_sold': 'sum',
                'sales_amount': 'sum' if 'sales_amount' in df.columns else 'count'
            }).reset_index()
            insights['category_performance'] = category_performance.sort_values('quantity_sold', ascending=False)

        return insights

def create_visualizations(analyzer):
    if analyzer.processed_data is None:
        return

    df = analyzer.processed_data

    st.subheader("📈 Tendencias de Ventas a lo Largo del Tiempo")

    col1, col2 = st.columns(2)

    with col1:
        monthly_sales = df.groupby('year_month').agg({
            'quantity_sold': 'sum' if 'quantity_sold' in df.columns else 'count',
            'sales_amount': 'sum' if 'sales_amount' in df.columns else 'count'
        }).reset_index()
        monthly_sales['year_month_str'] = monthly_sales['year_month'].astype(str)

        fig_monthly = px.line(monthly_sales, x='year_month_str', y='quantity_sold',
                             title='Tendencia de Cantidad Vendida Mensual',
                             labels={'year_month_str': 'Mes', 'quantity_sold': 'Cantidad Vendida'})
        fig_monthly.update_layout(height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col2:
        if 'sales_amount' in df.columns:
            fig_revenue = px.line(monthly_sales, x='year_month_str', y='sales_amount',
                                 title='Tendencia de Ingresos Mensual',
                                 labels={'year_month_str': 'Mes', 'sales_amount': 'Ingresos ($)'})
            fig_revenue.update_layout(height=400)
            st.plotly_chart(fig_revenue, use_container_width=True)

    st.subheader("🏆 Productos de Mejor Rendimiento")

    col1, col2 = st.columns(2)

    with col1:
        top_items_qty = df.groupby('item_name')['quantity_sold'].sum().sort_values(ascending=False).head(10)
        fig_top_qty = px.bar(x=top_items_qty.values, y=top_items_qty.index, orientation='h',
                            title='Top 10 Productos por Cantidad Vendida',
                            labels={'x': 'Cantidad Vendida', 'y': 'Producto'})
        fig_top_qty.update_layout(height=500)
        st.plotly_chart(fig_top_qty, use_container_width=True)

    with col2:
        if 'sales_amount' in df.columns:
            top_items_revenue = df.groupby('item_name')['sales_amount'].sum().sort_values(ascending=False).head(10)
            fig_top_revenue = px.bar(x=top_items_revenue.values, y=top_items_revenue.index, orientation='h',
                                    title='Top 10 Productos por Ingresos',
                                    labels={'x': 'Ingresos ($)', 'y': 'Producto'})
            fig_top_revenue.update_layout(height=500)
            st.plotly_chart(fig_top_revenue, use_container_width=True)

    if 'category' in df.columns:
        st.subheader("📊 Rendimiento por Categoría")

        col1, col2 = st.columns(2)

        with col1:
            category_qty = df.groupby('category')['quantity_sold'].sum().sort_values(ascending=False)
            fig_cat_qty = px.pie(values=category_qty.values, names=category_qty.index,
                                title='Cantidad Vendida por Categoría')
            st.plotly_chart(fig_cat_qty, use_container_width=True)

        with col2:
            if 'sales_amount' in df.columns:
                category_revenue = df.groupby('category')['sales_amount'].sum().sort_values(ascending=False)
                fig_cat_revenue = px.pie(values=category_revenue.values, names=category_revenue.index,
                                        title='Ingresos por Categoría')
                st.plotly_chart(fig_cat_revenue, use_container_width=True)

    years = df['year'].unique()
    if len(years) > 1:
        st.subheader("📅 Comparación Año contra Año")

        yearly_comparison = df.groupby(['year', 'month']).agg({
            'quantity_sold': 'sum',
            'sales_amount': 'sum' if 'sales_amount' in df.columns else 'count'
        }).reset_index()

        fig_yoy = px.line(yearly_comparison, x='month', y='quantity_sold', color='year',
                         title='Comparación Mensual Año contra Año',
                         labels={'month': 'Mes', 'quantity_sold': 'Cantidad Vendida'})
        st.plotly_chart(fig_yoy, use_container_width=True)

    st.subheader("🗓️ Análisis Estacional")

    seasonal_data = df.groupby(['month'])['quantity_sold'].sum().reset_index()
    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                   'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    seasonal_data['month_name'] = seasonal_data['month'].apply(lambda x: month_names[x-1])

    fig_seasonal = px.bar(seasonal_data, x='month_name', y='quantity_sold',
                         title='Ventas por Mes (Tendencias Estacionales)',
                         labels={'month_name': 'Mes', 'quantity_sold': 'Cantidad Vendida'})
    st.plotly_chart(fig_seasonal, use_container_width=True)

def generate_pdf_report(analyzer, insights):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1f77b4'),
        alignment=1
    )

    story.append(Paragraph("Reporte de Análisis de Inventario", title_style))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Resumen Ejecutivo", styles['Heading2']))

    overview = insights.get('overview', {})
    summary_text = f"""
    Este reporte analiza datos de inventario del {overview.get('date_range', 'N/A')}.
    El análisis cubre {overview.get('total_items', 0)} productos únicos con ventas totales de
    ${overview.get('total_sales', 0):,.2f} y {overview.get('total_quantity', 0):,} unidades vendidas.
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("📈 Comprar Más de Estos Productos", styles['Heading2']))
    if 'buy_more' in insights:
        buy_more_items = insights['buy_more']
        for i, item in enumerate(buy_more_items, 1):
            story.append(Paragraph(f"{i}. {item}", styles['Normal']))
    story.append(Spacer(1, 15))

    story.append(Paragraph("📉 Reducir Stock de Estos Productos", styles['Heading2']))
    if 'reduce_stock' in insights:
        reduce_items = insights['reduce_stock']
        for i, item in enumerate(reduce_items, 1):
            story.append(Paragraph(f"{i}. {item}", styles['Normal']))
    story.append(Spacer(1, 15))

    story.append(Paragraph("⚠️ Vigilar Estos Productos (Ventas Cayendo Rápidamente)", styles['Heading2']))
    if 'watch_out' in insights:
        watch_items = insights.get('watch_out', [])
        if watch_items:
            for i, item in enumerate(watch_items, 1):
                story.append(Paragraph(f"{i}. {item}", styles['Normal']))
        else:
            story.append(Paragraph("No hay productos con tendencias preocupantes.", styles['Normal']))

    if 'category_performance' in insights:
        story.append(PageBreak())
        story.append(Paragraph("Análisis de Rendimiento por Categoría", styles['Heading2']))

        cat_data = insights['category_performance']
        table_data = [['Categoría', 'Cantidad Total Vendida', 'Ingresos Totales']]
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

    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    st.markdown("<h1 class='main-header'>📊 Panel de Análisis de Inventario</h1>", unsafe_allow_html=True)

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = InventoryAnalyzer()

    st.sidebar.header("📁 Cargar Datos")

    uploaded_file = st.sidebar.file_uploader(
        "Selecciona un archivo CSV o Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Sube tu archivo de datos de inventario. Asegúrate de que contenga columnas para fechas, nombres de productos, cantidades y opcionalmente precios/categorías."
    )

    if uploaded_file is not None:
        if st.session_state.analyzer.load_data(uploaded_file):
            st.sidebar.success("✅ ¡Archivo cargado exitosamente!")

            st.sidebar.subheader("Vista Previa de Datos")
            st.sidebar.dataframe(st.session_state.analyzer.data.head())

            if st.session_state.analyzer.preprocess_data():
                df = st.session_state.analyzer.processed_data

                st.subheader("📊 Resumen General")

                insights = st.session_state.analyzer.generate_insights()
                overview = insights.get('overview', {})

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total de Productos", f"{overview.get('total_items', 0):,}")

                with col2:
                    st.metric("Cantidad Total Vendida", f"{overview.get('total_quantity', 0):,}")

                with col3:
                    st.metric("Ingresos Totales", f"${overview.get('total_sales', 0):,.2f}")

                with col4:
                    st.metric("Días de Datos", len(df['date'].dt.date.unique()))

                st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
                st.subheader("🎯 Conocimientos Clave")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**📈 Productos para Comprar Más:**")
                    if 'buy_more' in insights:
                        for item in insights['buy_more'][:5]:
                            st.write(f"• {item}")

                with col2:
                    st.write("**📉 Productos para Reducir Stock:**")
                    if 'reduce_stock' in insights:
                        for item in insights['reduce_stock'][:5]:
                            st.write(f"• {item}")

                with col3:
                    st.write("**⚠️ Productos a Vigilar:**")
                    watch_items = insights.get('watch_out', [])
                    if watch_items:
                        for item in watch_items:
                            st.write(f"• {item}")
                    else:
                        st.write("• No se detectaron tendencias preocupantes")

                st.markdown("</div>", unsafe_allow_html=True)

                create_visualizations(st.session_state.analyzer)

                st.subheader("📄 Exportar Reporte")

                if st.button("Generar Reporte PDF", type="primary"):
                    with st.spinner("Generando reporte PDF..."):
                        pdf_buffer = generate_pdf_report(st.session_state.analyzer, insights)

                        st.download_button(
                            label="📥 Descargar Reporte PDF",
                            data=pdf_buffer,
                            file_name=f"reporte_analisis_inventario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )

                        st.success("✅ ¡Reporte PDF generado exitosamente!")

                st.subheader("💾 Exportar Datos Procesados")

                col1, col2 = st.columns(2)

                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Descargar Datos Procesados (CSV)",
                        data=csv,
                        file_name=f"datos_inventario_procesados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    summary_stats = df.describe()
                    summary_csv = summary_stats.to_csv()
                    st.download_button(
                        label="📈 Descargar Estadísticas Resumen",
                        data=summary_csv,
                        file_name=f"estadisticas_resumen_inventario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("❌ Error procesando datos. Por favor verifica el formato de tu archivo y los nombres de columnas.")
    else:
        st.markdown("""
        ### 📋 Instrucciones

        1. **Sube tus datos de inventario** usando el cargador de archivos en la barra lateral
        2. **Formatos soportados**: CSV, Excel (.xlsx, .xls)
        3. **Columnas requeridas**: Tus datos deben incluir:
           - Columna de fecha (fecha de venta, fecha de transacción, etc.)
           - Nombre del producto/artículo
           - Cantidad vendida
           - Precio/Ingresos (opcional)
           - Categoría (opcional)

        ### 📊 Lo que Obtendrás

        - **Visualizaciones interactivas** mostrando tendencias y patrones de ventas
        - **Conocimientos accionables** sobre qué productos comprar más o menos
        - **Análisis de categorías** para entender qué categorías impulsan las ventas
        - **Tendencias estacionales** para identificar períodos de mayor venta
        - **Comparaciones año contra año** (cuando sea aplicable)
        - **Reportes PDF** para compartir con stakeholders

        ### 📁 Formato de Datos de Ejemplo

        Tu archivo CSV/Excel debe verse algo así:

        | Fecha | Nombre Producto | Cantidad Vendida | Precio | Categoría |
        |-------|-----------------|------------------|--------|-----------|
        | 2024-01-01 | Producto A | 10 | 25.00 | Electrónicos |
        | 2024-01-02 | Producto B | 5 | 15.00 | Hogar |

        ¡Sube tu archivo para comenzar! 🚀
        """)

if __name__ == "__main__":
    main()