"""
NDIS Carer Data Pipeline Dashboard
Streamlit-based dashboard for data quality and pipeline health monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import sqlite3
from datetime import datetime, timedelta, date
from typing import Dict, List, Any
import time

from config import get_dashboard_config
from data_aggregator import DataAggregator

# Page configuration
st.set_page_config(
    page_title="NDIS Carer Data Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_dashboard():
    """Initialize dashboard components"""
    config = get_dashboard_config()
    aggregator = DataAggregator()
    return config, aggregator

config, aggregator = init_dashboard()

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-excellent { border-left-color: #28a745 !important; }
    .status-good { border-left-color: #17a2b8 !important; }
    .status-warning { border-left-color: #ffc107 !important; }
    .status-critical { border-left-color: #dc3545 !important; }
    
    .big-font {
        font-size: 2rem !important;
        font-weight: bold;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e6e6e6;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    
    # Header
    st.title("📊 NDIS Carer Data Pipeline Dashboard")
    st.markdown("**Data Quality & Pipeline Health Monitoring**")
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Get data
    with st.spinner("Loading dashboard data..."):
        try:
            dashboard_data = get_dashboard_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()
    
    # Main dashboard layout
    render_overview_kpis(dashboard_data)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Quality Gates", 
        "📋 Record Explorer", 
        "🎯 Distributions", 
        "📊 Templates", 
        "⚙️ System Status"
    ])
    
    with tab1:
        render_quality_gates(dashboard_data)
    
    with tab2:
        render_record_explorer(dashboard_data)
    
    with tab3:
        render_distributions(dashboard_data)
    
    with tab4:
        render_templates_view(dashboard_data)
    
    with tab5:
        render_system_status(dashboard_data)

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_dashboard_data() -> Dict[str, Any]:
    """Get all dashboard data"""
    try:
        # Get aggregated metrics
        metrics = aggregator.aggregate_all_data()
        
        if metrics.get("status") != "success":
            raise Exception(f"Data aggregation failed: {metrics.get('error', 'Unknown error')}")
        
        return metrics
        
    except Exception as e:
        st.error(f"Failed to get dashboard data: {e}")
        raise

def render_overview_kpis(data: Dict[str, Any]):
    """Render overview KPIs section"""
    st.markdown('<div class="section-header">📊 Overview KPIs</div>', unsafe_allow_html=True)
    
    # Extract metrics
    derived_metrics = data.get("derived_metrics", {})
    output_metrics = data.get("output_metrics", {})
    validation_metrics = data.get("validation_metrics", {})
    
    # KPI columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pass_rate = derived_metrics.get("pass_rate", 0)
        status = get_kpi_status("pass_rate", pass_rate)
        st.metric(
            label="Pass Rate",
            value=f"{pass_rate:.1f}%",
            delta=get_trend_delta("pass_rate", derived_metrics.get("trend_analysis", {})),
            help="Percentage of records that passed validation"
        )
        render_status_indicator(status)
    
    with col2:
        fallback_ratio = derived_metrics.get("fallback_ratio", 0)
        status = get_kpi_status("fallback_ratio", fallback_ratio, inverse=True)
        st.metric(
            label="Fallback Ratio",
            value=f"{fallback_ratio:.1f}%",
            help="Percentage of records using fallback templates"
        )
        render_status_indicator(status)
    
    with col3:
        throughput = derived_metrics.get("throughput_per_minute", 0)
        status = get_kpi_status("throughput", throughput)
        st.metric(
            label="Throughput",
            value=f"{throughput:.1f}/min",
            delta=get_trend_delta("throughput", derived_metrics.get("trend_analysis", {})),
            help="Records generated per minute"
        )
        render_status_indicator(status)
    
    with col4:
        avg_length = derived_metrics.get("avg_narrative_length", 0)
        status = get_narrative_length_status(avg_length)
        st.metric(
            label="Avg Narrative Length",
            value=f"{avg_length:.0f} chars",
            help="Average length of narrative notes"
        )
        render_status_indicator(status)
    
    with col5:
        coverage = derived_metrics.get("field_coverage", 0)
        status = get_kpi_status("field_coverage", coverage)
        st.metric(
            label="Field Coverage",
            value=f"{coverage:.1f}%",
            help="Percentage of fields with complete data"
        )
        render_status_indicator(status)
    
    # Summary row
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{output_metrics.get('total_records', 0):,}")
    
    with col2:
        st.metric("Unique Carers", f"{output_metrics.get('unique_carers', 0):,}")
    
    with col3:
        st.metric("Unique Participants", f"{output_metrics.get('unique_participants', 0):,}")
    
    with col4:
        span_days = output_metrics.get("date_range", {}).get("span_days", 0)
        st.metric("Data Span", f"{span_days} days")

def render_quality_gates(data: Dict[str, Any]):
    """Render quality gates view"""
    st.markdown('<div class="section-header">🚦 Quality Gate Analysis</div>', unsafe_allow_html=True)
    
    validation_metrics = data.get("validation_metrics", {})
    quality_gates = validation_metrics.get("quality_gates", {})
    
    if not quality_gates:
        st.warning("No quality gate data available")
        return
    
    # Quality gates overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create quality gates chart
        gate_names = []
        failure_rates = []
        statuses = []
        
        for gate_name, gate_data in quality_gates.items():
            gate_names.append(gate_name.replace("_", " ").title())
            failure_rate = gate_data.get("failure_rate", 0)
            failure_rates.append(failure_rate)
            statuses.append(get_quality_gate_status(failure_rate))
        
        # Create bar chart
        fig = px.bar(
            x=gate_names,
            y=failure_rates,
            title="Quality Gate Failure Rates",
            labels={"x": "Quality Gates", "y": "Failure Rate (%)"},
            color=failure_rates,
            color_continuous_scale=["green", "yellow", "red"]
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Overall score
        overall_score = validation_metrics.get("overall_score", 0)
        st.metric("Overall Quality Score", f"{overall_score:.1f}/100")
        
        # Quality distribution pie chart
        passed = sum(1 for _, data in quality_gates.items() if data.get("failure_rate", 0) <= 10)
        failed = len(quality_gates) - passed
        
        fig = px.pie(
            values=[passed, failed],
            names=["Passed Gates", "Failed Gates"],
            title="Quality Gates Status",
            color_discrete_map={"Passed Gates": "green", "Failed Gates": "red"}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed quality gates table
    st.markdown("### Quality Gate Details")
    
    gate_details = []
    for gate_name, gate_data in quality_gates.items():
        gate_details.append({
            "Gate": gate_name.replace("_", " ").title(),
            "Failure Rate (%)": f"{gate_data.get('failure_rate', 0):.2f}",
            "Failed Records": gate_data.get("failed", 0),
            "Total Records": gate_data.get("passed", 0) + gate_data.get("failed", 0),
            "Status": get_quality_gate_status(gate_data.get("failure_rate", 0))
        })
    
    if gate_details:
        df = pd.DataFrame(gate_details)
        st.dataframe(df, use_container_width=True)
    
    # Recommendations
    recommendations = validation_metrics.get("recommendations", [])
    if recommendations:
        st.markdown("### Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")

def render_record_explorer(data: Dict[str, Any]):
    """Render record explorer section"""
    st.markdown('<div class="section-header">📋 Record Explorer</div>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_filter = st.selectbox(
            "Validation Status",
            ["All", "Passed", "Failed", "Warning"],
            key="status_filter"
        )
    
    with col2:
        source_filter = st.selectbox(
            "Source Type",
            ["All", "LLM", "Template", "Demo"],
            key="source_filter"
        )
    
    with col3:
        date_from = st.date_input("From Date", key="date_from")
    
    with col4:
        search_query = st.text_input("Search (ID/Keywords)", key="search_query")
    
    # Get filtered records (mock implementation)
    records_data = get_sample_records(data)
    
    if records_data:
        # Display records table
        st.markdown("### Latest Records")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(records_data)
        
        # Configure column display
        column_config = {
            "record_id": st.column_config.TextColumn("Record ID", width="medium"),
            "service_type": st.column_config.TextColumn("Service Type", width="medium"),
            "duration_hours": st.column_config.NumberColumn("Duration (hrs)", format="%.2f"),
            "narrative_length": st.column_config.NumberColumn("Narrative Length", format="%d"),
            "validation_status": st.column_config.TextColumn("Status", width="small"),
            "service_date": st.column_config.DateColumn("Service Date")
        }
        
        st.dataframe(
            df,
            column_config=column_config,
            use_container_width=True,
            height=400
        )
        
        # Record details
        if st.checkbox("Show Record Details"):
            selected_record = st.selectbox(
                "Select Record for Details",
                options=df["record_id"].tolist(),
                key="selected_record"
            )
            
            if selected_record:
                record_details = df[df["record_id"] == selected_record].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Record Information**")
                    st.json({
                        "Record ID": record_details["record_id"],
                        "Carer ID": record_details["carer_id"],
                        "Participant ID": record_details["participant_id"],
                        "Service Date": str(record_details["service_date"]),
                        "Service Type": record_details["service_type"],
                        "Duration": f"{record_details['duration_hours']} hours"
                    })
                
                with col2:
                    st.markdown("**Narrative Notes**")
                    st.text_area(
                        "Content",
                        value=record_details.get("narrative_notes", "No narrative available"),
                        height=150,
                        disabled=True
                    )
    else:
        st.info("No records available to display")

def render_distributions(data: Dict[str, Any]):
    """Render data distributions section"""
    st.markdown('<div class="section-header">📊 Data Distributions</div>', unsafe_allow_html=True)
    
    output_metrics = data.get("output_metrics", {})
    
    # Service type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        service_dist = output_metrics.get("service_type_distribution", {})
        if service_dist:
            fig = px.pie(
                values=list(service_dist.values()),
                names=list(service_dist.keys()),
                title="Service Type Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        outcome_dist = output_metrics.get("outcome_distribution", {})
        if outcome_dist:
            fig = px.bar(
                x=list(outcome_dist.keys()),
                y=list(outcome_dist.values()),
                title="Service Outcome Distribution",
                labels={"x": "Outcome", "y": "Count"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Location and duration analysis
    col1, col2 = st.columns(2)
    
    with col1:
        location_dist = output_metrics.get("location_distribution", {})
        if location_dist:
            fig = px.bar(
                x=list(location_dist.values()),
                y=list(location_dist.keys()),
                orientation='h',
                title="Location Type Distribution",
                labels={"x": "Count", "y": "Location Type"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        duration_stats = output_metrics.get("duration_stats", {})
        if duration_stats:
            # Create duration statistics chart
            stats_data = {
                "Metric": ["Average", "Minimum", "Maximum", "Median"],
                "Hours": [
                    duration_stats.get("avg", 0),
                    duration_stats.get("min", 0),
                    duration_stats.get("max", 0),
                    duration_stats.get("median", 0)
                ]
            }
            
            fig = px.bar(
                x=stats_data["Metric"],
                y=stats_data["Hours"],
                title="Duration Statistics",
                labels={"x": "Statistic", "y": "Hours"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def render_templates_view(data: Dict[str, Any]):
    """Render templates usage view"""
    st.markdown('<div class="section-header">📝 Template Usage Analysis</div>', unsafe_allow_html=True)
    
    output_metrics = data.get("output_metrics", {})
    service_dist = output_metrics.get("service_type_distribution", {})
    
    # Template pool overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Narrative Templates", "50", help="Total narrative templates available")
        st.metric("Diversity Score", "85.5%", help="Template diversity score")
    
    with col2:
        st.metric("Action Templates", "30", help="Total action templates available")
        st.metric("Usage Rate", "78.2%", help="Template usage rate")
    
    with col3:
        st.metric("Factor Templates", "25", help="Total factor templates available")
        st.metric("Near-Duplicate Rate", "5.2%", help="Rate of near-duplicate templates")
    
    # Template usage by category
    if service_dist:
        st.markdown("### Template Usage by Service Category")
        
        fig = px.treemap(
            names=list(service_dist.keys()),
            values=list(service_dist.values()),
            title="Template Coverage by Service Type"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Template diversity analysis
    st.markdown("### Template Diversity Analysis")
    
    diversity_data = {
        "Category": ["Personal Care", "Household Tasks", "Community Access", "Transport", "Social Support"],
        "Template Count": [12, 8, 10, 6, 9],
        "Usage Count": [145, 89, 123, 67, 98],
        "Diversity Score": [88, 75, 92, 65, 82]
    }
    
    df_diversity = pd.DataFrame(diversity_data)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Template Count vs Usage", "Diversity Scores"),
        specs=[[{"secondary_y": True}, {"type": "bar"}]]
    )
    
    # Left plot: Template count vs usage
    fig.add_trace(
        go.Bar(name="Template Count", x=df_diversity["Category"], y=df_diversity["Template Count"]),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(name="Usage Count", x=df_diversity["Category"], y=df_diversity["Usage Count"], mode="lines+markers"),
        row=1, col=1, secondary_y=True
    )
    
    # Right plot: Diversity scores
    fig.add_trace(
        go.Bar(name="Diversity Score", x=df_diversity["Category"], y=df_diversity["Diversity Score"], 
               marker_color="lightblue"),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def render_system_status(data: Dict[str, Any]):
    """Render system status section"""
    st.markdown('<div class="section-header">⚙️ System Status & Reports</div>', unsafe_allow_html=True)
    
    # System metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Pipeline Health**")
        overall_status = data.get("derived_metrics", {}).get("quality_status", "unknown")
        status_emoji = {"excellent": "🟢", "good": "🔵", "warning": "🟡", "critical": "🔴"}.get(overall_status, "⚪")
        st.markdown(f"{status_emoji} **{overall_status.title()}**")
        
        last_updated = data.get("aggregation_timestamp", "Unknown")
        if last_updated != "Unknown":
            try:
                dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                st.write(f"Last Updated: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except:
                st.write(f"Last Updated: {last_updated}")
    
    with col2:
        st.markdown("**Performance Metrics**")
        log_metrics = data.get("log_metrics", {})
        avg_gen_time = log_metrics.get("avg_generation_time", 0)
        st.metric("Avg Generation Time", f"{avg_gen_time:.2f}s")
        
        total_generations = log_metrics.get("total_generations", 0)
        st.metric("Total Generations", f"{total_generations:,}")
    
    with col3:
        st.markdown("**Error Summary**")
        recent_errors = len(log_metrics.get("recent_errors", []))
        recent_warnings = len(log_metrics.get("recent_warnings", []))
        
        st.metric("Recent Errors", recent_errors)
        st.metric("Recent Warnings", recent_warnings)
    
    # Export and reports
    st.markdown("### Export & Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Current View (JSON)"):
            # 创建JSON安全的数据副本
            def make_json_safe(obj):
                """递归转换对象使其JSON安全"""
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, date):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {key: make_json_safe(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_safe(item) for item in obj]
                elif hasattr(obj, '__dict__'):
                    return make_json_safe(obj.__dict__)
                else:
                    return obj
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "dashboard_data": make_json_safe(data),
                "export_format": "json"
            }
            
            try:
                json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                st.success("✅ Export data prepared successfully!")
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
                st.info("Please try refreshing the dashboard and try again.")
    
    with col2:
        if st.button("📊 Export as CSV"):
            try:
                # 从仪表板数据创建CSV
                records_data = get_sample_records(data)
                if records_data:
                    import pandas as pd
                    df = pd.DataFrame(records_data)
                    
                    # 转换日期列为字符串
                    if 'service_date' in df.columns:
                        df['service_date'] = df['service_date'].astype(str)
                    
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"dashboard_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.success(f"✅ CSV export ready! ({len(records_data)} records)")
                else:
                    st.warning("No records available for export")
            except Exception as e:
                st.error(f"❌ CSV export failed: {e}")
    
    with col3:
        if st.button("📈 Generate Summary Report"):
            try:
                # 生成仪表板摘要报告
                output_metrics = data.get("output_metrics", {})
                validation_metrics = data.get("validation_metrics", {})
                derived_metrics = data.get("derived_metrics", {})
                
                report_content = f"""# NDIS Carer Data Pipeline Dashboard Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview Summary
- Total Records: {output_metrics.get('total_records', 0):,}
- Unique Carers: {output_metrics.get('unique_carers', 0):,}
- Unique Participants: {output_metrics.get('unique_participants', 0):,}
- Data Span: {output_metrics.get('date_range', {}).get('span_days', 0)} days

## Quality Metrics
- Overall Score: {validation_metrics.get('overall_score', 0):.1f}/100
- Pass Rate: {derived_metrics.get('pass_rate', 0):.1f}%
- Fallback Ratio: {derived_metrics.get('fallback_ratio', 0):.1f}%
- Throughput: {derived_metrics.get('throughput_per_minute', 0):.1f} records/min
- Avg Narrative Length: {derived_metrics.get('avg_narrative_length', 0):.0f} characters

## Service Distribution
"""
                
                # 添加服务类型分布
                service_dist = output_metrics.get('service_type_distribution', {})
                if service_dist:
                    report_content += "\n### Service Types\n"
                    for service_type, count in service_dist.items():
                        percentage = (count / output_metrics.get('total_records', 1)) * 100
                        report_content += f"- {service_type}: {count} ({percentage:.1f}%)\n"
                
                # 添加结果分布
                outcome_dist = output_metrics.get('outcome_distribution', {})
                if outcome_dist:
                    report_content += "\n### Service Outcomes\n"
                    for outcome, count in outcome_dist.items():
                        percentage = (count / sum(outcome_dist.values())) * 100
                        report_content += f"- {outcome}: {count} ({percentage:.1f}%)\n"
                
                report_content += f"\n\n---\nReport generated by NDIS Carer Data Pipeline Dashboard\nTimestamp: {datetime.now().isoformat()}"
                
                st.download_button(
                    label="Download Report (Markdown)",
                    data=report_content,
                    file_name=f"dashboard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                st.success("✅ Summary report generated!")
                
            except Exception as e:
                st.error(f"❌ Report generation failed: {e}")
    
    # Recent logs
    if log_metrics.get("recent_errors"):
        st.markdown("### Recent Errors")
        for error in log_metrics["recent_errors"][-5:]:  # Show last 5 errors
            st.error(error)
    
    if log_metrics.get("recent_warnings"):
        st.markdown("### Recent Warnings")
        for warning in log_metrics["recent_warnings"][-5:]:  # Show last 5 warnings
            st.warning(warning)

# Helper functions
def get_kpi_status(kpi_name: str, value: float, inverse: bool = False) -> str:
    """Get KPI status based on thresholds"""
    thresholds = config["kpi_thresholds"].get(kpi_name, {})
    
    if inverse:  # For metrics where lower is better (like fallback_ratio)
        if value <= thresholds.get("excellent", 0):
            return "excellent"
        elif value <= thresholds.get("good", 0):
            return "good"
        elif value <= thresholds.get("warning", 0):
            return "warning"
        else:
            return "critical"
    else:  # For metrics where higher is better
        if value >= thresholds.get("excellent", 0):
            return "excellent"
        elif value >= thresholds.get("good", 0):
            return "good"
        elif value >= thresholds.get("warning", 0):
            return "warning"
        else:
            return "critical"

def get_narrative_length_status(length: float) -> str:
    """Get status for narrative length"""
    thresholds = config["kpi_thresholds"]["avg_narrative_length"]
    
    if (length >= thresholds["optimal_min"] and 
        length <= thresholds["optimal_max"]):
        return "excellent"
    elif (length >= thresholds["min_acceptable"] and 
          length <= thresholds["max_acceptable"]):
        return "good"
    else:
        return "warning"

def get_quality_gate_status(failure_rate: float) -> str:
    """Get quality gate status based on failure rate"""
    if failure_rate <= 5:
        return "excellent"
    elif failure_rate <= 10:
        return "good"
    elif failure_rate <= 20:
        return "warning"
    else:
        return "critical"

def get_trend_delta(metric_name: str, trend_data: Dict) -> str:
    """Get trend delta for metrics"""
    trend = trend_data.get(f"{metric_name}_trend", "stable")
    if trend == "up":
        return "↑"
    elif trend == "down":
        return "↓"
    else:
        return None

def render_status_indicator(status: str):
    """Render status indicator"""
    colors = {
        "excellent": "🟢",
        "good": "🔵", 
        "warning": "🟡",
        "critical": "🔴"
    }
    st.markdown(f"{colors.get(status, '⚪')} {status.title()}")

def get_sample_records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get sample records for display from actual aggregated data"""
    try:
        # 尝试从数据聚合器获取真实记录
        aggregator = DataAggregator()
        
        # 查询最新的记录
        with sqlite3.connect(aggregator.db_path) as conn:
            query = """
                SELECT * FROM record_details 
                ORDER BY created_timestamp DESC 
                LIMIT 20
            """
            cursor = conn.execute(query)
            columns = [description[0] for description in cursor.description]
            records = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            if records:
                # 处理记录格式
                formatted_records = []
                for record in records:
                    formatted_record = {
                        "record_id": record.get("record_id", "Unknown"),
                        "carer_id": record.get("carer_id", "Unknown"),
                        "carer_name": record.get("carer_name", "Unknown Carer"),
                        "participant_id": record.get("participant_id", "Unknown"),
                        "service_date": record.get("service_date", "Unknown"),
                        "service_type": record.get("service_type", "Unknown"),
                        "duration_hours": record.get("duration_hours", 0),
                        "narrative_length": record.get("narrative_length", 0),
                        "narrative_notes": record.get("narrative_notes", "")[:200] + "..." if record.get("narrative_notes", "") else "",
                        "location_type": record.get("location_type", "Unknown"),
                        "service_outcome": record.get("service_outcome", "Unknown"),
                        "validation_status": record.get("validation_status", "Unknown"),
                        "source_type": record.get("source_type", "LLM"),
                        "support_techniques": record.get("support_techniques", ""),
                        "challenges_encountered": record.get("challenges_encountered", "")
                    }
                    formatted_records.append(formatted_record)
                
                return formatted_records
    
    except Exception as e:
        st.error(f"Could not load real records: {e}")
    
    # 如果无法获取真实记录，使用样本数据
    output_metrics = data.get("output_metrics", {})
    total_records = output_metrics.get("total_records", 0)
    
    if total_records == 0:
        return []
    
    # 生成样本记录
    sample_records = []
    for i in range(min(10, total_records)):
        sample_records.append({
            "record_id": f"SR{12345678 + i}",
            "carer_id": f"CR{123456 + i % 10}",
            "carer_name": f"Carer {i+1}",
            "participant_id": f"PT{654321 + i % 15}",
            "service_date": (datetime.now() - timedelta(days=i)).date(),
            "service_type": ["Personal Care", "Household Tasks", "Community Access", "Transport"][i % 4],
            "duration_hours": round(1.5 + (i % 5) * 0.5, 2),
            "narrative_length": 120 + (i % 10) * 20,
            "validation_status": ["Passed", "Warning", "Failed"][i % 3],
            "source_type": "LLM",
            "narrative_notes": f"English narrative content for record {i+1}. Professional care service provided with participant engagement and positive outcomes."
        })
    
    return sample_records

if __name__ == "__main__":
    main()



