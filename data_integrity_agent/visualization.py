"""
Visualization module for Data Integrity Agent
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import base64
from io import BytesIO

class DataIntegrityVisualizer:
    """Creates visualizations for data integrity validation results"""
    
    def __init__(self, theme: str = 'plotly_white'):
        self.theme = theme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#8c564b'
        }
    
    def create_comprehensive_report(self, results: Dict[str, Any], df: pd.DataFrame, 
                                  output_path: str, title: str = "Data Integrity Report") -> str:
        """Create a comprehensive HTML report with all visualizations"""
        
        # Create the main figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Data Quality Score', 'Missing Values by Column',
                'Outlier Distribution', 'Data Types Distribution',
                'Sample Data Preview', 'Validation Summary'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "pie"}],
                [{"type": "table"}, {"type": "bar"}]
            ]
        )
        
        # 1. Data Quality Score (Gauge)
        if 'data_quality_score' in results:
            score = results['data_quality_score']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Data Quality Score"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self._get_score_color(score)},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 70], 'color': "yellow"},
                            {'range': [70, 90], 'color': "lightgreen"},
                            {'range': [90, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=1, col=1
            )
        
        # 2. Missing Values by Column
        if 'validation_results' in results and 'missing_values' in results['validation_results']:
            missing_data = results['validation_results']['missing_values']
            if 'percentages' in missing_data:
                columns = list(missing_data['percentages'].keys())
                percentages = list(missing_data['percentages'].values())
                
                fig.add_trace(
                    go.Bar(
                        x=columns,
                        y=percentages,
                        name='Missing %',
                        marker_color=self.colors['warning'],
                        text=[f'{p:.1f}%' for p in percentages],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
        
        # 3. Outlier Distribution
        if 'validation_results' in results and 'outliers' in results['validation_results']:
            outliers_data = results['validation_results']['outliers']
            outlier_cols = list(outliers_data.keys())
            outlier_counts = [outliers_data[col]['count'] for col in outlier_cols]
            
            fig.add_trace(
                go.Bar(
                    x=outlier_cols,
                    y=outlier_counts,
                    name='Outlier Count',
                    marker_color=self.colors['secondary'],
                    text=outlier_counts,
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # 4. Data Types Distribution
        dtype_counts = df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(
                labels=dtype_counts.index.astype(str),
                values=dtype_counts.values,
                name="Data Types",
                hole=0.3
            ),
            row=2, col=2
        )
        
        # 5. Sample Data Preview
        sample_df = df.head(10)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Column'] + list(sample_df.columns),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[sample_df.index] + [sample_df[col] for col in sample_df.columns],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=3, col=1
        )
        
        # 6. Validation Summary
        validation_summary = self._create_validation_summary(results)
        fig.add_trace(
            go.Bar(
                x=list(validation_summary.keys()),
                y=list(validation_summary.values()),
                name='Validation Results',
                marker_color=[self.colors['success'] if v == 'Pass' else self.colors['warning'] for v in validation_summary.values()],
                text=list(validation_summary.values()),
                textposition='auto'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text=title,
            template=self.theme
        )
        
        # Save the plot
        output_path = Path(output_path)
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    def create_missing_values_heatmap(self, df: pd.DataFrame, output_path: str) -> str:
        """Create a heatmap showing missing values pattern"""
        
        # Create missing values matrix
        missing_matrix = df.isnull().astype(int)
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix.values,
            x=missing_matrix.columns,
            y=missing_matrix.index,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Missing Value")
        ))
        
        fig.update_layout(
            title="Missing Values Heatmap",
            xaxis_title="Columns",
            yaxis_title="Rows",
            height=600
        )
        
        output_path = Path(output_path)
        fig.write_html(str(output_path))
        return str(output_path)
    
    def create_outlier_analysis(self, df: pd.DataFrame, results: Dict[str, Any], output_path: str) -> str:
        """Create outlier analysis visualizations"""
        
        if 'validation_results' not in results or 'outliers' not in results['validation_results']:
            return ""
        
        outliers_data = results['validation_results']['outliers']
        numeric_cols = [col for col in outliers_data.keys() if col in df.select_dtypes(include=[np.number]).columns]
        
        if not numeric_cols:
            return ""
        
        # Create subplots for each numeric column
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols,
            specs=[[{"type": "box"}] * n_cols] * n_rows
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Box(
                    y=df[col].dropna(),
                    name=col,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="Outlier Analysis by Column",
            showlegend=False
        )
        
        output_path = Path(output_path)
        fig.write_html(str(output_path))
        return str(output_path)
    
    def create_distribution_plots(self, df: pd.DataFrame, output_path: str) -> str:
        """Create distribution plots for numeric columns"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return ""
        
        n_cols = min(2, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols,
            specs=[[{"type": "histogram"}] * n_cols] * n_rows
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    name=col,
                    nbinsx=30
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="Distribution Analysis",
            showlegend=False
        )
        
        output_path = Path(output_path)
        fig.write_html(str(output_path))
        return str(output_path)
    
    def create_correlation_heatmap(self, df: pd.DataFrame, output_path: str) -> str:
        """Create correlation heatmap for numeric columns"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return ""
        
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            height=600
        )
        
        output_path = Path(output_path)
        fig.write_html(str(output_path))
        return str(output_path)
    
    def create_interactive_dashboard(self, results: Dict[str, Any], df: pd.DataFrame, 
                                   output_path: str) -> str:
        """Create an interactive dashboard with multiple visualizations"""
        
        # Create a comprehensive dashboard
        dashboard_html = self._generate_dashboard_html(results, df)
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            f.write(dashboard_html)
        
        return str(output_path)
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on quality score"""
        if score >= 90:
            return self.colors['success']
        elif score >= 70:
            return self.colors['info']
        elif score >= 50:
            return self.colors['warning']
        else:
            return self.colors['light']
    
    def _create_validation_summary(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Create validation summary for visualization"""
        summary = {}
        
        if 'validation_results' in results:
            validation = results['validation_results']
            
            # Missing values
            if 'missing_values' in validation:
                missing = validation['missing_values']
                total_missing = missing.get('total_missing', 0)
                summary['Missing Values'] = 'Pass' if total_missing == 0 else 'Fail'
            
            # Outliers
            if 'outliers' in validation:
                outliers = validation['outliers']
                total_outliers = sum(info['count'] for info in outliers.values())
                summary['Outliers'] = 'Pass' if total_outliers == 0 else 'Warning'
            
            # Format validation
            if 'format_validation' in validation:
                format_issues = validation['format_validation']
                total_format_issues = sum(info['invalid_count'] for info in format_issues.values())
                summary['Format Issues'] = 'Pass' if total_format_issues == 0 else 'Fail'
        
        return summary
    
    def _generate_dashboard_html(self, results: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate interactive dashboard HTML"""
        
        # Create individual plots
        plots = {}
        
        # Quality score gauge
        if 'data_quality_score' in results:
            score = results['data_quality_score']
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Data Quality Score"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_score_color(score)},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "yellow"},
                        {'range': [70, 90], 'color': "lightgreen"},
                        {'range': [90, 100], 'color': "green"}
                    ]
                }
            ))
            plots['gauge'] = gauge_fig.to_html(full_html=False, include_plotlyjs=False)
        
        # Missing values chart
        if 'validation_results' in results and 'missing_values' in results['validation_results']:
            missing_data = results['validation_results']['missing_values']
            if 'percentages' in missing_data:
                missing_fig = go.Figure(go.Bar(
                    x=list(missing_data['percentages'].keys()),
                    y=list(missing_data['percentages'].values()),
                    marker_color=self.colors['warning']
                ))
                missing_fig.update_layout(title="Missing Values by Column")
                plots['missing'] = missing_fig.to_html(full_html=False, include_plotlyjs=False)
        
        # Generate dashboard HTML
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Integrity Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .recommendations {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .recommendation-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Data Integrity Dashboard</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>Data Quality Score</h3>
                {plots.get('gauge', '<p>No score available</p>')}
            </div>
            
            <div class="metric-card">
                <h3>Missing Values Analysis</h3>
                {plots.get('missing', '<p>No missing values data</p>')}
            </div>
        </div>
        
        <div class="recommendations">
            <h3>Recommendations</h3>
            {self._format_recommendations(results)}
        </div>
    </div>
</body>
</html>
        """
        
        return dashboard_html
    
    def _format_recommendations(self, results: Dict[str, Any]) -> str:
        """Format recommendations for HTML display"""
        if 'recommendations' not in results:
            return "<p>No recommendations available</p>"
        
        recommendations_html = ""
        for i, rec in enumerate(results['recommendations'], 1):
            recommendations_html += f"""
            <div class="recommendation-item">
                <strong>{i}.</strong> {rec}
            </div>
            """
        
        return recommendations_html
