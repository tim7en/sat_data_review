#!/usr/bin/env python3
"""
Satellite Data Review - Comprehensive JSON Data Analysis Tool

This tool parses JSON files containing satellite data analysis results and provides
comprehensive data analysis with tables and visualizations.
"""

import json
import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from tabulate import tabulate
from datetime import datetime


class SatelliteDataAnalyzer:
    """Main class for analyzing satellite data from JSON files."""
    
    def __init__(self, json_file_path):
        """Initialize the analyzer with a JSON file path."""
        self.json_file_path = json_file_path
        self.data = None
        self.df = None
        self.analysis_results = {}
        
    def load_json_data(self):
        """Load and parse JSON data from file."""
        try:
            with open(self.json_file_path, 'r') as f:
                self.data = json.load(f)
            print(f"✓ Successfully loaded JSON data from {self.json_file_path}")
            return True
        except FileNotFoundError:
            print(f"✗ Error: File {self.json_file_path} not found")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Error: Invalid JSON format - {e}")
            return False
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return False
    
    def convert_to_dataframe(self):
        """Convert JSON data to pandas DataFrame for analysis."""
        try:
            if isinstance(self.data, dict):
                # If data is a dict, try to find tabular data
                if 'data' in self.data:
                    self.df = pd.DataFrame(self.data['data'])
                elif 'results' in self.data:
                    self.df = pd.DataFrame(self.data['results'])
                else:
                    # Convert dict to DataFrame with one row
                    self.df = pd.DataFrame([self.data])
            elif isinstance(self.data, list):
                # If data is a list, convert directly
                self.df = pd.DataFrame(self.data)
            else:
                print("✗ Error: Unsupported JSON structure")
                return False
                
            print(f"✓ Converted to DataFrame: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"✗ Error converting to DataFrame: {e}")
            return False
    
    def basic_data_analysis(self):
        """Perform basic data analysis and statistics."""
        if self.df is None:
            return False
            
        analysis = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum()
        }
        
        # Numeric statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['numeric_summary'] = self.df[numeric_cols].describe().to_dict()
        
        # Categorical statistics
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            analysis['categorical_summary'] = {}
            for col in categorical_cols:
                analysis['categorical_summary'][col] = {
                    'unique_values': self.df[col].nunique(),
                    'top_values': self.df[col].value_counts().head().to_dict()
                }
        
        self.analysis_results = analysis
        return True
    
    def print_data_summary(self):
        """Print comprehensive data summary in table format."""
        if not self.analysis_results:
            print("No analysis results available")
            return
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE DATA ANALYSIS REPORT")
        print("="*80)
        
        # Basic info
        print(f"\n📁 File: {self.json_file_path}")
        print(f"📏 Shape: {self.analysis_results['shape'][0]} rows × {self.analysis_results['shape'][1]} columns")
        print(f"💾 Memory Usage: {self.analysis_results['memory_usage'] / 1024:.2f} KB")
        
        # Column information
        print(f"\n🗂️  COLUMN INFORMATION")
        print("-" * 40)
        col_info = []
        for col, dtype in self.analysis_results['dtypes'].items():
            missing = self.analysis_results['missing_values'][col]
            missing_pct = (missing / self.analysis_results['shape'][0]) * 100
            col_info.append([col, str(dtype), missing, f"{missing_pct:.1f}%"])
        
        print(tabulate(col_info, headers=['Column', 'Type', 'Missing', 'Missing %'], tablefmt='grid'))
        
        # Numeric summary
        if 'numeric_summary' in self.analysis_results:
            print(f"\n📈 NUMERIC DATA SUMMARY")
            print("-" * 40)
            numeric_df = pd.DataFrame(self.analysis_results['numeric_summary'])
            print(tabulate(numeric_df.round(3), headers=numeric_df.columns, tablefmt='grid'))
        
        # Categorical summary
        if 'categorical_summary' in self.analysis_results:
            print(f"\n📋 CATEGORICAL DATA SUMMARY")
            print("-" * 40)
            for col, summary in self.analysis_results['categorical_summary'].items():
                print(f"\n{col}:")
                print(f"  Unique values: {summary['unique_values']}")
                print("  Top values:")
                for value, count in summary['top_values'].items():
                    print(f"    {value}: {count}")
        
        # Data sample
        print(f"\n📄 DATA SAMPLE (First 5 rows)")
        print("-" * 40)
        if len(self.df) > 0:
            print(tabulate(self.df.head(), headers=self.df.columns, tablefmt='grid'))
        
    def create_visualizations(self, output_dir="analysis_output"):
        """Create comprehensive visualizations of the data."""
        if self.df is None:
            print("No data available for visualization")
            return False
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        print(f"\n🎨 Creating visualizations in '{output_dir}' directory...")
        
        # 1. Data overview visualization
        if len(numeric_cols) > 0 or len(categorical_cols) > 0:
            self._create_data_overview(output_dir)
        
        # 2. Numeric data visualizations
        if len(numeric_cols) > 0:
            self._create_numeric_visualizations(numeric_cols, output_dir)
        
        # 3. Categorical data visualizations
        if len(categorical_cols) > 0:
            self._create_categorical_visualizations(categorical_cols, output_dir)
        
        # 4. Correlation analysis
        if len(numeric_cols) > 1:
            self._create_correlation_analysis(numeric_cols, output_dir)
        
        print("✓ All visualizations created successfully!")
        return True
    
    def _create_data_overview(self, output_dir):
        """Create data overview visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Overview Dashboard', fontsize=16, fontweight='bold')
        
        # Missing values heatmap
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            axes[0, 0].bar(range(len(missing_data)), missing_data.values)
            axes[0, 0].set_title('Missing Values by Column')
            axes[0, 0].set_xticks(range(len(missing_data)))
            axes[0, 0].set_xticklabels(missing_data.index, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Count')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Missing Values by Column')
        
        # Data types distribution
        dtype_counts = self.df.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Data Types Distribution')
        
        # Data size visualization
        sizes = ['Rows', 'Columns']
        values = [self.df.shape[0], self.df.shape[1]]
        axes[1, 0].bar(sizes, values, color=['skyblue', 'lightcoral'])
        axes[1, 0].set_title('Dataset Dimensions')
        axes[1, 0].set_ylabel('Count')
        
        # Memory usage
        memory_usage = self.df.memory_usage(deep=True)
        axes[1, 1].barh(range(len(memory_usage)), memory_usage.values)
        axes[1, 1].set_title('Memory Usage by Column')
        axes[1, 1].set_yticks(range(len(memory_usage)))
        axes[1, 1].set_yticklabels(memory_usage.index, fontsize=8)
        axes[1, 1].set_xlabel('Memory (bytes)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/data_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_numeric_visualizations(self, numeric_cols, output_dir):
        """Create visualizations for numeric data."""
        n_cols = len(numeric_cols)
        
        # Distribution plots
        if n_cols > 0:
            fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(15, 5 * ((n_cols + 1) // 2)))
            if n_cols == 1:
                axes = [axes]
            elif n_cols == 2:
                axes = axes.reshape(-1)
            else:
                axes = axes.flatten()
            
            fig.suptitle('Numeric Data Distributions', fontsize=16, fontweight='bold')
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    self.df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(n_cols, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/numeric_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Box plots
        if n_cols > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            self.df[numeric_cols].boxplot(ax=ax)
            ax.set_title('Box Plots of Numeric Variables', fontsize=14, fontweight='bold')
            ax.set_ylabel('Values')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/numeric_boxplots.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_categorical_visualizations(self, categorical_cols, output_dir):
        """Create visualizations for categorical data."""
        for col in categorical_cols:
            if self.df[col].nunique() <= 20:  # Only plot if not too many categories
                plt.figure(figsize=(12, 6))
                value_counts = self.df[col].value_counts()
                
                plt.subplot(1, 2, 1)
                value_counts.plot(kind='bar', color='lightcoral', alpha=0.7)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                
                plt.subplot(1, 2, 2)
                plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                plt.title(f'Proportion of {col}')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/categorical_{col}.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_correlation_analysis(self, numeric_cols, output_dir):
        """Create correlation analysis visualization."""
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, fmt='.2f')
            plt.title('Correlation Matrix of Numeric Variables', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_html_report(self, output_dir="analysis_output"):
        """Generate an HTML report with all analysis results."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Satellite Data Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .timestamp {{ color: #7f8c8d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🛰️ Satellite Data Analysis Report</h1>
                <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>📊 Dataset Overview</h2>
                <div class="metric">
                    <strong>File:</strong> {self.json_file_path}<br>
                    <strong>Rows:</strong> <span class="metric-value">{self.analysis_results['shape'][0]}</span><br>
                    <strong>Columns:</strong> <span class="metric-value">{self.analysis_results['shape'][1]}</span><br>
                    <strong>Memory Usage:</strong> {self.analysis_results['memory_usage'] / 1024:.2f} KB
                </div>
                
                <h2>📈 Data Overview</h2>
                <img src="data_overview.png" alt="Data Overview Dashboard">
                
        """
        
        # Add numeric analysis if available
        if 'numeric_summary' in self.analysis_results:
            html_content += """
                <h2>📊 Numeric Data Analysis</h2>
                <img src="numeric_distributions.png" alt="Numeric Distributions">
                <img src="numeric_boxplots.png" alt="Numeric Box Plots">
            """
            
            if len(self.df.select_dtypes(include=[np.number]).columns) > 1:
                html_content += '<img src="correlation_matrix.png" alt="Correlation Matrix">'
        
        # Add categorical analysis if available
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            html_content += "<h2>📋 Categorical Data Analysis</h2>"
            for col in categorical_cols:
                if self.df[col].nunique() <= 20:
                    html_content += f'<img src="categorical_{col}.png" alt="Distribution of {col}">'
        
        # Add data table
        html_content += f"""
                <h2>📄 Data Sample</h2>
                {self.df.head(10).to_html(classes="data-table", table_id="data-sample")}
                
                <h2>📋 Column Information</h2>
                <table>
                    <tr><th>Column</th><th>Type</th><th>Missing Values</th><th>Missing %</th></tr>
        """
        
        for col, dtype in self.analysis_results['dtypes'].items():
            missing = self.analysis_results['missing_values'][col]
            missing_pct = (missing / self.analysis_results['shape'][0]) * 100
            html_content += f"<tr><td>{col}</td><td>{dtype}</td><td>{missing}</td><td>{missing_pct:.1f}%</td></tr>"
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(f"{output_dir}/analysis_report.html", 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML report generated: {output_dir}/analysis_report.html")
    
    def run_complete_analysis(self, output_dir="analysis_output"):
        """Run the complete analysis pipeline."""
        print("🚀 Starting Satellite Data Analysis...")
        
        # Load data
        if not self.load_json_data():
            return False
        
        # Convert to DataFrame
        if not self.convert_to_dataframe():
            return False
        
        # Perform analysis
        if not self.basic_data_analysis():
            return False
        
        # Print summary
        self.print_data_summary()
        
        # Create visualizations
        if not self.create_visualizations(output_dir):
            return False
        
        # Generate HTML report
        self.generate_html_report(output_dir)
        
        print(f"\n✅ Analysis complete! Check the '{output_dir}' directory for results.")
        return True


def main():
    """Main function to run the satellite data analyzer."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Satellite Data Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sat_data_analyzer.py data.json
  python sat_data_analyzer.py satellite_results.json --output analysis_results
        """
    )
    
    parser.add_argument('json_file', help='Path to the JSON file containing satellite data')
    parser.add_argument('--output', '-o', default='analysis_output', 
                       help='Output directory for analysis results (default: analysis_output)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.json_file):
        print(f"✗ Error: File '{args.json_file}' does not exist")
        sys.exit(1)
    
    # Create analyzer and run analysis
    analyzer = SatelliteDataAnalyzer(args.json_file)
    success = analyzer.run_complete_analysis(args.output)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()