# 🛰️ Satellite Data Review Tool

A comprehensive Python application for analyzing satellite data from JSON files. This tool parses JSON data containing satellite analysis results and provides detailed data analysis with tables, statistics, and visualizations.

## 🚀 Features

- **JSON Data Parsing**: Automatically loads and parses JSON files with satellite data
- **Data Analysis**: Comprehensive statistical analysis including:
  - Basic statistics (mean, std, min, max, quartiles)
  - Missing value analysis
  - Data type analysis
  - Memory usage statistics
- **Interactive Tables**: Well-formatted tables showing data samples and summaries
- **Rich Visualizations**:
  - Data overview dashboard
  - Distribution plots for numeric variables
  - Box plots for outlier detection
  - Correlation matrix heatmaps
  - Categorical data pie charts and bar plots
- **HTML Report Generation**: Professional HTML reports with embedded charts
- **Command Line Interface**: Easy-to-use CLI for batch processing

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/tim7en/sat_data_review.git
cd sat_data_review
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Usage

### Basic Usage

Analyze a JSON file:
```bash
python3 sat_data_analyzer.py your_satellite_data.json
```

Specify output directory:
```bash
python3 sat_data_analyzer.py your_data.json --output my_analysis
```

### Example with Sample Data

The repository includes sample satellite data for testing:
```bash
python3 sat_data_analyzer.py sample_satellite_data.json
```

This will generate:
- Console output with detailed statistics
- `analysis_output/` directory containing:
  - `analysis_report.html` - Interactive HTML report
  - `data_overview.png` - Data overview dashboard
  - `numeric_distributions.png` - Distribution plots
  - `numeric_boxplots.png` - Box plots
  - `correlation_matrix.png` - Correlation analysis
  - `categorical_*.png` - Categorical variable plots

## 📊 JSON Data Format

The tool supports flexible JSON formats:

### Option 1: Data array in JSON
```json
{
  "metadata": {
    "satellite": "Landsat-8",
    "date": "2023-08-15"
  },
  "data": [
    {
      "pixel_id": 1,
      "latitude": 40.7128,
      "longitude": -74.0060,
      "ndvi": 0.67,
      "temperature": 25.3,
      "land_cover": "vegetation"
    }
  ]
}
```

### Option 2: Results array
```json
{
  "results": [
    {"param1": "value1", "param2": 123},
    {"param1": "value2", "param2": 456}
  ]
}
```

### Option 3: Direct array
```json
[
  {"column1": "value1", "column2": 123},
  {"column1": "value2", "column2": 456}
]
```

## 📈 Output Description

### Console Output
- 📊 Comprehensive data analysis report
- 📏 Dataset dimensions and memory usage
- 🗂️ Column information with data types and missing values
- 📈 Numeric data summary statistics
- 📋 Categorical data value counts
- 📄 Data sample preview

### Generated Files
- **HTML Report**: Interactive report with all visualizations
- **PNG Images**: High-quality charts and plots
- **Analysis Dashboard**: Overview of data quality and structure

## 🛠️ Requirements

- Python 3.7+
- pandas >= 1.5.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- plotly >= 5.0.0
- numpy >= 1.21.0
- tabulate >= 0.9.0

## 🎯 Use Cases

- **Satellite Data Analysis**: Analyze NDVI, temperature, and other satellite-derived metrics
- **Environmental Monitoring**: Process land cover classification results
- **Quality Assessment**: Review data quality flags and missing value patterns
- **Research Reporting**: Generate professional reports for scientific publications
- **Data Exploration**: Quick exploratory data analysis of JSON datasets

## 📝 Command Line Options

```
usage: sat_data_analyzer.py [-h] [--output OUTPUT] json_file

Comprehensive Satellite Data Analysis Tool

positional arguments:
  json_file             Path to the JSON file containing satellite data

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Output directory for analysis results (default: analysis_output)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.
