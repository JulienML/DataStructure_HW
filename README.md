# Big Data Structure - Homework

This application provides interactive interfaces for analyzing database schemas, query costs, and sharding distributions.

## Installation

Install the required dependencies:

```bash
pip install streamlit
```

## Running the Application

### Option 1: Using Streamlit

To start the Streamlit application, run:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

### Option 2: Using the `test.py` Script

You can also run the `test.py` script to execute predefined tests:

```bash
python test.py
```

The results will be printed to the console.
You can modify the `test.py` file to test different queries and database configurations.

## Features

### Homework 1
- **Database Size Computation**: Calculate estimated sizes for different database schemas
- **Sharding Distribution**: Analyze data distribution across servers with different sharding strategies

### Homework 2
- **Filter Query Costs**: Compute execution costs for filter queries including time, carbon footprint, and price
- **Join Query Costs**: Analyze costs for join operations between two collections