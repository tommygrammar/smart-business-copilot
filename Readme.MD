# Computational Business Analyst

Computational Business Analyst is a high-performance decision intelligence system designed to help businesses understand operational dynamics, forecast future states, and make optimized strategic decisions. This repository provides a suite of computational models and data pipelines for business diagnostics, financial forecasting, and scenario analysis.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Directory Structure](#directory-structure)
* [Models Overview](#models-overview)
* [API Endpoints](#api-endpoints)
* [Contributing](#contributing)
* [License](#license)

## Overview

This repository integrates a broad set of models designed to generate interpretable, data-driven insights. Key functionalities include:

* Sensitivity analysis and simulation of business scenarios.
* Forecasting of cash flow, sales, and growth metrics.
* Competitive landscape analysis and strategy impact evaluation.
* Event probability estimation and product-level diagnostics.

## Features

* **Multi-Domain Inference**: Integrated model suite for operations, finance, and market behavior.
* **Forecasting & Risk Modeling**: Predictive modules for cash flow, demand, and exposure to risk events.
* **Strategic Diagnostics**: Tools for competitor response modeling and resource allocation optimization.
* **Product & KPI Segmentation**: Event probability analysis and performance clustering across products and KPIs.
* **RESTful API Access**: Flask-based endpoints for seamless integration and modular use.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/tommygrammar/smart-business-copilot.git
cd smart-business-copilot
pip install -r requirements.txt
```

## Usage

Start the Flask API server:

```bash
python api.py
```

Use tools such as `curl` or Postman to send requests to model endpoints. Refer to the [API Endpoints](#api-endpoints) section for route details.

## Directory Structure

```bash
smart-business-copilot/
├── Data/
│   └── business_data.py                  # Business-specific data handling scripts
├── Models/
│   ├── business_twin_sensitivity_model.py
│   ├── competitor_analysis.py
│   ├── demand_analysis.py
│   ├── event_probability_models.py
│   ├── gradient_ascent_sensitivity_allocation_model.py
│   ├── growth.py
│   ├── risk_model.py
│   ├── sales_forecast.py
│   └── ...                               # Additional analytics models
├── api.py                                # Flask API to expose model functionality
├── requirements.txt                      # Python dependency list
```

## Models Overview

### 1. Business Twin Sensitivity Model

* **File**: `Models/business_twin_sensitivity_model.py`
* **Purpose**: Simulates business dynamics to evaluate how input changes affect outcomes.
* **Application**: Sensitivity testing for core KPIs under different operational assumptions.

### 2. Competitor Analysis

* **File**: `Models/competitor_analysis.py`
* **Purpose**: Applies Bayesian inference to model and assess competitor strategies and their influence.
* **Application**: Evaluating relative market positioning and expected competitive response.

### 3. Demand Analysis

* **File**: `Models/demand_analysis.py`
* **Purpose**: Forecasts product-level demand over time horizons using historical patterns and external signals.
* **Application**: Planning inventory, marketing, and production in alignment with projected demand.

### 4. Event Probability Models

* **File**: `Models/event_probability_models.py`
* **Purpose**: Quantifies likelihoods of predefined business events using statistical modeling.
* **Application**: Anticipating disruptions or inflection points (e.g., stockouts, customer churn).

### 5. Growth Models

* **File**: `Models/growth.py`
* **Purpose**: Projects growth trajectories, interactions between KPIs, and systemic behavior over time.
* **Application**: Long-term planning, target setting, and monitoring early warning indicators.

### 6. Sales Forecasting

* **File**: `Models/sales_forecast.py`
* **Purpose**: Predicts sales volumes across short- and medium-term windows.
* **Application**: Revenue planning, pricing strategy, and supply chain coordination.

### 7. Risk Analysis

* **File**: `Models/risk_model.py`
* **Purpose**: Estimates probability and impact of business risks using scenario-based methods.
* **Application**: Operational continuity planning and mitigation strategy design.

*(Additional models can be found in the Models directory.)*

## API Endpoints

Each model is exposed as a RESTful endpoint via the Flask API. Example routes:

| Endpoint            | Method | Description                                                  |
| ------------------- | ------ | ------------------------------------------------------------ |
| `/twin`             | POST   | Run Business Twin sensitivity simulation.                    |
| `/cashflowsales`    | POST   | Analyze weekly cash flow and sales activity.                 |
| `/competitorimpact` | POST   | Assess strategic influence of competitors.                   |
| `/demand`           | POST   | Forecast product-level demand.                               |
| `/eventprobability` | POST   | Compute likelihoods of business-critical events.             |
| `/optimize`         | POST   | Perform resource allocation via gradient-based optimization. |
| `/growth`           | POST   | Simulate growth paths and KPI interaction.                   |
| `/salesforecast`    | POST   | Generate forward-looking sales forecasts.                    |

Detailed implementation is provided in `api.py`.

## Contributing

To contribute, follow the standard development workflow:

1. Fork this repository.
2. Create a new feature branch:

```bash
git checkout -b my-feature-branch
```

3. Make your changes and commit:

```bash
git commit -am "Add feature or fix"
```

4. Push to your fork and submit a pull request:

```bash
git push origin my-feature-branch
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for full terms.