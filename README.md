# Climate-Change-Risk-and-Opportunity-Analysis
A project to analyze and predict carbon emissions across organizations and sectors to enable targeted climate interventions.

# EmissionsImpact-Analyzer

## Overview
The EmissionsImpact-Analyzer is a machine learning project designed to assess carbon emissions across various organizations and sectors, enabling targeted interventions for climate change mitigation. This repository includes scripts for preprocessing data, selecting influential features, and training and evaluating models to predict emissions.

## Authors
- Anuj Patil
- Srideepthi Vootla
- Jayakishan Minnekanti
- Srihaarika Viswanadhapalli

## Objective
To analyze and predict the carbon emissions of different organizations and sectors using machine learning techniques, providing a basis for informed climate change interventions.

## Methodology
### Data Preprocessing
- **Categorical Encoding**: Transformed categorical variables into numerical formats using label encoding.
- **Normalization**: Applied MinMax scaling to ensure equal feature weighting.

### Feature Selection
- **Lasso Regression**: Employed to retain the most influential features for predicting emissions.

### Model Selection & Training
- **Random Forest Regressor**: Utilized for its robustness and suitability for regression tasks.
- **Hyperparameter Tuning**: Conducted via RandomizedSearchCV to find the optimal model parameters.

### Validation
- **Data Split**: Employed an 80-20 split for training and testing to ensure model performance on unseen data.

## Key Findings
- **Model Performance**: Achieved a high R-squared and low error metrics on both training and testing sets.
- **Top Emissions Contributors**: Identified the top companies and sectors with the highest predicted emissions.

## Results Visualization
- Visual representations of the top contributors to emissions, both at the company and sector levels, using bar charts.

## Usage
To run the code in this repository, you will need Python 3 and the following packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run
1. Clone this repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the script `emissions_analysis.py` to perform the analysis.

## Conclusion
The EmissionsImpact-Analyzer provides valuable insights into the carbon emission profiles of major organizations and sectors, aiding stakeholders in making data-driven decisions for climate change mitigation efforts. Continuous refinement and expansion of the model's capabilities are recommended for enhanced accuracy.

## Future Work
- Explore sector-specific models.
- Incorporate additional data sources such as global temperature and economic indicators.
- Conduct a deeper analysis of prediction anomalies to refine the model.
