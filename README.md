# State repression prediction
A poster project for the ETH Methods IV course. This analysis uses statistical‐learning techniques to predict whether a state will respond to anti‐government protests with repressive methods.


## Motivation
From a policy standpoint, a predictor of state repression can help international organizations and human‐rights activists spot early warning signs of punitive countermeasures. That, in turn, would allow them to deploy proactive measures—diplomatic pressure, media campaigns, or humanitarian support—before violence escalates.

From a research standpoint, existing studies on government protest responses are usually case‐by‐case. Previous work often overlooks cross‐country variation or “softer” repressive tactics (e.g., orchestrated arrests, crowd dispersal without fatalities). A predictive model could (a) test competing theoretical explanations, and (b) produce quantitative forecasts that direct scholars toward especially high‐risk scenarios.



## Data
### Mass mobilization protest data
**Source**
Clark, D., & Regan, P. (2016). Mass mobilization protest data (V5). Harvard Dataverse. https://doi.org/10.7910/DVN/HTTWYL

**Description:**  
Covers anti‐government protests (50+ participants) from 1990–2020 in 162 countries. Includes:
- Protest start/end dates  
- Location (country)  
- Number of participants (mixed numeric/character entries)  
- A binary indicator of violence against the state  
- Seven‐point categorical variables for protester demands (e.g., labor disputes, police brutality, political corruption)  
- Seven‐point categorical variables for state responses (accommodation, arrests, beatings, crowd dispersal, ignore, killings, shootings)  
Multiple state responses can be recorded per event.

**Preprocessing**
- Dropped observations with no protest data.
- Created a binary variable `harsh`:
  - 1 if any repressive response (arrests, beatings, crowd dispersal, killings, or shootings) was recorded.
  - 0 if only non‐repressive responses (accommodation or “ignore”) occurred.
- Standardized the “protest count” column (which mixed numeric and character) into categorical bins (e.g., “51–100,” “101–500,” “>500”).
- Manually dummy‐coded:
  - Group‐size categories  
  - Protest start month (Jan – Dec)  
  - Protester demands (labour, corruption, police brutality, etc.)
- Created a `multi_day` flag (1 if protest lasted > 1 day, else 0).
- Merged with V-Dem data (see below) and then:
  - Removed intermediate columns no longer needed  
  - Dropped ~2,000 observations with missing values (acceptable because the remaining 13,000+ observations still form a balanced sample).


### Varieties of Democracy
**Source**
Coppedge, M., Gerring, J., Knutsen, C. H., Lindberg, S. I., Teorell, J., Altman, D., ... & Ziblatt, D. (2024b). V-Dem [Country-Year/Country-Date] Dataset v14. Varieties of Democracy (V-Dem) Project. https://doi.org/10.23696/mcwt-fr58

**Description**
Country‐level data from 1789 to present. Contains hundreds of indicators on governance, democracy, civil liberties, media freedom, corruption, and more.

**Preprocessing**
Data limited to 56 variables which I deemed theoretically relevant, including democracy indices and indicators related to equality, corruption, accountability, media freedom, and political liberties. 

### Final Merged Dataset
- **Observations:** 13,082  
- **Countries:** 152  
- **Features:** 82 (56 from V-Dem + 26 from protest data)  
- **Class Balance:**  
  - Repressive (1): 5,245 (40%)  
  - Non‐repressive (0): 7,837 (60%)

## Methods and Results 

### Data Split & Resampling
- **Hold‐out split:**  
  - Training: 1990–2015 (80% of observations)  
  - Testing: 2016–2020 (20% of observations)
- **Rolling origin resampling** on the 1990–2015 block:
  1. **Fold 1**: Train 1990–2007 (60%), Assess 2008–2009 (15%), Skip 2010 (7.5%), Test 2011–2012 (17.5%)  
  2. **Fold 2**: Train 1990–2009, Assess 2010–2011, Skip 2012, Test 2013–2014  
  3. **Fold 3**: Train 1990–2011, Assess 2012–2013, Skip 2014, Test 2015 
  4. **Fold 4**: Train 1990–2013, Assess 2014–2014, Skip 2015, Test 2016  
- This simulates real‐world forecasting on time‐ordered events. Final evaluation occurs on 2016–2020.

### Logistic Regression (Baseline)
- **Features:** 17 theoretically selected predictors (no interactions)  
- **Training:** Fit once on 1990–2015; test on 2016–2020.  
- **Metrics (Test):**  
  - ROC AUC: 0.608  
  - Precision (“repression”): NA (no positive predictions)  
  - Recall (“repression”): 0.000  
  - Accuracy: 0.600 (predicts only non‐repressive)
- **Conclusion:** Model cannot discriminate. It always predicts “no repression,” so recall = 0.

### Lasso
- **Why Random Forest?**  
  - Automatic feature selection.
- **Features:** All 82 predictors (Lasso will zero‐out uninformative ones).  
- **Standardization:** All predictors standardized (prevent data leakage).  
- **Tuning:**  
  - Rolling origin CV on training set with 50 λ values.  
  - Selected λ = 0.00355648 (maximized ROC AUC).  
- **Top Predictors (nonzero coefficients):**  
  1. `violent_protest_flag` (+1.55 log‐odds)  
  2. `v2x_exlbrbr` (executive bribery, +0.82)  
  3. `v2x_frptprm_men` (freedom of discussion for men, –0.67)  
  4. `v2x_polyarchy` (polyarchy index, –0.44)  
  5. `v2x_cltcensor` (media censorship, –0.35)  
  (Total of 30 nonzero predictors.)
- **Metrics (Test):**  
  - ROC AUC: 0.830  
  - Precision (“repression”): 0.760  
  - Recall (“repression”): 0.590  
- **Interpretation:**  
  - Violent protests strongly increase predicted repression risk (but beware of reverse causality and issue of simultaneousness of          state and protester violence).  
  - Institutional factors (bribery, polarization) ↑ repression; civil‐liberties indicators ↓ repression.  
  - Precision (76%) is decent, but recall (59%) means ~41% of truly repressed events are missed.

### Random Forest
- **Why Random Forest?**  
  - Robust to overfitting, can handle nonlinearities and interactions.  
- **Tuning Grid:**  
  - `mtry` (predictors per split): 10 – 40  
  - `ntree` (number of trees): 800 – 2000  
  - `nodesize`: 1 – 8  
- **Procedure:**  
  1. Preliminary run of 20 models to narrow hyperparameter ranges.  
  2. Final grid search of 50 models (via rolling origin CV).  
  3. Best-fold parameters:  
     - `mtry` = 16  
     - `ntree` = 1600  
     - `nodesize` = 8  
- **Metrics (Test):**  
  - ROC AUC: 0.806  
  - Precision (“repression”): 0.744  
  - Recall (“repression”): 0.618  
- **Comparison to Lasso:**  
  - Slightly higher recall (+2.8%) but lower precision (–1.6%) and AUC (–0.024).  
  - Given the similar overall performance and higher computational cost, Lasso remains the preferred model.


## Usage
### 1. Clone the Repository

```bash
git clone https://github.com/Dem-guy/State-repression-prediction.git
cd State-repression-prediction
```

### 2. Decompress V-dem Data 
```bash
unzip data/V-Dem-CY-Full+Others-v14.csv.zip -d data/
```

### 3. Install R Packages

In R
```r
install.packages(c(
  "tidyverse",
  "car",
  "readxl",
  "broom",       
  "tidymodels",
  "gt",
  "install_phantomjs"
))

```
### 4. Render the R Markdown

In R
```r
rmarkdown::render("State_repression_prediction.Rmd")
```
Note: the Random Forest tuning step can take several hours, depending on your machine.

## License: All rights reserved


