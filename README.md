# State repression prediction
Poster Project for ETH course Methods IV. The goal of this project is to predict whether a state will respond to anti-government protests with repressive methods or not. using statistical learning techniques.


## Motivation
From a policy standpoint, such a predictor could help international organisations and human rights activists to identify early warning signs of governments choosing repressive countermeasures, allowing for the deployment of proactive measures to protect civil liberties and mitigate harm. It could also support policymakers in other countries to mediate or apply international pressure to prevent conflict escalation and promote peaceful resolutions. 

Form a research standpoint, this goal holds value too. There has been much research on conducted on how governments respond to protests, often as detailed case studies. Pevious work often fails to explore more nuanced diversities between states, or that modern autocracies were often capable of using “softer measures”. This predictive model could therefore also refine existing theoretical frameworks by testing a variety of explanations for state repressive responses. Additionally, it can provide more precise forecasts that help researchers focus on high-risk scenarios, advancing both theory and empirical study.


## Data

### Mass mobilization protest data
**Source**
Clark, D., & Regan, P. (2016). Mass mobilization protest data (V5). Harvard Dataverse. https://doi.org/10.7910/DVN/HTTWYL

**Description**
Covers anti-government protests (50+ participants) from 1990 to 2020 in 162 countries. The data, compiled from news reports, includes protest start and end dates, location, participant numbers, a binary indicator of violence against the state, and seven-point categorical variables for protester demands and state responses. Protester demands include issues like labour disputes, police brutality, and political corruption. Possible state responses are categorized as accommodation, arrests, beatings, crowd dispersal, ignore, killings, and shootings. Multiple state responses can be recorded for each protest event.

**Preprocessing**
I excluded observations with no protests and created a binary variable for state response, where 1 indicates repressive responses (arrests, beatings, crowd dispersal, killings, and shootings) and 0 indicates repressive responses (accommodation and ignore). As a single protest could spark multiple state responses (e.g., harsh measures followed by accommodation), this indicator would take the value of 1 if any repressive response occurred.

Protest count data, which mixed numeric and character entries, was standardized using a categorical variable indicating group size. I manually created dummy variables for group size, the protest start month, and protester demands. I also added a binary variable for if protests lasting more than one day. After merging the datasets, I removed no longer needed columns and dropped observations with missing values. While this latter method of handling missing data is indeed often suboptimal, and did result in the loss of approximately 2,000 observations from the original 15,000, I deemed it an acceptable trade for simplicity as the dataset was still quite large and relatively well balanced on the to be predicted class. 

### Varieties of Democracy

**Source**
Coppedge, M., Gerring, J., Knutsen, C. H., Lindberg, S. I., Teorell, J., Altman, D., ... & Ziblatt, D. (2024b). V-Dem [Country-Year/Country-Date] Dataset v14. Varieties of Democracy (V-Dem) Project. https://doi.org/10.23696/mcwt-fr58

**Description**
Contry level data which provides comprehensive data on democracy and governance across nearly all countries from 1789 to the present. It includes several hundred indicators, coded by country experts.

**Preprocessing**
Data limited to 56 variables which I deemed theoretically relevant, including democracy indices and indicators related to equality, corruption, accountability, media freedom, and political liberties.

### Final Merged Dataset
The final merged dataset contained 13,082 observations over 152 countries and 82 features, with 7,837 non-repressive and 5,245 repressive state responses.

## Methods and Results 

### Data Split and Resampling Method
The first step in the process was to split the dataset into training and testing sets for predictive modelling. Since the data is longitudinal, I split it by year, with the training set covering 1990–2015 and the testing set covering 2016–2020. The 2016 cutoff was chosen as it represents the point where roughly 80% of the observations occur, creating an 80/20 split.

I used rolling origin resampling, simulating real-world forecasting on time-ordered events. I defined five temporal windows: the first 60% of the training data as the initial training window, followed by a 15% assessment window and a 7.5% skip. This process created four training folds, with each fold growing to include all previous data, and testing always occurring on the subsequent block.

### Logistical Regression
I begin with a logistic‐regression baseline for state response. Rather than include all 82 predictors and risk overfitting, I use theoretical expectations to select 17 features (no interactions). I fit this model once (i.e., without resampling) on the training data and then apply it to the held‐out test set. The results show it’s is ineffective as a classifier. Although the ROC AUC of 0.608 appears to suggest the model performs slightly better than random guesswork, the model’s recall is 0: it never correctly identifies any instances of state repression, instead predicting only the majority “no-repression” class. I tried alternative feature subsets and specifications, but none improved its ability to discriminate. In short, this logistic model explains no meaningful variation in state responses.

### Lasso
Given the poor performance of the simple logistic model, I applied the Lasso method next. Lasso’s ability to effectively perform variable selection is a great boon here, as my last model was clearly not specified well. To select the optimal penalty value (lambda), I used rolling origin resampling to generate a grid of 50 candidate values. To prevent data leakage, I standardized the predictors beforehand. Since the lasso method automatically removes uninformative predictors, I included all 82 variables in the model.

The candidate value that maximized the ROC AUC was selected. I rely on ROC AUC because I care most about the model’s ability to rank instances of “repression” vs. “no repression”, and because it is a sold method to compare models. Most models performed similarly on the training data, with an average ROC AUC of around 0.81. The best model, with a ROC AUC of 0.814, selected a lambda of 0.00355648 and retained 30 predictors.

Violent protests are by far the strongest predictive feature in the Lasso model, significantly increasing the likelihood of a repressive government response. However, this should not be interpreted as causation; it could be that the state chooses repression first, and protesters then react with violence. This feature is also problematic as a predictor for futrure data, as protest and state violence are likely to occur simultaneously. The other top predictors are institutional in nature: executive bribery and political polarization appear to increase repression, while respect for the constitution and freedom of discussion (for men) decrease the likelihood of repression. It is interesting that neither specific democracy-autocracy measures or protest motivation are among the top five, given their presence in the literature.

When testing the model on the hold-out set, performance improved significantly compared to the logistic model, as it can now differentiate between cases. With a ROC AUC of 0.830, it does a relatively good job at correctly categorizing classes. Given how misclassing an outcome could have negative consequences both for false positives (people may not protest if falsely told they will be at risk) and false negatives (people could have their liberties infringed upon if not properly warned), I check for both precision and recall. With a precision of 0.760, 76% of the positively predicted cases (where the state used repression) are correct, indicating relatively high precision. However, the model's recall of 0.590 suggests it misses about 41% of positive events, leading to many false negatives. The model is far from ideal, but it represent a notable improvement over the simple the logistical model or random guessing.

### Random Forest
Next, I chose to apply a random forest method. I choose it over other tree based methods due to its robustness against overfitting, meaning including many predictors non-problematic. It can also handel non-linear relationships, which my previous models could not. I performed rolling origin resampling to tune the random forest model over multiple folds and adjusted the hyperparameters. Rather than using point estimates, I tuned the hyperparameters over a range to better capture the optimal values while limiting the number of generated models to reduce computational costs. I first ran a preliminary test with 20 models and adjusted the hyperparameter ranges before conducting the final tuning, which involved 50 models. The chosen hyperparameter ranges were: 10-40 predictors randomly sampled at each split, 800-2000 trees, 1-8 minimum amount of datapoints in a node required for a split. The best performing random forest model by ROC AUC had similar scores to the lasso model, around 0.81. The highest scoring model (0.816) tuned to 16 predictors per split, 1600 trees, and a minimum of 8 data points per split. I selected this model for evaluation on the hold-out data.

On the hold-out data, the random forest model performed worse than the lasso in its ROC AUC, scoring a value of 0.806. It achieved a precision of 0.744, about 1.6% lower than the lasso model’s precision. However, it had a slightly higher recall (0.618), meaning it misclassified around 38% of positive cases. Overall, the lasso and random forest models performed similarly, with the lasso having slightly higher precision and lower recall. The similar performance suggests that non-linearities and high-order interactions provide minimal additional discrimination. Given the higher computational cost of the random forest, the lasso model appears to be the optimal choice to predict state response to violence.


## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Dem-guy/State-repression-prediction.git
cd State-repression-prediction
```

### 2. Decompress V-dem Data (Data/V-Dem-CY-Full+Others-v14)

### 3. Install R Packages
```r
install.packages(c(
  "tidyverse",
  "car",
  "readxl",
  "broom",       
  "tidymodels",
  "gt"
))
```




## License: All rights reserved


