# UFC Submission Predictor

Predicting whether a UFC fight ends by submission using fighter statistics.

## Dataset
UFC fight data from Kaggle — raw_total_fight_data.csv + raw_fighter_details.csv

## Model
RandomForestClassifier

## Tools
- pandas, numpy, scikit-learn
- StandardScaler, classification_report, train_test_split

## Key concepts covered
- Merging two datasets with suffixes (_R, _B)
- Data cleaning: height conversion, weight conversion, handling missing values
- Target encoding with np.where
- Class imbalance handling with class_weight='balanced'
- StandardScaler for feature normalization

## Results
Accuracy: 78% | F1-score (submission): 0.16
