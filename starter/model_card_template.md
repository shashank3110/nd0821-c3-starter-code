# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
problem type: binary classification
model: Gradent boosting classifier

## Intended Use
To predict if an adult earns >=50K per year.

## Dataset
Census data: https://archive.ics.uci.edu/ml/datasets/census+income
Dataset shape: (32561, 15)
label: "salary"

## Training Data
Train split : 80%

## Evaluation Data
Test split : 20%

## Metrics
- Precision: 0.759
- Recall: 0.648
- Fbeta score (beta=1 i.e. F1 score): 0.699

## Ethical Considerations
The dataset involves sensitive features such as race, sex, v to name a few.
The model could infer certain kinds of people more favorably. One needs
to be cognizant of this.

## Caveats and Recommendations
To avoid any bias on a certain slice of data.
One can further evaluate the model on data slices on features such as:
- sex
- race
- native-country


