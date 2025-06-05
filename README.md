# Human Activity Recognition using UCI HAR Dataset  
**Classification and Regression with Feature Reduction and Preprocessing**

---

## Overview

This project aims to classify and predict human activity using wearable sensor data from the UCI HAR dataset.  
We conducted feature reduction, preprocessing, model training, evaluation, and visualization using multiple machine learning algorithms.

---

## Dataset Information

- **Total Samples**: 10,298  
- **Used Samples**: 7,552 (after cleaning)  
- **Features**: 563  
- **Label Classes**: 6  
  (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`)

| Column     | Description                       |
|------------|-----------------------------------|
| subject    | Participant ID (1–30)             |
| activity   | Activity label (1–6)              |
| 561 others | Sensor measurements (float64)     |

---

## Pipeline Summary

1. **Feature Reduction**  
   Manually selected 28 key features based on domain knowledge.

2. **Data Preprocessing**  
   - Remove duplicates and missing values  
   - Normalize using `StandardScaler`  
   - Encode labels using `LabelEncoder` or `OneHotEncoder`

3. **Model Training**  
   - Classification: MLP, Decision Tree, Naive Bayes, SVM, KNN  
   - Regression: Random Forest, MLP, KNN, Ridge, Lasso

4. **Evaluation**  
   - Classification: Stratified K-Fold  
   - Regression: Accuracy, MSE, Cosine Similarity

---

## Feature Reduction

| Sensor         | Stats                     | Axis   | Count | Description            |
|----------------|---------------------------|--------|-------|------------------------|
| tBodyAcc       | mean, std, energy, entropy| X/Y/Z  | 12    | Motion features        |
| tBodyGyro      | mean, entropy             | X/Y/Z  | 6     | Rotational features    |
| tGravityAcc    | mean, std                 | X/Y/Z  | 6     | Posture/balance        |
| Magnitude      | mean, std                 | -      | 4     | Signal magnitude       |
| Misc           | sma                       | -      | 1     | Signal strength summary|

---

## Classification Models

### Objective

To classify human activities into 6 categories using various machine learning models.

### Cross-Validation Results (5-Fold)

| Model                     | Accuracy | Std Dev | Notes                     |
|--------------------------|----------|---------|---------------------------|
| MLP (Neural Network)     | 0.9431   | 0.0059  | Best performance overall  |
| KNN (k=1)                | 0.8992   | 0.0059  | Distance-based method     |
| Decision Tree (depth=20) | 0.8983   | 0.0095  | Balanced complexity       |
| SVM (poly kernel)        | 0.8968   | 0.0081  | Non-linear kernel         |
| Gaussian Naive Bayes     | 0.7783   | 0.0130  | Simple baseline           |
| Decision Tree (depth=2)  | 0.5392   | 0.0003  | Underfitting              |

> MLP issued a convergence warning (max_iter=300). Consider increasing to 500 or adjusting learning rate.

### Test Set Evaluation (MLP)

| Class              | Precision | Recall | F1-score |
|--------------------|-----------|--------|----------|
| WALKING            | 0.975     | 0.924  | 0.949    |
| WALKING_UPSTAIRS   | 0.923     | 0.956  | 0.939    |
| WALKING_DOWNSTAIRS | 0.963     | 0.986  | 0.974    |
| SITTING            | 0.897     | 0.904  | 0.901    |
| STANDING           | 0.908     | 0.905  | 0.907    |
| LAYING             | 1.000     | 0.997  | 0.998    |
| **Overall Accuracy** | **0.944** |        |          |

---

## Regression Models

### Objective

To predict one-hot activity vectors using regression models and assess generalization performance.

### Evaluation Metrics

- **Accuracy** (based on `argmax` of predicted vector)  
- **Mean Squared Error (MSE)**  
- **Cosine Similarity** (vector direction alignment)

### Model Comparison

| Model               | Accuracy | MSE    | Cosine Similarity |
|---------------------|----------|--------|-------------------|
| Random Forest       | 0.9373   | 0.0157 | 0.9489            |
| MLP (100-50)        | 0.9314   | 0.0193 | 0.9405            |
| MLP (100)           | 0.9234   | 0.0268 | 0.9177            |
| MLP (50)            | 0.9030   | 0.0327 | 0.9006            |
| KNN (k=3)           | 0.8957   | 0.0267 | 0.9119            |
| Ridge (alpha=1.0)   | 0.8660   | 0.0526 | 0.8342            |
| Lasso (alpha=0.1)   | 0.6792   | 0.0923 | 0.6845            |

### Per-Class Report (Random Forest)

| Class              | Precision | Recall | F1-score |
|--------------------|-----------|--------|----------|
| WALKING            | 0.9215    | 0.9331 | 0.9272   |
| WALKING_UPSTAIRS   | 0.8991    | 0.9276 | 0.9131   |
| WALKING_DOWNSTAIRS | 0.9691    | 0.9216 | 0.9447   |
| SITTING            | 0.8977    | 0.9222 | 0.9098   |
| STANDING           | 0.9301    | 0.9079 | 0.9188   |
| LAYING             | 1.0000    | 1.0000 | 1.0000   |
| **Overall Accuracy** | **0.9373** |        |          |

---

## Highlights & Takeaways

| Aspect                | Description |
|-----------------------|-------------|
| Top Classifier        | MLP with 94.3% accuracy using 5-fold CV |
| Top Regressor         | Random Forest with 93.7% accuracy and highest cosine similarity |
| Feature Selection     | 28 meaningful features selected based on sensor type and signal stats |
| Improvement Suggestions | Increase MLP `max_iter` or adjust learning rate |
| Interpretability      | Regression approach allows fine-grained vector analysis beyond classification |

---

## Repository Structure


project-root/
├── data/ # Raw dataset files (excluded from repo)
├── preprocessing/ # Scripts for cleaning, encoding, and scaling data
├── classification/ # Scripts for training and evaluating classification models
├── regression/ # Scripts for regression-based activity prediction
├── reports/ # Final plots, evaluation metrics, and model comparison outputs
└── README.md # Project overview and documentation

yaml
Copy
Edit


---

## References

- UCI HAR Dataset: [https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)  
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
