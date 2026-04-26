# Handling Imbalanced Datasets in Machine Learning

A complete guide to building customer churn prediction models with imbalanced data using neural networks and multiple resampling techniques.

---

## Project Overview

This project demonstrates how to tackle one of the most common challenges in machine learning: **class imbalance**. Using a real-world customer churn dataset, we build an
Artificial Neural Network and test four different techniques to improve performance on the minority class (customers who churn).

**You'll learn:**
- Why imbalanced data breaks your models
- How to clean messy real-world data
- Four practical methods to handle imbalance
- How to evaluate models beyond accuracy
- Which techniques work best for different scenarios

---

## The Dataset

**Source:** Kaggle Telco Customer Churn Dataset

The dataset contains customer information from a telecommunications company, including demographics, account details, services subscribed to,
billing information, and whether they churned (left the company).

**Key Features:**
- Demographics: gender, senior citizen status, partner, dependents
- Services: phone service, internet type, online security, streaming services
- Account: contract type, payment method, paperless billing
- Charges: monthly charges, total charges
- **Target:** Churn (Yes/No)

**Size:** 7,043 customers with 21 features

---

## The Imbalance Problem

After loading the data, the class distribution revealed a major problem:

**Class Distribution:**
- Not Churned (0): **5,163 samples** (73%)
- Churned (1): **1,869 samples** (27%)

### Why This Matters

When a model sees mostly non-churners during training, it learns to predict "not churn" for almost everyone. This gives high accuracy (since most customers don't churn),
but completely fails to identify the customers who are actually leaving—which is the whole point of the model.

In business terms: you can't retain customers you don't know are leaving. Missing churners means lost revenue.

---

## Data Issues Found

Real-world data is messy. Here's what was broken in this dataset:

### Issue 1: TotalCharges as Text
The `TotalCharges` column was imported as text (object type) instead of numbers because some entries contained whitespace. This would crash any model trying to use it.

**Fix:** Removed rows with whitespace and converted the column to numeric type.

### Issue 2: Inconsistent Categories
Categorical features had values like "No internet service" and "No phone service" that should just be "No" for consistency. 
Having separate categories creates unnecessary complexity during encoding.

**Fix:** Replaced "No internet service" and "No phone service" with "No" across all relevant columns.

### Issue 3: Useless ID Column
The `customerID` column is a unique identifier that has no predictive value—it just adds noise.

**Fix:** Dropped the column entirely.

### Issue 4: Categorical Encoding
Machine learning models need numbers, not text. All categorical features needed to be converted to numeric format.

**Fix:** Applied one-hot encoding to all categorical columns (gender, contract type, payment method, internet service, etc.) 
with `drop_first=True` to avoid multicollinearity.

---

## Building the Baseline Model

After cleaning and encoding, the data was split 80/20 into training and test sets. Continuous features (monthly charges, total charges) 
were normalized to improve neural network training.

**Model Architecture:**
- Input layer: 23 features
- Hidden layer 1: 26 neurons with ReLU activation + 50% dropout
- Hidden layer 2: 15 neurons with ReLU activation + 50% dropout
- Output layer: 1 neuron with sigmoid activation (binary classification)

**Training:** 20 epochs with Adam optimizer and binary crossentropy loss

### Baseline Results (Imbalanced Data)

**Metrics:**
- Loss: 0.444
- Accuracy: **78.3%**

**Classification Report:**
```
          precision    recall  f1-score   support
       0       0.81      0.93      0.86      1033
       1       0.66      0.39      0.49       374

accuracy                           0.78      1407
```

### The Problem is Clear

The model has **78% accuracy**, which sounds good until you look deeper:
- Non-churners (class 0): 0.86 F1-score, 93% recall 
- Churners (class 1): 0.49 F1-score, **39% recall** 

The model is missing **61% of churners**. This is unacceptable for a churn prediction system.

---

## Method 1: Undersampling the Majority Class

### How It Works

Take a random subset of the majority class (non-churners) equal to the size of the minority class (churners). This balances the dataset but 
throws away data.

**Before Undersampling:**
- Class 0: 5,163 samples
- Class 1: 1,869 samples

**After Undersampling:**
- Class 0: 1,869 samples (randomly selected)
- Class 1: 1,869 samples (all kept)
- **Total:** 3,738 samples

**When to use:** When you have a massive dataset and can afford to discard majority samples without losing important patterns.

### Results

The model was retrained for 200 epochs on the undersampled data.

**Metrics:**
- Loss: 0.538
- Accuracy: **74.5%**

**Classification Report:**
```
          precision    recall  f1-score   support
       0       0.77      0.69      0.73       374
       1       0.72      0.80      0.76       374

accuracy                           0.74       748
```

### Analysis

- Accuracy dropped from 78% to 74%, but that's okay
- Churner recall jumped from **39% to 80%** 
- Both classes now have similar F1-scores (0.73 vs 0.76)
- The model is much more balanced and useful for business decisions

**Trade-off:** You lose information from the discarded majority samples, but you gain a model that actually catches churners.

---

## Method 2: Oversampling the Minority Class

### How It Works

Duplicate minority class samples randomly (with replacement) until it matches the majority class size. No data is lost, but you're repeating 
existing examples.

**Before Oversampling:**
- Class 0: 5,163 samples
- Class 1: 1,869 samples

**After Oversampling:**
- Class 0: 5,163 samples (all kept)
- Class 1: 5,163 samples (duplicated with replacement)
- **Total:** 10,326 samples

**When to use:** When you have limited data and can't afford to discard any samples. Watch for overfitting since you're duplicating minority examples.

### Results

The model was retrained for 100 epochs on the oversampled data.

**Metrics:**
- Loss: 0.543
- Accuracy: **75.6%**

**Classification Report:**
```
          precision    recall  f1-score   support
       0       0.84      0.63      0.72      1033
       1       0.70      0.88      0.78      1033

accuracy                           0.76      2066
```

### Analysis

- Churner recall: **88%** (best so far!)   -Churner F1-score: 0.78 (significantly improved from baseline's 0.49)
- Non-churner precision: 0.84 (still strong)
- The model finds almost 9 out of 10 churners

**Trade-off:** Risk of overfitting because the model sees the same minority samples multiple times, but in practice this often works well.

---

## Method 3: SMOTE (Synthetic Minority Over-sampling Technique)

### How It Works

Instead of just duplicating existing minority samples, SMOTE **generates synthetic new samples** using k-nearest neighbors. Here's the process:

1. For each minority class sample, find its k-nearest neighbors (default k=5)
2. Randomly select one of those neighbors
3. Create a new synthetic sample somewhere along the line connecting the original sample and the neighbor
4. Repeat until classes are balanced

This creates new, plausible examples rather than exact duplicates, potentially reducing overfitting.

**Before SMOTE:**
- Class 0: 5,163 samples
- Class 1: 1,869 samples

**After SMOTE:**
- Class 0: 5,163 samples (original)
- Class 1: 5,163 samples (original + synthetic)
- **Total:** 10,326 samples

**When to use:** When you want the benefits of oversampling but with more diversity in the synthetic samples. Requires careful feature scaling and tuning.

### Results

The model was retrained for 100 epochs on the SMOTE-augmented data.

**Metrics:**
- Loss: 0.549
- Accuracy: **75.4%**

**Classification Report:**
```
          precision    recall  f1-score   support
       0       0.83      0.64      0.72      1033
       1       0.70      0.87      0.77      1033

accuracy                           0.75      2066
```

### Analysis

- Churner recall: **87%** (nearly as good as oversampling)
- Churner F1-score: 0.77
- Very similar performance to simple oversampling
- The synthetic samples helped without introducing overfitting

**Trade-off:** More complex than simple oversampling and requires tuning (k neighbors, sampling strategy), but often produces slightly 
better generalization.

---

## Method 4: Ensemble with Majority Voting

### How It Works

Train multiple models on different undersampled batches of the majority class, then combine their predictions using majority voting.
Each model sees all minority samples but different majority samples.

**Process:**
1. Split majority class into 3 batches
2. Train Model 1 on batch 1 + all minority samples
3. Train Model 2 on batch 2 + all minority samples  
4. Train Model 3 on batch 3 + all minority samples
5. For each test sample, take the majority vote from all 3 models

**When to use:** When you want to use all your majority class data but still maintain balance during individual model training. 
Computationally expensive but can be robust.

### Results

Three models were trained for 100 epochs each, then predictions were combined via majority voting.

**Classification Report:**
```
          precision    recall  f1-score   support
       0       0.73      0.84      0.78      1033
       1       0.24      0.14      0.18       374

accuracy                           0.65      1407
```

### Analysis

- Churner recall: **14%** (worse than baseline!) 
- Churner F1-score: 0.18 (terrible)
- Accuracy: 65% (lowest of all methods)
- **The ensemble failed for this dataset**

**Why it failed:** The individual models may not have converged properly, or the voting strategy wasn't optimal for this problem.
This demonstrates that not every technique works for every dataset—experimentation is essential.

---

## Results Comparison

Here's how all methods stack up:

| Method | Accuracy | Class 0 Precision | Class 0 Recall | Class 0 F1 | Class 1 Precision | Class 1 Recall | Class 1 F1 |
|--------|----------|-------------------|----------------|------------|-------------------|----------------|------------|
| **Baseline (Imbalanced)** | 78.3% | 0.81 | 0.93 | 0.86 | 0.66 | 0.39 | 0.49 |
| **Undersampling** | 74.5% | 0.77 | 0.69 | 0.73 | 0.72 | 0.80 | 0.76 |
| **Oversampling** | 75.6% | 0.84 | 0.63 | 0.72 | 0.70 | 0.88 | 0.78 |
| **SMOTE** | 75.4% | 0.83 | 0.64 | 0.72 | 0.70 | 0.87 | 0.77 |
| **Ensemble** | 65.0% | 0.73 | 0.84 | 0.78 | 0.24 | 0.14 | 0.18 |

### Key Insights

**Best for churner detection:** Oversampling achieved the highest churner recall (88%) and F1-score (0.78).

**Most balanced:** Undersampling gave the most similar performance across both classes (F1 of 0.73 vs 0.76).

**SMOTE performance:** Nearly identical to oversampling, showing synthetic samples can work just as well as duplicates.

**Ensemble failure:** Sometimes techniques that work in theory don't work in practice—always validate.

---

## Key Takeaways

### About Imbalanced Data

1. **Accuracy is a trap.** With imbalanced data, high accuracy often means your model is just predicting the majority class.Always check precision, recall,
2. and F1-score for each class.

3. **The minority class matters most.** In churn prediction, catching 88% of churners at 75% accuracy is far more valuable than 78% accuracy while missing 61% of churners.

4. **Business context drives metrics.** For churn, false negatives (missing churners) are more costly than false positives (flagging someone who won't churn),
so prioritize recall on class 1.

### About the Methods

4. **Undersampling is fast and simple** but wastes data. Best when you have millions of samples and can afford to discard most of the majority class.

5. **Oversampling keeps all data** but risks overfitting. Worked best in this project for churner detection.

6. **SMOTE creates synthetic samples** that can reduce overfitting compared to duplication. Requires proper feature scaling and tuning to shine.

7. **Ensemble methods aren't magic.** They can fail if individual models don't train properly or the combination strategy is wrong.

### About Model Evaluation

8. **Clean your data first.** Handling missing values, fixing data types, and normalizing features are non-negotiable before any modeling.

9. **Test multiple approaches.** What works for one dataset might fail for another. Always experiment and compare.

10. **Real-world ML is iterative.** This project tested 4 methods and one failed completely—that's normal and expected.

---

## What's Next: Advanced Techniques to Try

### Focal Loss

Instead of treating all misclassifications equally, focal loss gives more weight to hard-to-classify minority samples. 
The math behind focal loss adjusts the standard cross-entropy by down-weighting easy examples, forcing the model to focus on difficult ones.

**Research this:** Understand how the focusing parameter γ controls the rate at which easy examples are down-weighted.

### Class Weighting

Tell the model to care more about minority class errors during training by assigning higher loss weights to class 1. Can be combined with resampling techniques.

### Threshold Optimization

Instead of using 0.5 as the decision threshold, optimize it based on validation set recall or F1-score. This can shift the precision-recall trade-off in your favor.

### Feature Engineering

Create interaction terms or derived features like:
- Charge per month of tenure
- Contract type × senior citizen
- Service bundle indicators

### Model Tuning

Experiment with:
- Different architectures (more/fewer layers)
- Dropout rates
- Learning rates
- Batch sizes
- Number of epochs with early stopping

### Try Other Models

Compare the neural network to:
- Random Forest (handles imbalance naturally)
- XGBoost (has built-in class weighting)
- SVM with class weights

---

## How to Use This Project

**Requirements:**
- Python 3.x
- pandas, numpy, scikit-learn, tensorflow, imbalanced-learn

**Files:**
- `Imbalance.ipynb`: Complete notebook with all experiments
- `customer_churn.csv`: Dataset from Kaggle

**To reproduce:**
1. Install required libraries
2. Download the dataset from Kaggle
3. Open the notebook and run cells sequentially
4. Each section is clearly labeled with the technique being tested

**To extend:**
- Try the advanced techniques listed above
- Experiment with different model architectures
- Test on your own imbalanced dataset
- Compare more resampling strategies (ADASYN, BorderlineSMOTE, etc.)

---

## References and Further Learning

**Imbalanced Data Techniques:**
- Imbalanced-learn documentation: https://imbalanced-learn.org/
- SMOTE paper: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique"

**Neural Networks:**
- TensorFlow/Keras documentation: https://www.tensorflow.org/

**Evaluation Metrics:**
- Scikit-learn metrics guide: https://scikit-learn.org/stable/modules/model_evaluation.html

**Focal Loss:**
- Original paper: Lin et al., "Focal Loss for Dense Object Detection"

---

**This project serves as a complete reference for anyone dealing with imbalanced classification problems. The techniques are applicable to fraud detection,
medical diagnosis, anomaly detection, and many other real-world scenarios where one class is rare but critically important.**
