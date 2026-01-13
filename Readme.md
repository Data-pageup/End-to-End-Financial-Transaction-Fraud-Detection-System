# Fraud Detection System — End-to-End Machine Learning Project

**Author:** Amirtha Ganesh R.  
**Project Type:** Production-Ready Fraud Detection Model  
**Technologies:** Python, XGBoost, scikit-learn, Pandas, Matplotlib, Seaborn

---

## Project Overview

This project presents a **complete, end-to-end fraud detection system** designed to identify fraudulent financial transactions in real-time while minimizing customer friction. The solution goes beyond model building to address deployment strategy, threshold optimization, explainability, and production monitoring considerations.

### Business Context

Digital financial fraud results in billions of dollars in annual losses globally. This project tackles fraud detection as a **cost-sensitive, highly imbalanced classification problem** where:
- **False negatives** lead to direct financial loss
- **False positives** damage customer experience and brand trust
- The fraud rate is extremely low (~0.13% of transactions)
- Real-time decision-making is critical

---

##  Problem Statement

Develop a machine learning model to:
1. **Detect fraudulent transactions** with high accuracy
2. **Handle extreme class imbalance** (~99.87% legitimate transactions)
3. **Minimize customer friction** from false alerts
4. Provide **explainable, production-ready** fraud scoring
5. Design **threshold-based decision logic** for operational deployment

---

## Dataset Summary

- **Source:** Synthetic financial transaction data (Kaggle)
- **Size:** ~6.36 million transactions
- **Fraud Rate:** 0.13% (8,213 fraudulent out of 6,362,620 total)
- **Transaction Types:** PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT
- **Key Finding:** Fraud occurs **exclusively** in TRANSFER and CASH_OUT transactions

### Features
- `step`: Time unit (1 hour intervals over 30 days)
- `type`: Transaction type
- `amount`: Transaction amount
- `oldbalanceOrg/newbalanceOrig`: Origin account balances before/after
- `oldbalanceDest/newbalanceDest`: Destination account balances before/after
- `isFraud`: Target variable (fraud label)

---

##  Methodology

### 1. **Data Understanding & Exploratory Analysis**
- Identified fraud concentration in specific transaction types
- Discovered hidden missing values in destination balances (~50% of fraud)
- Analyzed amount distributions and behavioral patterns
- Validated domain assumptions about fraud mechanics

**Key Insights:**
- Fraud exists **only** in TRANSFER (4,097 cases) and CASH_OUT (4,116 cases)
- Fraudulent transactions show systematic balance inconsistencies
- Account IDs carry no behavioral signal (removed to prevent overfitting)

### 2. **Fraud-Aware Data Cleaning**
**Critical Design Decisions:**
- **Hidden Missing Values:** Encoded zero destination balances as `-1` rather than imputing
  - *Rationale:* Missingness itself is a fraud signal
- **Outliers Preserved:** No capping or removal
  - *Rationale:* Fraud manifests as extreme values
- **Class Imbalance Maintained:** No SMOTE or undersampling
  - *Rationale:* Model must operate on real-world distribution
- **Multicollinearity:** Addressed via feature engineering, not variable removal

### 3. **Feature Engineering**
Created domain-driven features encoding **money conservation violations**:
```python
errorBalanceOrig = newBalanceOrig + amount - oldBalanceOrig
errorBalanceDest = oldBalanceDest + amount - newBalanceDest
```

**Why These Features Matter:**
- Capture accounting inconsistencies characteristic of fraud
- Provide strongest predictive signal (confirmed via feature importance)
- Enable interpretable fraud detection logic

**Features Removed:**
- `nameOrig`, `nameDest`: High-cardinality identifiers with no behavioral value
- `isFlaggedFraud`: Inconsistent and uninformative

### 4. **Model Selection & Training**

**Models Evaluated:**
| Model | AUPRC | Selection Rationale |
|-------|-------|---------------------|
| Logistic Regression | 0.5717 | Linear baseline — confirms non-linear patterns exist |
| Random Forest | 0.9987 | Strong performance, slight overfitting tendency |
| **XGBoost** | **0.9930** |  **Final choice**: Better generalization, production-ready |

**Why XGBoost?**
- Native class imbalance handling (`scale_pos_weight`)
- Controlled regularization (prevents memorization)
- Fast inference suitable for real-time scoring (<100ms target)
- Industry standard for fraud detection
- Strong feature importance interpretability

**Training Configuration:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=774,  # handles 1:774 class ratio
    eval_metric='aucpr',
    random_state=42
)
```

### 5. **Evaluation Metrics**

**Primary Metric: AUPRC** (Area Under Precision-Recall Curve)
- **Why not Accuracy?** 99.87% accuracy achievable by predicting all non-fraud
- **Why not ROC-AUC?** Masks false-positive cost under extreme imbalance
- **AUPRC** focuses on rare fraud class and ranking quality

**Final Performance:**
- **AUPRC:** 0.9930
- **Precision @ 95% Recall:** 0.9574
- **Recall @ 99% Precision:** 0.8728

### 6. **Threshold Optimization**

Rather than a single cutoff, defined **business-aligned operating zones**:

| Score Range | Decision | Precision | Recall | False Positives |
|-------------|----------|-----------|--------|-----------------|
| ≥ 0.9997 | **Auto-block** | 99.03% | 87.28% | 14 |
| 0.9985 – 0.9997 | **Manual review** | 95.74% | 95.74% | 70 |
| 0.9964 – 0.9985 | **High-recall monitoring** | 89.44% | 99.03% | 192 |
| < 0.9964 | **Approve** | — | — | — |

**Business Impact Translation:**
- **High Precision Zone:** Minimizes customer disruption for auto-blocking
- **Balanced Zone:** Maximizes fraud capture with acceptable alerts
- **High Recall Zone:** Reserved for high-risk periods (holidays, attacks)

---

## 🔍 Model Explainability

### Feature Importance (Top Drivers)
1. **newbalanceOrig** (40%) — Post-transaction balance behavior
2. **errorBalanceOrig** (19%) — Origin balance inconsistencies
3. **errorBalanceDest** (12%) — Destination balance violations
4. **type_PAYMENT** (8%) — Negative indicator (inherently safe)
5. **amount** (4%) — Transaction size
6. **oldbalanceOrig** (3%) — Pre-transaction state

**Domain Validation:**
✅ Balance-based features dominate → confirms fraud violates money conservation  
✅ Transaction type acts as risk gate → aligns with fraud mechanics  
✅ Amount refines but doesn't define risk → prevents crude heuristics  
✅ No identity-based features → ensures behavioral, not memorization-based detection

**Leakage Check:**
- ❌ No target variables
- ❌ No future information
- ❌ No account identifiers
- ✅ All features available at transaction time

---
##  Deployment Strategy

### System Architecture
```
Transaction Initiated
         ↓
Feature Extraction (real-time)
         ↓
XGBoost Model (REST API)
         ↓
Fraud Risk Score (0–1)
         ↓
Decision Engine (threshold-based)
         ↓
Action: Approve / Review / Block
```

### Key Components
- **Real-time Scoring:** REST API with <100ms latency target
- **Threshold-based Decisions:** Multi-zone approach instead of binary cutoff
- **Human-in-the-Loop:** Borderline cases routed to fraud analysts
- **Fallback Mechanisms:** Rule-based system if model unavailable
- **Logging:** All decisions tracked for retraining

---

##  Monitoring & Maintenance

### Model Performance Metrics (Daily/Weekly)
- Precision, Recall, False Positive Rate
- AUPRC (offline validation)
- Alert volume trends

### Business Metrics (Primary)
-  Fraud loss prevented ($)
-  Customer complaint rate
-  Manual review workload
-  Review SLA compliance

### Drift Detection
- **Feature Distribution Shifts:** PSI (Population Stability Index)
- **Score Distribution Changes:** Trigger investigation if PSI > 0.2
- **Transaction Mix Changes:** Monitor type/amount patterns

### Retraining Strategy
- **Scheduled:** Monthly retraining with new fraud labels
- **Triggered:** Earlier retraining if drift detected
- **Feedback Loop:** Incorporate analyst decisions
- **Validation:** Revalidate thresholds before deployment

### Success Validation (A/B Testing)
- **Control:** Existing rule-based system
- **Treatment:** ML-driven decisions
- **Success Criteria:**
  - ≥20% fraud loss reduction
  - ≤5% false positive increase
- **Duration:** 30-day test window

---

##  Key Technical Decisions

### Why These Choices Matter

| Decision | Reasoning |
|----------|-----------|
| **Preserve class imbalance** | Models must perform on real-world distribution |
| **No SMOTE/undersampling** | Artificial balancing distorts fraud patterns |
| **Encode zeros as -1** | Missing balances are fraud signal, not noise |
| **Keep all outliers** | Fraud lives in distribution tails |
| **XGBoost over Random Forest** | Better generalization despite slightly lower AUPRC |
| **AUPRC over accuracy** | Focuses on rare fraud class, not majority |
| **Multi-threshold zones** | Balances fraud prevention with customer experience |
| **Balance error features** | Domain knowledge encoded mathematically |


---

##  Technologies Used

- **Python 3.10+**
- **Core ML:** XGBoost, scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Evaluation:** Precision-Recall curves, Feature importance
- **Development:** Jupyter Notebook

---

##  Evaluation Questions Answered

This project addresses eight critical evaluation criteria:

1. ✅ **Data Cleaning:** Fraud-aware handling of missing values, outliers, multicollinearity
2. ✅ **Model Description:** XGBoost with class weighting and threshold-based decisions
3. ✅ **Variable Selection:** Domain knowledge + EDA + feature importance
4. ✅ **Performance Demonstration:** AUPRC, threshold analysis, business impact translation
5. ✅ **Key Fraud Factors:** Balance behavior, transaction type, amount patterns
6. ✅ **Domain Validation:** Features align with fraud mechanics and business logic
7. ✅ **Prevention Strategy:** Layered defense (ML + rules + human review)
8. ✅ **Success Measurement:** A/B testing, monitoring, drift detection

---

##  Results Summary

### Model Performance
- **AUPRC:** 0.9930 (exceptional for extreme imbalance)
- **Balanced Threshold:** 95.74% precision and recall
- **False Positives:** 70 out of 1.27M test transactions at optimal threshold

### Business Impact
- Catches ~96% of fraud with minimal customer disruption
- Provides interpretable, threshold-based decision framework
- Production-ready architecture with monitoring strategy

### Technical Achievements
- Domain-driven feature engineering (balance error features)
- Fraud-aware data cleaning (preserved signal, not distorted)
- Multi-threshold decision logic (not binary classification)
- Explainable predictions (feature importance aligns with domain)
- Deployment considerations (monitoring, drift, retraining)

---

## 🔗 Links & References

- **Dataset Source:** [Kaggle - Synthetic Financial Fraud Detection] - [https://www.kaggle.com/code/arjunjoshua/predicting-fraud-in-financial-payment-services/input]


---

## 📧 Contact

**Amirtha Ganesh R.**  
- **Email:** amirthaganeshramesh@gmail.com 

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note:** This project was developed as part of a data science internship assignment, demonstrating end-to-end machine learning capabilities from problem understanding to deployment strategy in 3 hours given time.
 
 
 
 
