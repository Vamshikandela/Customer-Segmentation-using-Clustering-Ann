# Customer-Segmentation-using-Clustering-Ann


# 🧠 Customer Segmentation Using ANN

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-Statistical%20Plots-7DB0BC?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Git-Version%20Control-F05032?style=for-the-badge&logo=git&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Dataset-200K%2B%20Records-4CAF50?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Features-29%20Columns-2196F3?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Completed-9C27B0?style=for-the-badge" />
</p>

---

A deep learning project that applies Artificial Neural Networks (ANN) to segment e-commerce customers into meaningful behavioral groups — enabling smarter marketing, churn prevention, and personalized customer experiences.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Machine Learning | Scikit-learn |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook / Google Colab |
| Deployment | Streamlit |
| Version Control | Git & GitHub |

---

## 🧰 Tools & Libraries

- **TensorFlow / Keras** — Building, training, and saving the ANN model
- **Scikit-learn** — Preprocessing (scaling, encoding), evaluation metrics, train-test split
- **Pandas & NumPy** — Data loading, cleaning, feature engineering
- **Matplotlib & Seaborn** — EDA visualizations, confusion matrices, training curves
- **Jupyter Notebook** — Interactive development environment
- **Streamlit** *(optional)* — Deploying an interactive segmentation dashboard

---

## 📊 Dataset Overview

The dataset is a rich e-commerce behavioral dataset with **200,000 records** and **29 features** per session.

| Property | Details |
|---|---|
| Total Records | 200,000 |
| Total Features | 29 |
| Purchase Rate | ~23.1% |
| Cart Abandonment Rate | ~42.6% |

**Key Features:**

- **Customer Behavior:** `pages_viewed`, `time_on_site_sec`, `added_to_cart`, `purchased`, `cart_abandoned`
- **Transaction Info:** `unit_price`, `quantity`, `discount_percent`, `discount_amount`, `revenue`, `revenue_normalized`
- **Session Context:** `device_type`, `user_type`, `marketing_channel`, `session_duration_bucket`
- **Product Info:** `product_id`, `product_category`, `rating`, `review_helpful_votes`
- **Temporal Features:** `visit_date`, `visit_day`, `visit_month`, `visit_weekday`, `visit_season`
- **Other:** `payment_method`, `location`

---

## 📁 Project Structure

```
customer-segmentation-ann/
│
├── data/
│   ├── customers.csv             # Raw dataset (200K records, 29 features)
│   └── processed/                # Cleaned and encoded data
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb    # Feature engineering & scaling
│   └── 03_Model_Training.ipynb   # ANN training & evaluation
│
├── models/
│   └── ann_model.h5              # Saved trained model
│
├── src/
│   ├── preprocess.py             # Data preprocessing pipeline
│   ├── model.py                  # ANN architecture definition
│   └── evaluate.py               # Metrics and evaluation utilities
│
├── outputs/
│   └── segmentation_results.csv  # Predicted customer segments
│
├── app.py                        # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## 🔄 Project Workflow

### Step 1 — Data Collection & Understanding
- Load the 200K e-commerce session dataset
- Inspect all 29 features: data types, missing values, distributions
- Understand business context — purchase behavior, cart abandonment (42.6%), revenue patterns

### Step 2 — Exploratory Data Analysis (EDA)
- Visualize purchase rates by device type, marketing channel, and season
- Analyze revenue distribution and discount impact
- Explore cart abandonment patterns across user types and session duration buckets
- Identify high-value behavioral signals: `pages_viewed`, `time_on_site_sec`, `rating`

### Step 3 — Data Preprocessing
- Handle any missing or anomalous values
- Encode categorical features (`device_type`, `user_type`, `marketing_channel`, `product_category`, `payment_method`, `session_duration_bucket`, `visit_season`)
- Scale numerical features using `StandardScaler` or `MinMaxScaler`
- Engineer derived features if needed (e.g., revenue per page, engagement score)
- Split data into training and testing sets (80/20)

### Step 4 — ANN Model Design
- Define the neural network architecture:
  - **Input Layer** — 29 neurons (one per feature)
  - **Hidden Layers** — Dense layers with ReLU activation + Dropout regularization
  - **Output Layer** — Softmax (multi-class segmentation) or Sigmoid (binary classification)
- Select optimizer (Adam), loss function, and evaluation metrics
- Apply Batch Normalization for training stability

### Step 5 — Model Training
- Train the ANN on preprocessed data with early stopping
- Monitor training vs. validation loss and accuracy
- Apply regularization techniques:
  - Dropout layers
  - L2 weight regularization
  - Early stopping with `restore_best_weights`

### Step 6 — Model Evaluation
- Evaluate on the held-out test set using:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC Curve
- Compare against baseline models (Logistic Regression, Random Forest)

### Step 7 — Customer Segmentation & Insights
- Assign each customer to a behavioral segment
- Profile each segment using statistical summaries:
  - **High-Value Buyers** — High revenue, high purchase rate, repeat visitors
  - **Browsers** — High page views, low conversion, frequent cart abandonment
  - **Discount Seekers** — High discount usage, price-sensitive behavior
  - **At-Risk / Dormant** — Low engagement, no recent purchases
- Visualize segments via 2D scatter plots (with PCA for dimensionality reduction)

### Step 8 — Deployment (Optional)
- Save the trained model as `.h5`
- Build a Streamlit app for real-time customer segment prediction
- Input a customer's session data → get their predicted segment

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/customer-segmentation-ann.git
cd customer-segmentation-ann

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook

# 5. (Optional) Run Streamlit app
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow>=2.10
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
streamlit
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Test Accuracy | 	99.94%


> *Update this section with your actual results after model training.*

---

## 🚀 Future Improvements

- Pre-cluster customers with K-Means before ANN classification for hybrid segmentation
- Hyperparameter tuning with Keras Tuner or Optuna
- Add LSTM layers to model sequential visit behavior over time
- Build a real-time Streamlit dashboard for segment exploration
- Export segment predictions to a CRM or marketing tool

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Kandela Vamshi**
- GitHub: https://github.com/Vamshikandela
- LinkedIn: https://www.linkedin.com/in/kandela-vamshi-2b4457258/
