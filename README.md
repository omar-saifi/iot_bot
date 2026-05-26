# 🛡️ IDS BoT-IoT: Intrusion Detection System for IoT

An advanced Intrusion Detection System (IDS) designed to detect and classify malicious activities in IoT networks using machine learning. This project combines a comprehensive Jupyter notebook for model development with an interactive Streamlit dashboard for real-time predictions.

---

## 📋 Project Overview

This IDS system is built on the **BoT-IoT dataset** and uses Naive Bayes classifiers to detect network attacks in IoT environments. The project provides two complementary interfaces:

- **Model Development & Analysis**: Jupyter notebook for data exploration, preprocessing, and model training
- **Interactive Dashboard**: Real-time prediction interface with detailed metrics and visualizations

---

## 🎯 Key Features

- **Binary Classification**: Normal vs. Attack detection
- **Multi-class Classification**: Detailed attack type categorization (DDoS, DoS, Reconnaissance, Theft, etc.)
- **Multiple Models**: Gaussian Naive Bayes and Complement Naive Bayes
- **Imbalanced Data Handling**: SMOTE for handling class imbalance
- **Cross-Validation**: Stratified K-fold cross-validation for robust evaluation
- **Real-time Predictions**: Interactive dashboard for live threat detection
- **Comprehensive Metrics**: Accuracy, F1-score, ROC-AUC, Precision, Recall, and Confusion Matrix
- **Modern UI**: Dark-themed, responsive Streamlit interface

---

## 📁 Project Structure

```
├── IDS_BoT_IoT.ipynb          # Main Jupyter notebook for model development
├── ids_dashboard.py            # Interactive Streamlit dashboard application
├── ids_models/                 # Trained models and metrics
├── data/                       # Dataset directory
└── data_1.csv                  # BoT-IoT dataset
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone or download the project**:
   ```bash
   cd projet_IOT/files
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn category-encoders imbalanced-learn streamlit joblib
   ```

3. **Ensure dataset is present**: Place `data_1.csv` in the project directory

---

## 📓 IDS_BoT_IoT.ipynb

The main Jupyter notebook for model development and analysis.

### Workflow

1. **📚 Data Loading**: Loads the BoT-IoT CSV dataset
2. **🔍 EDA**: Exploratory Data Analysis including:
   - Data type and null value analysis
   - Class distribution visualization
   - Feature statistics and correlations
3. **🔧 Data Preprocessing**:
   - Feature encoding and scaling
   - Handling missing values
   - Feature selection
4. **🤖 Model Development**:
   - Train/test split with stratification
   - Pipeline construction with SMOTE
   - Gaussian Naive Bayes and Complement Naive Bayes training
   - Cross-validation evaluation
5. **📊 Model Evaluation**:
   - Classification reports
   - Confusion matrices
   - ROC curves and AUC scores
   - Performance metrics visualization

### Usage

1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook IDS_BoT_IoT.ipynb
   ```

2. Run cells sequentially from top to bottom

3. Models and metrics are saved to `ids_models/` directory

---

## 💻 ids_dashboard.py

Interactive Streamlit dashboard for real-time threat detection and model evaluation.

### Features

- **Single Prediction**: Analyze individual network flows
- **Batch Prediction**: Process multiple records simultaneously
- **Model Comparison**: Compare performance of different classifiers
- **Metrics Dashboard**: View comprehensive model performance statistics
- **Visual Analytics**: Charts and graphs for attack patterns
- **Dark Theme UI**: Modern, professional interface with glassmorphism effects

### Running the Dashboard

```bash
streamlit run ids_dashboard.py
```

The application will open at `http://localhost:8501`

### Dashboard Sections

1. **Home**: Project overview and quick stats
2. **Predictions**: 
   - Single prediction for individual network flows
   - Batch prediction for multiple samples
3. **Model Metrics**: Detailed performance evaluation
4. **Analytics**: Visual analysis of attack patterns and model behavior

---

## 📊 Dataset

**BoT-IoT Dataset**:
- Comprehensive IoT network traffic dataset
- Contains normal and attack traffic samples
- Features: Network flow statistics (packet sizes, timing, protocols, etc.)
- Target columns:
  - `attack`: Binary (0=Normal, 1=Attack)
  - `category`: Multi-class attack types
  - `subcategory`: Granular attack classification

---

## 🎓 Models Used

### Gaussian Naive Bayes
- Assumes features follow Gaussian distribution
- Good baseline for continuous features
- Fast training and inference

### Complement Naive Bayes
- Variant designed for imbalanced datasets
- Better performance with skewed class distributions
- Complement to standard Naive Bayes

---

## 📈 Performance Metrics

The models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions

---

## 🔧 Configuration

### Model Parameters

Modify in the notebook:
- Train/test split ratio
- SMOTE sampling strategy
- Cross-validation folds
- Feature scaling parameters

### Dashboard Customization

Edit `ids_dashboard.py`:
- Color schemes and styling
- Layout configuration
- Input parameter ranges
- Display settings

---

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
category-encoders>=2.3.0
imbalanced-learn>=0.8.0
streamlit>=1.0.0
joblib>=1.0.0
```

---

## 💡 Usage Examples

### Notebook: Training a Model

```python
# Models are trained in the notebook with full preprocessing pipeline
# Predictions are made using trained models
y_pred = model.predict(X_test)
```

### Dashboard: Making Predictions

1. Navigate to "Predictions" section
2. Enter network flow features
3. Click "Predict" to get threat classification
4. View confidence scores and detailed analysis

---

## 🐛 Troubleshooting

**Data not loading?**
- Ensure `data_1.csv` is in the project root
- Check file path in notebook first cell

**Dashboard won't start?**
- Verify Streamlit is installed: `pip install streamlit`
- Check port 8501 is available

**Model loading errors?**
- Ensure models in `ids_models/` are properly saved
- Re-run notebook to regenerate models

---

## 📝 License

This project is designed for educational and research purposes.

---

## 👨‍💻 Development

### Future Enhancements
- Deep learning models (LSTM, Neural Networks)
- Real-time API integration
- Advanced feature engineering
- Explainability features (SHAP, LIME)
- Multi-model ensemble approach

---

## 📞 Support

For issues or questions, refer to:
- Jupyter notebook comments and markdown cells
- Dashboard code documentation
- Dataset documentation and papers on BoT-IoT

---

**Last Updated**: May 2026  
**Status**: Active Development
