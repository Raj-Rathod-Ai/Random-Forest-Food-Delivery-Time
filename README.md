# Food Delivery Time Prediction using Random Forest

A Machine Learning project that predicts food delivery time using the Random Forest algorithm. Built with Python and Scikit-learn, this project analyzes delivery-related factors to provide accurate delivery time estimates. It features a custom **Random Forest Regressor** trained on historical delivery data, a clean preprocessing pipeline with saved label encoders, and an interactive **Streamlit web application** for real-time predictions.

## 🚀 Live Streamlit Application
The application is ready for deployment on platforms like Streamlit Community Cloud. It allows users to input trip details, environment settings, and courier parameters through a clean question-based interface (no sliders) to get immediate delivery time estimations.

## 📊 Model & Hyperparameter Tuning

Through exploratory data analysis (EDA), ANOVA statistical testing, and preprocessing, we optimized a **Random Forest Regressor** using randomized search:

### Custom Hyperparameters
- **N Estimators**: `781`
- **Max Depth**: `11`
- **Max Features**: `'sqrt'`
- **Min Samples Split**: `3`
- **Min Samples Leaf**: `3`
- **Random State**: `42`

### Model Performance
- **Mean Absolute Error (MAE)**: `7.36` mins
- **Root Mean Squared Error (RMSE)**: `9.95` mins
- **R-squared (R2)**: `0.78`

---

## 🛠️ Project Structure
```
├── .github/workflows/
│   └── keep_alive.yml          # GitHub Action to keep deployed app awake
├── app.py                      # Streamlit Web Application
├── random_forest_model.pkl      # Pre-trained Random Forest model
├── label_encoders.pkl          # Saved label encoders for categorical fields
├── RandomForest.ipynb          # Jupyter notebook for training and EDA
├── Food_Delivery_Times.csv     # Historical delivery dataset
├── requirements.txt            # Package dependencies
└── .gitignore                  # Git ignore rules
```

---

## 🏃 Local Run Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8501`.

---

## 🔄 Sleep Prevention (Streamlit Community Cloud)

Streamlit Community Cloud automatically puts applications to sleep if they do not receive traffic. To prevent this, this repository includes a GitHub Actions keep-alive workflow ([keep_alive.yml](.github/workflows/keep_alive.yml)) that pings the application every 4 hours.

**Setup Instructions**:
1. Go to your repository settings on GitHub.
2. Navigate to **Settings** > **Secrets and variables** > **Actions**.
3. Create a **New repository secret**:
   - **Name**: `APP_URL`
   - **Value**: `https://<your-app-subdomain>.streamlit.app` (your deployed app URL)
