🌌 Stellar Object Classification (Galaxy, Star, QSO)

A full-stack Machine Learning project that classifies celestial objects — Galaxies, Stars, and Quasars (QSO) — using real astronomical data.

This project combines data science, astrophysics insights, and web development into one application.
It’s not just a model, but a deployable interactive tool where users can input magnitude and spectral data to classify celestial bodies.

🚀 Features

Exploratory Data Analysis (EDA):

Visualized distributions of magnitudes (u, g, r, i, z).

Plotted Color Index vs Redshift to show astrophysical separations.

Multiple Models Compared:

Logistic Regression

Decision Tree

Random Forest

XGBoost (with GridSearchCV hyperparameter tuning)

Scientific Insights:

Feature importance reveals which magnitudes matter most.

Physics connection:

Redshift separates galaxies from stars.

Color indices (g-r) are critical for finding quasars.

Full-Stack Application:

Flask backend serving predictions.

Frontend (HTML/CSS/JS) with a clean dark-mode astronomy UI.

Dynamic form inputs with real-time JSON communication.

Deployment Ready:

REST API with JSON responses.

Class probabilities shown with confidence scores.

📊 Dataset

The project uses the SDSS (Sloan Digital Sky Survey) star classification dataset, containing spectral and magnitude information for different celestial objects.

Features include:

Magnitudes: u, g, r, i, z

Spatial Coordinates: alpha (Right Ascension), delta (Declination)

Redshift

Derived feature: g-r (Color Index)

Target classes:

Galaxy

QSO (Quasi-Stellar Object / Quasar)

Star

🧠 Machine Learning Workflow

Data Preprocessing

Converted magnitudes to numeric.

Scaled numerical features using StandardScaler.

Encoded target labels with LabelEncoder.

Model Training & Evaluation

Trained multiple classifiers.

Evaluated using Classification Report, ROC-AUC, Confusion Matrices.

XGBoost with GridSearchCV achieved the best results.

Feature Importance (Scientific Insight)

Redshift = strong separator of galaxies.

Color Index (g-r) = essential in detecting quasars.

Magnitudes (u, g, r, i, z) drive classification accuracy.

🖥️ Application (Frontend + Backend)

Users enter values for magnitudes, redshift, and coordinates.

Flask backend processes inputs, applies preprocessing pipeline, and returns predictions.

Frontend shows:

Predicted class (Galaxy / Star / QSO)

Class probabilities with percentage confidence.

🔧 Tech Stack

Python (Pandas, NumPy, Matplotlib, Seaborn) – Data Analysis & Visualization

Scikit-learn, XGBoost – ML Modeling & Hyperparameter Tuning

Flask – Backend API

HTML, CSS, JavaScript – Frontend UI

Joblib – Model persistence

📸 Demo Screenshots

Color Index vs Redshift

ROC Curves for Multi-Class Classification

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/20883582-94ba-4929-b972-8fd1e90911e1" />


Feature Importance (XGBoost)

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/b5aa5ea2-5d28-4408-93db-02946133d92b" />


Frontend Web App (Dark UI with prediction output)

<img width="1915" height="868" alt="image" src="https://github.com/user-attachments/assets/2f39fe9e-0386-441f-8432-d449f8ec4417" />

<img width="1886" height="873" alt="image" src="https://github.com/user-attachments/assets/e1b285fe-c462-4f34-9660-47b27f281e8d" />




🚀 How to Run Locally

Clone the repo:

git clone https://github.com/your-username/stellar-classifier.git
cd stellar-classifier


Install dependencies:

pip install -r requirements.txt


Train model & save:

python model.py


Run Flask app:

python app.py


Open in browser:

http://127.0.0.1:5000

🌠 Why This Project Matters

Shows ability to work with real-world scientific datasets.

Solves a real world Astrophysics problem.

Combines ML + Hyperparameter Tuning + Web Development.

Demonstrates end-to-end ownership of a project:

Data → Model → Insights → Frontend → Deployment.

This is the kind of project that bridges machine learning and scientific application.

📌 Future Improvements

Deploy app on Heroku / AWS / Render.

Add interactive visualizations of redshift and color index.

Explore Deep Learning (CNNs on spectra) for higher accuracy.


