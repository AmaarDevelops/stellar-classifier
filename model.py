import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder,label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



df = pd.read_csv('star_classification.csv',encoding='latin1')

#Exploratry Data Analysis
null_values = df.isnull().sum()
df_shape = df.shape
columns = df.columns.unique()


print('Null values :-',null_values)
print('\nDF Shape :-', df_shape)
print('\nColumns :-', columns)

#Visualizing distribution of magnitudes

magnitude_cols = ["u","g","r","i","z"]
for col in magnitude_cols:
    df[col] = pd.to_numeric(df[col],errors='coerce')

df.dropna(subset=magnitude_cols,inplace=True)




fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(15,15))
fig.suptitle('Distribution of Magnitudes',fontsize=20)

for i, mag in enumerate(magnitude_cols):
    row = i // 2
    col = i % 2
    sns.histplot(data=df,x=mag,kde=True,ax=axes[row,col],color='teal')
    axes[row,col].set_title(f'Histogram of {mag} Magnitude',fontsize=14)
    axes[row,col].set_xlabel(f'${mag} Magnitude')
    axes[row,col].set_ylabel('Count')

axes[2,1].axis('off')
plt.tight_layout(rect=[0,0.03,1,0.95])

plt.savefig('visuals/distrubtion_of_magnitudes.png')
plt.close()

#Visualizing Red Shift vs Color Index
df['g-r'] = df['g'] - df['r']
plt.figure(figsize=(10,8))

sns.scatterplot(data=df,x='g-r',y='redshift',hue='class',style='class',s=5)

plt.title('Color Shift vs. Red index')
plt.xlabel('Color Index')
plt.ylabel('Redshift')
plt.grid(True)

plt.savefig('visuals/color_index vs redshift')
plt.close()



#Data preprocessing


x = df.drop(columns=['obj_ID', 'class', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID'])
y = df['class']

Numerical_cols = ['u','g','i','r','z','alpha','delta','redshift','g-r']

preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),Numerical_cols)
    ],
    remainder='passthrough'
)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.20,random_state=42)


#Model Training & Evaluating

#logistic regression

lr_pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression(random_state=42,max_iter=1000))
])

lr_pipeline.fit(x_train,y_train)

y_pred_lr = lr_pipeline.predict(x_test)

classification_report_lr = classification_report(y_test,y_pred_lr)
roc_auc_score_lr = roc_auc_score(y_test,lr_pipeline.predict_proba(x_test),multi_class='ovr')
confusion_matrix_lr = confusion_matrix(y_test,y_pred_lr)

print('\nClassification Report of Logistic Regression :-', classification_report_lr)
print('\n ROC AUC SCORE OF LOGISTIC REGRESSION :-', roc_auc_score_lr)
print('\n Confusion Matrix :-', confusion_matrix_lr)

#Decision Tree

dt_pipeline = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('classifier',DecisionTreeClassifier(random_state=42,max_depth=10))
    ]
)

dt_pipeline.fit(x_train,y_train)
y_pred_dt = dt_pipeline.predict(x_test)

classification_report_dt = classification_report(y_test,y_pred_dt)
roc_auc_score_dt = roc_auc_score(y_test,dt_pipeline.predict_proba(x_test),multi_class='ovr')
confusion_matrix_dt = confusion_matrix(y_test,y_pred_dt)

print('\n Classification Report of decision tree :-', classification_report_dt)
print(f'\nROC AUC Score of Decision Tree {roc_auc_score_dt}')
print(f'\nConfusion Matrix Decision Tree {confusion_matrix_dt}')

#Random Forest

pipeline_rf = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('classifier',RandomForestClassifier(random_state=42))
    ]
)

pipeline_rf.fit(x_train,y_train)
y_pred_rf = pipeline_rf.predict(x_test)

classifcation_report_rf = classification_report(y_test,y_pred_rf)
roc_auc_score_rf = roc_auc_score(y_test,pipeline_rf.predict_proba(x_test),multi_class='ovr')
confusion_matrix_rf = confusion_matrix(y_test,y_pred_rf)

print(f'\nClassification report of Random Forest {classifcation_report_rf}')
print(f'\nRoc Auc Score of random forest {roc_auc_score_rf}')
print(f'\nConfusion Matrix of Random Forest {confusion_matrix_rf}')

#XGBoost with Hyperparameter Tuning

cv_splitter = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)

pipeline_xg = Pipeline(
    steps=[
        ('preprocesser',preprocessor),
        ('classifier',XGBClassifier(eval_metric='mlogloss',use_label_encoder=False,random_state=42))
    ]
)

param_grid_xg = {
    'classifier__n_estimators' : [100,200,300],
    'classifier__learning_rate' : [0.05,0.1,0.2],
    'classifier__max_depth' : [3,5,7],
    'classifier__subsample' : [0.6,0.8,1.0]
} 

grid_search_xg = GridSearchCV(
    estimator=pipeline_xg,
    param_grid=param_grid_xg,
    cv=cv_splitter,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)


grid_search_xg.fit(x_train,y_train)

print(f'Best params for grid search XG {grid_search_xg.best_params_}')
print(f'Best cross validated ROC AUC Score {grid_search_xg.best_score_}')

best_model_xg = grid_search_xg.best_estimator_
y_pred_xg = best_model_xg.predict(x_test)

classification_report_xg = classification_report(y_test,y_pred_xg)
confusion_matrix_xg = confusion_matrix(y_test,y_pred_xg)
roc_auc_score_xg = roc_auc_score(y_test,best_model_xg.predict_proba(x_test),multi_class='ovr')

print(f'\n XG Classification report :- {classification_report_xg}')
print(f' XGB Confusion matrix :- {confusion_matrix_xg}')
print(f' XGB Roc Auc Score :- {roc_auc_score_xg}')

#Roc Curve for XGBOOST

y_test_binarized = label_binarize(y_test,classes=[0,1,2])
y_test_proba = best_model_xg.predict_proba(x_test)
class_labels = label_encoder.classes_

plt.figure(figsize=(10,8))
for i,class_name in enumerate(class_labels):
    RocCurveDisplay.from_predictions(
        y_test_binarized[:,i],
        y_test_proba[:,i],
        name=f"ROC curve for {class_name} (AUC = {roc_auc_score(y_test_binarized[:, i], y_test_proba[:, i]):.2f})",
    )

plt.plot([0,1] , [0,1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.title('Multi-Class ROC Curve for Xg boost')
plt.xlabel('False Positive')
plt.ylabel('True positivies')
plt.savefig('visuals/xgb_roc_curves.png')
plt.close()

# --- Scientific Insights : Feature Importance --- 
feature_importances = best_model_xg.named_steps['classifier'].feature_importances_
feature_names = x.columns

important_df = pd.DataFrame({
    'Feature' : feature_names,
    'Importance' : feature_importances,
}).sort_values(by='Importance',ascending=False)

print('\n\n --- Feature Importance Results ---')
print(important_df)

plt.figure(figsize=(12,8))
sns.barplot(x='Importance',y='Feature',data=important_df,palette='viridis')
plt.title('XGBoost Feature Importance',fontsize=16)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('visuals/feature_importance.png')
plt.close()


# --- Scientific Insights ---

print("\n--- Scientific Insights ---")

print("1. Redshift is highly important, which makes sense: stars are nearby with low redshift, "
      "while galaxies and quasars have significant redshift values due to cosmic expansion.")

print("2. Color indices (like g-r, u-g) are also critical: astronomers use these to separate "
      "stellar populations from galaxies and quasars. For example, quasars are typically bluer "
      "while galaxies are redder.")

print("3. Magnitudes across different filters (u, g, r, i, z) help capture spectral energy "
      "distributions. Quasars are bright in u-band, galaxies show broader light distribution, "
      "and stars fall on the stellar locus.")

print("4. Positional features (alpha, delta) have low importance, as expected, since object "
      "classification should be independent of where the object lies in the sky.")

#JobLib Dumping

import joblib

best_model = best_model_xg
features = feature_names

joblib.dump(best_model,'best_model_stellar.joblib')
joblib.dump(features.tolist(),'feature_list.joblib')






