import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-darkgrid')
# Setting random seed
seed = 57

#### Data Preparation ####

# Loading data
df = pd.read_csv('heart.csv')

# Spliting data
y = df.pop('target')
X_train,  X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = seed)


#### Modeling ####

# Fitting a model
clssr = GradientBoostingClassifier(loss='exponential',criterion = 'mae', learning_rate = 0.9, random_state = seed)
#clssr = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = 0.9, random_state = seed, max_iter = 150)
clssr.fit(X_train, y_train)

# Reporting 
train_score = clssr.score(X_train, y_train) * 100
test_score = clssr.score(X_test, y_test) * 100

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Training accuracy explained: %2.1f%%\n" % train_score)
        outfile.write("Test accuracy explained: %2.1f%%\n" % test_score)        

#### Feature importances plot ####

# Calculate feature importance or weights
try:
        importances = clssr.coef_[0]
        col = "Weights"
        model_name = "Logistic Regression"
except:
        importances = clssr.feature_importances_
        col = "Importance"
        model_name = 'Gradient Boosting Classifier'
        
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature",col])
feature_df = feature_df.sort_values(by=col, ascending=False,)

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="darkgrid")

ax = sns.barplot(x=col, y="feature", data=feature_df)
ax.set_xlabel(col ,fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)
ax.set_title(model_name + '\nFeature ' + col, fontsize = title_fs)

plt.tight_layout()
plt.savefig("features.png",dpi=120) 
plt.close()

#### ROC CURVE ###

# Calculating probabilities, false positives and true positives
y_predict = clssr.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_predict[:,1])
roc_auc = auc(fpr, tpr)

# Image formatting
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize = axis_fs)
ax.set_ylabel('True Positive Rate', fontsize = axis_fs)
ax.set_title(model_name + '\nROC Curve', fontsize = title_fs)
ax.legend(loc="lower right")

fig.tight_layout()
fig.savefig("ROC curve.png",dpi=120)
plt.close()
