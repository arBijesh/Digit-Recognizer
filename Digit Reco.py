# %% [markdown]
# ### Digit Recognizer

# %%
#Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# %%
#Importing the Datasets

train = pd.read_csv(r"D:\Data Science\Hackathon Projects\Digit Recognizer\digit-recognizer\train.csv")
train.head()

# %%
test = pd.read_csv(r"D:\Data Science\Hackathon Projects\Digit Recognizer\digit-recognizer\test.csv")
test.head()

# %%
submission = pd.read_csv(r"D:\Data Science\Hackathon Projects\Digit Recognizer\digit-recognizer\sample_submission.csv")
submission.head()

# %% [markdown]
# ### Logistic Regression

# %%
#Logistic Regression can handle multiclass Problem...

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()

X = train.drop('label', axis = 1)
y = train['label']

pred = lg.fit(X, y).predict(test)

# %%
pred
submission['Label'] = pred
submission.to_csv('submission_logistic.csv', index = False)#Accuracy 0.91857

# %% [markdown]
# ### Decision Tree Classifier

# %%
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

pred = dtree.fit(X, y).predict(test)

submission['Label'] = pred

submission.to_csv('submission_dtree.csv', index = False)#Accuracy 0.85742

# %% [markdown]
# ### Random Forest Classifier

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()  

pred_rf = rf.fit(X, y).predict(test)

submission['Label'] = pred_rf
submission.to_csv('submission_rf.csv', index = False)#Accuracy 0.96578

# %% [markdown]
# ### ADA Boost

# %%
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(learning_rate=0.1, n_estimators=100)

pred_ada1 = ada.fit(X, y).predict(test)

submission['Label'] = pred_ada1
submission.to_csv('submission_ada1.csv', index = False)#Accuracy 0.73582

# %% [markdown]
# ### Gradient Boosting Classifier

# %%
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier()

pred_gbm = gbm.fit(X, y).predict(test)

submission['Label'] = pred_gbm
submission.to_csv('submission_gbm.csv', index = False)#Accuracy 0.94

# %% [markdown]
# ### Light Gradient Boosting Classifier

# %%
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
pred_lgbm = lgbm.fit(X, y).predict(test)

submission['Label'] = pred_lgbm
submission.to_csv('submission_lgbm.csv', index = False)#Accuracy 0.96942

# %%
submission

# %%



