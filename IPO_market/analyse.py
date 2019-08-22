import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
# load datas
%matplotlib inline
ipos = pd.read_csv(r'ipo_data_2.csv', encoding='latin-1')
ipos['Date'] = pd.to_datetime(ipos['Date'])
X = pd.read_csv(r'X.csv')

# tiaoshi model
idx = 1902
X_train, X_test = X[:idx], X[idx:]
y_train = ipos['$ Chg Open to Close'][:idx].map(lambda x: 1 if x >=.25 else 0)
y_test = ipos['$ Chg Open to Close'][idx:].map(lambda x: 1 if x >=.25 else 0)

clf = linear_model.LogisticRegression()
a = clf.fit(X_train, y_train)
print(a)
b = clf.score(X_test, y_test)
print(b)   #accuracy evaluation

c = ipos[(ipos['Date']>='2014-01-01')]['$ Chg Open to Close'].describe()
print(c)

# total table
pred_label = clf.predict(X_test)
results = []
for pl, tl, idx, chg in zip(pred_label, y_test, y_test.index, ipos.iloc[y_test.index]['$ Chg Open to Close']):
    if pl == tl:
        results.append([idx, chg, pl, tl, 1])
    else:
        results.append([idx, chg, pl, tl, 0])
rf = pd.DataFrame(results, columns=['index', '$ chg', 'predicted', 'actual', 'correct'])
print(rf)

# profit according to prediction
d = rf[rf['predicted']==1]['$ chg'].describe()
print(d) 


fig, ax = plt.subplots(figsize=(15,10))
rf[rf['predicted']==1]['$ chg'].plot(kind='bar')
ax.set_title('Model Predicted Buys', y=1.01)
ax.set_ylabel('$ Change Open to Close')
ax.set_xlabel('Index')
plt.show()

# to analyse the contribution-rate of each coef
fv = pd.DataFrame(X_train.columns, clf.coef_.T).reset_index()
fv.columns = ['Coef', 'Feature']
e = fv.sort_values('Coef', ascending=0).reset_index(drop=True)
print(fv)
f = fv[fv['Feature'].str.contains('Week Day')]
print(f)

"""
g = ipos[ipos['Lead Mgr'].str.contains('Keegan|Towbin')]
print(g)

"""

clf_rf = RandomForestClassifier(n_estimators=500)
model = clf_rf.fit(X_train, y_train)
clf_rf.score(X_test, y_test)

f_importances = clf_rf.feature_importances_
f_names = X_train
f_std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
zz = zip(f_importances, f_names, f_std)
zzs = sorted(zz, key=lambda x: x[0], reverse=True)
imps = [x[0] for x in zzs[:20]]
labels = [x[1] for x in zzs[:20]]
errs = [x[2] for x in zzs[:20]]
plt.subplots(figsize=(15,10))
plt.bar(range(20), imps, color='r', yerr=errs, align="center")
plt.xticks(range(20), labels, rotation=-70)
plt.show()

