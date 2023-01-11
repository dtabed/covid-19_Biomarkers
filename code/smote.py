# SMOTE method to oversample the minority class
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where

data_train = pd.read_csv('../data/All_samples.csv')
X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
