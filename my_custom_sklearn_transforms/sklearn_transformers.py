from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


class CustomClassifier(BaseEstimator):

    def __init__(self):        
        self.m1 =  KNeighborsClassifier(n_neighbors=10)
        self.m2 = DecisionTreeClassifier(max_depth=3)
        self.m3 = svm.SVC(kernel = 'rbf')


    def fit(self, X, y=None, **kwargs):
        self.m1.fit(X, y)
        self.m2.fit(X, y)      
        diff = (self.m1.predict(X) != self.m2.predict(X))
        print(len(X[diff]))
        self.m3.fit(X[diff],y[diff])    
        return self

    def predict(self, X, y=None):
        r1 = self.m1.predict(X)
        r2 = self.m2.predict(X)
        diff = r1 != r2
        r3 = self.m3.predict(X[diff])
        r1[diff] = r3
        return r1
