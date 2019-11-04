data = pd.read_csv("data/bike_day_raw.csv")
X = data.drop("cnt", axis=1)
y = data.cnt

display(data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.linear_model import LinearRegression

# for other models you should scale here

lr = LinearRegression().fit(X_train, y_train)

print(lr.score(X_train, y_train))

print(lr.score(X_test, y_test))

from sklearn.metrics import mean_squared_error
y_pred = lr.predict(X_test)
print(mean_squared_error(y_test, y_pred))


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
ohe = make_column_transformer(
    (OneHotEncoder(sparse=False), X_train.columns[:6]),
    remainder='passthrough')

X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe = ohe.transform(X_test)

X_train.shape

X_train_ohe.shape


lr = LinearRegression().fit(X_train_ohe, y_train)

print(lr.score(X_train_ohe, y_train))

print(lr.score(X_test_ohe, y_test))

from sklearn.metrics import mean_squared_error
y_pred = lr.predict(X_test_ohe)
