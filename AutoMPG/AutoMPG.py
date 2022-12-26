
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import stats
from scipy.stats import norm,skew


from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# XGBoost
import xgboost as xgb
from xgboost import XGBRegressor

#warning
import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)


### DATASET UPLOAD

columnname = ["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","Origin"]
df = pd.read_csv("Python/auto-mpg.data-original", names = columnname , comment="\t", sep=" ", skipinitialspace = True )



def check_data(dataFrame,head=5):
    print("################# SHAPE ##############")
    print(dataFrame.shape)
    print("########### Types ##################")
    print(dataFrame.dtypes)
    print("################## Head ###############")
    print(dataFrame.head(head))
    print("################## Tail ###############")
    print(dataFrame.tail(head))
    print("#################### Quantiles ###############")
    print(dataFrame.describe([0,0.05,0.25,0.5,0.75,0.95,0.99,1]).T)
    print("#################### INFO ###############")
    print(dataFrame.info)
    print("#################### NA ###############")
    print(dataFrame.isnull().sum())

check_data(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen data framee.
    cat_th: int, float
    numerik fakat kategorik olan değişkenler için sınıf eşif değeri
    car_th: int, float
    kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Lategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_card:
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_card  cat_cols'un içinde
    Return olan 3 liste toplamı toplam değişken sayısına eşittir

    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype in ["category","object","bool"]]
    num_cat_cols = [col for col in dataframe.columns if dataframe[col].dtype in ["int64","int32","float64","float32"]
                    and dataframe[col].nunique() < cat_th]
    cat_cols += num_cat_cols

    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtype in ["category","object"]
                   and dataframe[col].nunique() > car_th]

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ["int64","float64"]
                and dataframe[col].nunique() > cat_th]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_cat_cols: {len(num_cat_cols)}")

    return cat_cols,num_cols,cat_but_car


cat_cols,num_cols,cat_but_car = grab_col_names(df)

# 2 categorical variables
# 6 numeric variable

def cat_summary(dataframe, col_name,plot=False):
    if dataframe[col_name].dtype == 'bool':
        dataframe[col_name] = dataframe[col_name].astype(int)

    print(pd.DataFrame({col_name:df[col_name].value_counts(),
                        "Ratio":100*df[col_name].value_counts()/len(dataframe)}))
    print("###################################")

    if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)

cat_summary(df,"Cylinders",plot=True)
cat_summary(df,"Origin",plot=True)

# visualization categorical variables



# 8 missing value in MPG , 6 missing value in Horsepower
# MPG Feature -> Mean = 23.51  Median = 23 , positive skewness



sns.distplot(df.MPG)
plt.show()

sns.distplot(df.Cylinders)
plt.show()

sns.distplot(df.Displacement)
plt.show()

sns.distplot(df.Horsepower)
plt.show()

sns.distplot(df.Weight)
plt.show()

sns.distplot(df.Acceleration)
plt.show()

sns.distplot(df["Model Year"])
plt.show()

sns.distplot(df.Origin)
plt.show()


plt.scatter(df.MPG,df.Weight)
plt.title("MPG - Weight")
plt.xlabel('MPG')
plt.ylabel('Weight')
plt.show()

### MISSING VALUE
df.isna().sum()
df.describe()

df["Horsepower"] = df["Horsepower"].fillna(df["Horsepower"].mean())
df.MPG = df.MPG.fillna(df.MPG.mean())
df.isna().sum()
# Imputation is applied for missing values
df.describe()


### EDA

corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot=True, fmt = ".2f")
plt.show()
# CORR MATRIX


sns.set(rc={'figure.figsize': (10, 10)})
sns.heatmap(corr_matrix, cmap="RdBu")
plt.show()
# CORR MATRIX (HEATMAP)


sns.pairplot(df, diag_kind="kde")
plt.show()

# Origin and Cylinders features can be categorial features.

#boxplot

for i in df.columns:
    plt.figure()
    sns.boxplot(x = i , data = df,)
    plt.show()


# Acceleration has outliers.
# Horsepower has outliers.



### OUTLIER


# Threshold setting function
def outlier_thresholds(dataframe,col_name,q1=0.25,q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5*iqr
    up_limit = quartile3 + 1.5*iqr
    return low_limit,up_limit




# Is there an outlier or not?
def check_outlier(dataframe,col_name):
    low,up = outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name]>up)].any(axis=None):
        return True
    else:
        return False

check_outlier(df,"MPG")            # there is an outlier
check_outlier(df,"Cylinders")      # no outlier
check_outlier(df,"Displacement")   # no outlier
check_outlier(df,"Horsepower")     # there is an outlier
check_outlier(df,"Weight")         # no outlier
check_outlier(df,"Acceleration")   # there is an outlier
check_outlier(df,"Model Year")     # no outlier
check_outlier(df,"Origin")         # no outlier


## Looped version of above codes
for col in num_cols:
    print(col,":",check_outlier(df,col))

for col in cat_cols:
    print(col,":",check_outlier(df,col))





# Which observations outlier ?
def grab_outliers(dataframe,col,index=False):
    low,up = outlier_thresholds(dataframe,col)
    if not dataframe[(dataframe[col] < low) | (dataframe[col] > up)].any(axis=None):
        print("There is no outlier")
        return 0
    if dataframe[(dataframe[col] < low) | (dataframe[col] > up)].shape[0] > 10:
        print(dataframe[(dataframe[col] < low) | (dataframe[col] > up)].head())
    else:
        print(dataframe[(dataframe[col] < low) | (dataframe[col] > up)])
    if index:
        return dataframe[(dataframe[col] < low) | (dataframe[col] > up)].index


grab_outliers(df,"MPG")
grab_outliers(df,"Horsepower")
grab_outliers(df,"Acceleration")



# Removing outliers
def remove_outlier(dataframe,col):
    low,up = outlier_thresholds(dataframe,col)
    df_without_outliers = dataframe[~((df[col] < low) | (df[col]>up))]
    return df_without_outliers


df = remove_outlier(df,"MPG")           # Remove MPG outliers
df = remove_outlier(df,"Horsepower")    # Remove Horepower outliers
df = remove_outlier(df,"Acceleration")  # Remove Acceleration outliers

###### Feature Engineering

### Skewness


## Target Feature = MPG

sns.distplot(df.MPG, fit=norm)
plt.show()

(m,sigma) = norm.fit(df["MPG"])
print("m: {}, sigma: {}".format(m,sigma))

# qq plot
plt.figure()
stats.probplot(df["MPG"], plot=plt)
plt.show()

# MPG -> log transformation for skewness

df["MPG"] = np.log1p(df["MPG"])
sns.distplot(df.MPG, fit=norm)
plt.show()


(m,sigma) = norm.fit(df["MPG"])
print("m: {}, sigma: {}".format(m,sigma))

plt.figure()
stats.probplot(df["MPG"], plot=plt)
plt.show()

## Feature = Independent Variables

# Box-cox Transformation for skewness

skewed_feats = df.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

### One hot encoding

df.Cylinders = df.Cylinders.astype(str)
df.Origin = df.Origin.astype(str)

df = pd.get_dummies(df)


###### Train - Test - Split

x = df.drop(["MPG"],axis=1)
y = df["MPG"]

test_size=0.3
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=test_size, random_state = 35)

#Standardization

#scaler = StandardScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

## Linear Regression
linear_regression_model = LinearRegression().fit(x_train,y_train)
linear_regression_model.intercept_ # b0
linear_regression_model.coef_ # b1,b2...

y_pred = linear_regression_model.predict(x_test)

# ERROR VALUES
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)

# R squared R^2
linear_regression_model.score(x_train,y_train)


## Ridge Regression

ridge = Ridge(random_state = 35 , max_iter=10000)

alphas = np.logspace(-4,-0.5,30)

tuned_parameters = [{"alpha":alphas}]

n_folds = 5

ridge_model = GridSearchCV(ridge,
                   tuned_parameters,
                   cv = n_folds,
                   scoring="neg_mean_squared_error",
                   refit=True)

ridge_model.fit(x_train,y_train)

scores = ridge_model.cv_results_["mean_test_score"]
scores_std = ridge_model.cv_results_["std_test_score"]

print("Ridge Coef:",ridge_model.best_estimator_.coef_)
print("-----------------------------------------------")
print("Ridge Best Estimator:",ridge_model.best_estimator_)

y_pred = ridge_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
print("Ridge MSE:",mse)

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")
plt.show()


## Lasso Regression

lasso = Lasso(random_state=35 , max_iter=10000)
alphas = np.logspace(-4,-0.5,30)

tuned_parameters=[{"alpha":alphas}]
n_folds=5

lasso_model = GridSearchCV(lasso,
                   tuned_parameters,
                   cv=n_folds,
                   scoring="neg_mean_squared_error",
                   refit=True)

lasso_model.fit(x_train,y_train)
scores = lasso_model.cv_results_["mean_test_score"]
scores_std = lasso_model.cv_results_["std_test_score"]


print("Lasso Coef:",lasso_model.best_estimator_.coef_)
# Lasso assigns 0 to unnecessary features
print("-----------------------------------------------")
print("Lasso Best Estimator:",lasso_model.best_estimator_)

y_pred = lasso_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)

print("Lasso MSE:",mse)

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")
plt.show()


## ElasticNet

elasticnet = ElasticNet(random_state=35 , max_iter=10000)

alphas = np.logspace(-4,-0.5,30)
l1_ratio = np.arange(0,1,0.05)


parametersGrid = {"alpha":alphas,
                  "l1_ratio":l1_ratio}

n_folds=5

elasticnet_model = GridSearchCV(elasticnet,
                                parametersGrid,
                                cv=n_folds,
                                scoring="neg_mean_squared_error",
                                refit=True)

elasticnet_model.fit(x_train,y_train)

scores = elasticnet_model.cv_results_["mean_test_score"]
scores_std = lasso_model.cv_results_["std_test_score"]


print("ElasticNet Coef:",elasticnet_model.best_estimator_.coef_)
print("-----------------------------------------------")
print("ElasticNet Best Estimator:",elasticnet_model.best_estimator_)

y_pred = elasticnet_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)

print("ElasticNet MSE:",mse)


##############

    # StandartScaler
#   Linear Regression MSE = 0.013374975893501175
#   Ridge Regression MSE =  0.013344989776439942
#   Lasso Regression MSE =  0.014007689055519896
#   ElasticNet Reg. MSE  =  0.01405818214272070


    # RobustScaler
#   Linear Regression MSE = 0.013374975893501171
#   Ridge Regression MSE =  0.01330880188768746
#   Lasso Regression MSE =  0.014073139893940408
#   ElasticNet Reg. MSE  =  0.014112849571050995

    # continue with robustscaler

## XGBOOST

xgb= xgb.XGBRegressor()

parametersGrid = {"nthread":[4],
                  "objective":["reg:linear"],
                  "learning_rate":[0.03,0.05,0.07],
                  "max_depth":[5,6,7],
                  "min_child_weight":[4],
                  "subsample":[0.7],
                  "colsample_bytree":[0.7],
                  "n_estimators":[500,1000]}

xgboost_model = GridSearchCV(xgb,
                             parametersGrid,
                             cv=n_folds,
                             scoring="neg_mean_squared_error",
                             refit=True,
                             n_jobs=5)

xgboost_model.fit(x_train,y_train)

y_pred = xgboost_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)

print("XGBRegressor MSE:" , mse)

# XGBRegressor MSE: 0.01544511963741259


### Averaging Model

class AveragingModels():
    def __init__(self,models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self,x,y):
        self.models_ = [clone(x) for x in self.models]

        # train cloned base models
        for model in self.models_:
            model.fit(x,y)

        return self
    # now we do the predictions cloned models and average them
    def predict(self,x):
        predictions = np.column_stack([model.predict(x) for model in self.models_])
        return np.mean(predictions, axis=1)

averaged_models = AveragingModels(models=(xgboost_model,ridge_model))
averaged_models.fit(x_train,y_train)

y_pred = averaged_models.predict(x_test)
mse = mean_squared_error(y_test,y_pred)

print("Averaged Models MSE:" , mse)



##### Averaged Models MSE: 0.012344561484256522




    # StandartScaler
#   Linear Regression MSE = 0.013374975893501175
#   Ridge Regression MSE =  0.013344989776439942
#   Lasso Regression MSE =  0.014007689055519896
#   ElasticNet Reg. MSE  =  0.01405818214272070


    # RobustScaler
#   Linear Regression MSE = 0.013374975893501171
#   Ridge Regression MSE =  0.01330880188768746
#   Lasso Regression MSE =  0.014073139893940408
#   ElasticNet Reg. MSE  =  0.014112849571050995
#   Averaged Models MSE =   0.012344561484256522
