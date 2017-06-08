import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def explore(filepath):
    '''
    Read file into df, explore and return df
    INPUT: filepath - a string of the file path
    OUTPUT: a dataframe
    '''
    df = pd.read_csv(filepath)
    print '*********** Shape of df **************'
    print df.shape

    features = df.columns.values
    print '********** Number of features ***************'
    print len(features)
    print '********** Features ***************'
    print features

    print '******* Head of df ******************'
    print df.head(2)
    print '******* Info of df ******************'
    print df.info()

    get_unique(df)
    get_missing(df)

    print '******** Description of df *****************'
    print df.describe()
    return df

def get_unique(df):
    '''
    Print out the number of unique values for each columns
    '''
    features = df.columns.values
    print '********* Number of unique values **********'
    for feature in features:
        print feature,' ', len(df[feature].unique())
    return

def get_missing(df):
    '''
    Print out the number of missing values for each columns
    '''
    print '********* Number of missing values **********'
    print df.isnull().sum()
    print 'Total missing: ', df.isnull().sum().sum()
    return

def category_to_numer_dict(df, category, values):
    '''
    Convert a categorical feature to numeric by replace the catgorical values with integers and
    save the integer to categorical value pair in a dictionary for reference.
    INPUT: df - a dataframe which will be modified by the side effect of the function,
           category - the categorical feature to be converted
           values - a list of categorical values, which will be assigned to integer in the same order presented
    OUTPUT: a dictionary of integer to categoricl value pairs, the function does not return a dataframe!
    '''
    dict = defaultdict(str)
    for i,value in enumerate(values):
        dict[i] = value
        df.ix[df[category]==value, category] = i
    df[category] = df[category].astype(int)
    return dict

def get_percentage(df, target_column, target_value, relationship='eq'):
    '''
    Get the percentage of rows that satisfy the condition that the value of the target_column fits the relationship to the target_value
    e.g., get_percentage(df, 'A', 1, relationship='eq') returns percent of rows that its coulmn 'A' value equals 1
    INPUT: df - a dataframe, target_column - the target column name in string,
           target_value - target value in int or float, relationship - the relationship of column values
           to the target value (string, 'eq' - equal, 'leq' - less or equal than, 'geq' - greater or equal than)
    OUTPUT: count - number of rows satisfying the relationship (int),
            percent - percent of rows satisfying the relationship (float)
    '''
    if relationship == 'eq':
        cond = df[target_column]==target_value
    elif relationship =='leq':
        cond = df[target_column]<=target_value
    elif relationship =='geq':
        cond = df[target_column]>=target_value
    else:
        print 'Wrong value of condition!'
        return
    count = len(df[cond])
    total = len(df)
    percent = count*1.0/total
    return count, percent

def show_histogram(df,columns,ylabel='Number of emplyees'):
    '''
    Plots histogram of numerical data
    INPUT: pandas DataFrame, columns - name of columns in list of strings
    OUTPUT: None
    '''
    for column in columns:
        df[column].hist(bins=25)
        title = ylabel + ' vs. '+  column
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(column)
        plt.show()
    return

def plot_categories(df,columns,ylabel='Number of emplyees'):
    '''
    Plots bar chart of categorical data
    INPUT: pandas DataFrame, columns - name of columns in list of strings
    OUTPUT: None
    '''
    df_copy = df.copy()
    df_copy['count'] = 1
    for var in columns:
        df_copy.groupby(var).count()['count'].plot(kind = 'bar', figsize = (10,10))
        title = ylabel + ' vs. ' +  var
        plt.xticks(rotation = 'horizontal')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.show()
    return

def scatter_matrix(df, columns, fig_size=(10,10)):
    '''
    Plots scatter_matrix of numeric features
    INPUT: df - pandas DataFrame, columns - name of columns in list of strings
           fig_size - size of figure in tuple
    OUTPUT: None
    '''
    fig, axs = plt.subplots(1, 1, figsize = fig_size)
    axs = pd.tools.plotting.scatter_matrix(df[columns], ax = axs, diagonal = 'kde')

    ''' Rotate axis labels '''
    n = len(columns)
    for i in range(n):
        v = axs[i, 0]
        v.yaxis.label.set_rotation(0)
        v.yaxis.label.set_ha('right')
        h = axs[n-1, i]
        h.xaxis.label.set_rotation(90)
    plt.show()
    return

def dummify(df, columns):
    '''
    Dummifies categorical features
    INPUT: df - pandas Dataframe, columns -  name of columns in list of strings
    OUTPUT: pandas Dataframe with dummifed categorical variables
    '''
    df_out = df.copy()
    df_out = pd.get_dummies(df_out,columns = columns)
    return df_out

def train_vali_split(df, target_col, test_size = 0.2, random_seed = 100):
    '''
    Split the dataframe into training and testing set in forms of X_train, X_test, y_train, y_test
    INPUT: df - a dataframe, target_col - the column name of label in string,
        test_size - the proportion of the data to be used as test set in float
        random_seed - set the seed for random process in int
    OUTPUT: X_train, X_test, y_train, y_test as dataframe and series
    '''
    features = df.columns.tolist()
    del features[features.index(target_col)]

    X = df.ix[:, features]
    y = df[target_col].astype('int')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    return X_train, X_test, y_train, y_test

def log_Reg(X_train, y_train, X_test):
    '''
    Build a logistic regression model
    INPUT: dataframe and series of X_train, y_train, X_test from train_test_split
    OUTPUT: fitted model object and probabilities created by model
    '''
    model = LogisticRegression().fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    return model, probabilities

def random_Forest(X_train, y_train, X_test):
    '''
    Build a random forest classification model
    INPUT: dataframe and series of X_train, y_train, X_test from train_test_split
    OUTPUT: fitted model object and probabilities created by model
    '''
    model = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    return model, probabilities

def KNN(X_train, y_train, X_test, neighbors=3):
    '''
    Build a KNN classification model
    INPUT: dataframe and series of X_train, y_train, X_test from train_test_split, neighbors - number of neighbors in int
    OUTPUT: fitted model object and probabilities created by model
    '''
    neighbor = KNeighborsClassifier(n_neighbors=neighbors)
    neighbor.fit(X_train, y_train)
    probabilities = neighbor.predict_proba(X_test)
    return neighbor, probabilities

def get_classifer_scores(model, X_test, y_test):
    '''
    Get evaluation metrics of the model
    INPUT: model - a training data fitted model, X_test, y_test - dataframe and series from train_test_split
    OUTPUT: accuracy, recall in float
    '''
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    recall = recall_score(y_test, yhat)
    print 'Accuracy and recall are {:.3f} and {:.3f}'.format(accuracy, recall)
    return accuracy, recall

def plot_feature_importance(model, X_train, max_features=10):
    '''
    Plot feature importance
    INPUT: model - a training data fitted model, X_train - dataframe from train_test_split
           max_features - maxmium number of features in int to be ploted
    OUTPUT: feature_importance values normalized to the max_feature in an numpy array
    '''
    feature_importance = model.feature_importances_
    '''make importances relative to max importance'''
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    '''Show only top features'''
    pos = pos[-max_features:]
    feature_importance = (feature_importance[sorted_idx])[-max_features:]
    feature_names = (X_train.columns[sorted_idx])[-max_features:]

    plt.figure(figsize = (12,8))
    plt.barh(pos, feature_importance, align='center', color = 'g')
    plt.yticks(pos, feature_names)
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.show()
    return feature_importance

if __name__ == '__main__':
    filepath = '../data/HR_comma_sep.csv'
    df = explore(filepath)

    '''check if class is imbalanced'''
    left_number, left_percent = get_percentage(df, 'left', 1, relationship='eq')
    print 'Number and ratio of employees left are {:d} and {:.3f}'.format(left_number, left_percent)
    pass

    '''make correction of column names'''
    df=df.rename(columns = {'sales':'department','Work_accident':'work_accident',\
    'average_montly_hours':'average_monthly_hours'})

    '''plot histogram of numeric features'''
    numeric_list = ['satisfaction_level', 'last_evaluation', 'number_project', \
    'average_monthly_hours','time_spend_company', 'work_accident', 'left', 'promotion_last_5years']
    show_histogram(df, numeric_list)

    '''plot histogram of categrical features'''
    categorical_list = ['department','salary']
    plot_categories(df,categorical_list)

    '''plot scatter matrix of numeric features'''
    scatter_matrix(df, numeric_list)

    '''plot histogram and scatter_matrix of numeric and categrical features of left employees'''
    df_left = df[df['left']==1]
    show_histogram(df_left, numeric_list)
    plot_categories(df_left,categorical_list)
    numeric_list_short = ['satisfaction_level', 'last_evaluation', 'number_project', \
    'average_monthly_hours','time_spend_company', 'work_accident', 'promotion_last_5years']
    scatter_matrix(df_left, numeric_list_short)

    '''dummify the department feature'''
    categorical_cols = ['department']
    df = dummify(df,categorical_cols)

    '''convert salary to numeric feature ( 'low', 'medium', and 'high' to 0,1, and 2, )'''
    values = ['low', 'medium', 'high']
    category_to_numer_dict(df, 'salary', values)

    '''train test split'''
    X_train, X_test, y_train, y_test = train_vali_split(df, 'left')

    '''run logistic regression model'''
    logi_model,logi_prob  = log_Reg(X_train, y_train, X_test)
    print '****** Logistic regression model **********'
    accuracy, recall = get_classifer_scores(logi_model, X_test, y_test )
    cross_val_scores_logi = cross_val_score(logi_model, X_train, y_train, cv=5)
    print 'Mean cross validation score is {:.3f}'.format(np.mean(cross_val_scores_logi))

    '''run random forest classification model'''
    rf_model, rf_prob = random_Forest(X_train, y_train, X_test)
    print '****** Random forest classification model **********'
    accuracy, recall = get_classifer_scores(rf_model, X_test, y_test )
    cross_val_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5)
    print 'Mean cross validation score is {:.3f}'.format(np.mean(cross_val_scores_rf))
    plot_feature_importance(rf_model, X_train)

    '''run KNN classification model'''
    knn_model, knn_prob = KNN(X_train, y_train, X_test)
    print '****** KNN classification model **********'
    accuracy, recall = get_classifer_scores(knn_model, X_test, y_test )
    cross_val_scores_knn = cross_val_score(knn_model, X_train, y_train, cv=5)
    print 'Mean cross validation score is {:.3f}'.format(np.mean(cross_val_scores_knn))
