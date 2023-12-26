import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean, median, stdev

# Regression Models
# from sklearn.linear_model import LinearRegression, Lasso
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR

# Classification Models
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

import warnings
warnings.filterwarnings("ignore")


# MLR, 70:30, 2-features, ZM


def get_data(file_name, n_features):
    x_data = pd.read_csv('Paper_Data/{}.csv'.format(file_name)).drop(['Second', 'Repetition'], axis=1)
    y_data = x_data.pop('FAS')

    s = pd.Series(index=x_data.drop(['Subject'], axis=1).columns,
                  data=RandomForestRegressor(random_state=50).fit(x_data.drop(['Subject'], axis=1),
                                                                  y_data).feature_importances_).sort_values(
        ascending=False)

    # The sorted list is used to obtain only the best "n_features" according to the integer selected.
    x_data = x_data.loc[:, list(s.index[:n_features]) + ['Subject']]

    return x_data, y_data


w_csv = False
if w_csv:
    x, y = get_data('EEG_Z39M69N100_R', 10)
    df = x.copy()
    map_function = lambda x, k: (1 if (x >= 22) else 0) if k == 2 else (1 if (35 > x >= 22) else 0 if x < 22 else 2)
    df['FAS'] = list(y)
    df['FAS_2'] = [map_function(x, 2) for x in y]
    df['FAS_3'] = [map_function(x, 3) for x in y]
    df.to_csv('Paper_Data/small_df_2.csv', index=False)

df = pd.read_csv('Paper_Data/small_df.csv')


def feat_imp_latex(create=False):

    if create:
        x_data = pd.read_csv('Paper_Data/no_reference/EEG_Z39M69N100.csv').drop(['Second', 'Repetition', 'Subject'], axis=1)
        y_data = x_data.pop('FAS')

        # x_data = x_data.drop([x for x in x_data.columns if sum([x.find(y) for y in ['-L', '-M-', '-I']]) != -3], axis=1)

        s = pd.Series(index=x_data.columns,
                      data=RandomForestRegressor(random_state=50).fit(x_data, y_data).feature_importances_).sort_values(
            ascending=False)

        s.head(10).to_csv('Paper_Data/feat_imp10.csv')

    # Pandas DataFrame
    s = pd.read_csv('Paper_Data/feat_imp10.csv', index_col=0)
    latex_trans= lambda x : '\\' + x.split('_')[0].lower() + '_{' + x.split('_')[1] + '}'

    # Feature & GI
    for i, feature in enumerate(list(s.index)):
        # print([feature.split('-D-')[0], feature.split('-D-')[1], round(s.iloc[i, 0], 4)])
        print('${}$/${}$ & ${}$ \\\\'.format(latex_trans(feature.split('-D-')[0]), latex_trans(feature.split('-D-')[1]),
                                        round(s.iloc[i, 0], 4)))

    exit()


# feat_imp_latex(False)
# exit()
# df.loc[df.Subject == 2, ['FAS_2', 'FAS_3']] = 0, 0

# k = 2
# sns.scatterplot(data=df, x=df.columns[0], y=df.columns[2], hue='FAS_{}'.format(k))
# plt.show()
# exit()


def mean_df(df):
    df_2 = pd.DataFrame(columns=df.columns)

    for subject in df.Subject.unique():
        df_sub = df.loc[df.Subject == subject, :]
        df_2 = pd.concat([df_2, pd.DataFrame(df_sub.apply(mean, axis=0)).T], axis=0)

    return df_2


def create_plot(df, k, ap_mean=False):
    if ap_mean:
        df = mean_df(df)

    df['FAS_2'] = pd.Categorical(df.FAS_2.map({0: 'No Fatigue',
                                               1: 'Fatigue'}), categories=['No Fatigue', 'Fatigue'], ordered=True)
    df['FAS_3'] = pd.Categorical(df.FAS_3.map({0: 'No Fatigue',
                                               1: 'Fatigue',
                                               2: 'Substancial Fatigue'}), categories=['No Fatigue', 'Fatigue', 'Substancial Fatigue'], ordered=True)

    fig = plt.figure()
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue='FAS_{}'.format(k))
    ax = plt.gca()
    ax.set_xlim([-0.2, 5])
    ax.set_ylim([-0.2, 5])
    mean_name = 'M' if mean else 'N'

    # fig.savefig('Paper_Data/Scatter_{}_{}.pdf'.format(k, mean_name), bbox_inches='tight')


# create_plot(df, 2)
# plt.show()
# exit()
# create_plot(df, 3)
# create_plot(df, 2, True)
# create_plot(df, 3, True)
# plt.show()
# exit()

# validation_scheme = '80:20'
n_fold = 10
n_class = 2

from random import seed, sample

def divide_data(validation_name, x_data, y_data, n_seed):
    ids = x_data.Subject.unique()
    n = len(ids)
    train_ids = ids
    n_test = int(validation_name[3:])

    seed(n_seed)
    test_ids = sample(set(train_ids), int(n*n_test/100))
    train_ids = list(set(train_ids) - set(test_ids))

    train_index = x_data[x_data.Subject.isin(train_ids)].index
    test_index = x_data[x_data.Subject.isin(test_ids)].index

    x_train_f, x_test_f = x_data.iloc[train_index, :], x_data.iloc[test_index, :]

    def get_index(s):
        """
        :param s: A list with subject ids and their index numbers
        :return: [(index_start, index_end), ...] for n distinct subjects
        """
        l = []
        for distinct_id in s.unique():
            s_sub = s[s == distinct_id]
            l.append((s_sub.index[0], s_sub.index[-1]))
        return l

    train_sub, test_sub = get_index(x_train_f.pop('Subject')), get_index(x_test_f.pop('Subject'))
    y_train_f, y_test_f = y_data[train_index], y_data[test_index]

    return x_train_f, y_train_f, x_test_f, y_test_f, test_sub




def create_model(model_name, train_x, train_y):

    # Regression models
    if model_name == 'SVR':
        return LinearSVR(random_state=20).fit(train_x.iloc[:, :2], train_x.iloc[:, 2])

    train_x = train_x.drop('FAS', axis=1)

    # Classification models
    if model_name == 'SVC':
        return LinearSVC(C=1.5, random_state=20).fit(train_x, train_y)
    elif model_name == 'LOG':
        return LogisticRegression(max_iter=500, random_state=20).fit(train_x, train_y)
    elif model_name == 'RF':
        return RandomForestClassifier(random_state=20).fit(train_x, train_y)
    elif model_name == 'KNN':
        return KNeighborsClassifier(3).fit(train_x, train_y)
    elif model_name == 'CART':
        return DecisionTreeClassifier(random_state=20).fit(train_x, train_y)

df = mean_df(df)
y = pd.Categorical(df['FAS_2'])
x = df[[df.columns[0], df.columns[1], 'FAS']]

df_fold = pd.DataFrame(columns=['N_Fold', 'Accuracy', 'F1-Score', 'Precision', 'Recall'])
df_fold_sd = pd.DataFrame(columns=['N_Fold', 'Accuracy', 'F1-Score', 'Precision', 'Recall'])
map_function = lambda x: (1 if (x >= 22) else 0)

for n_fold in range(5, 11):
    for rand_seed in range(100):
        skf = StratifiedKFold(n_fold, shuffle=True, random_state=rand_seed)

        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        df_results = pd.DataFrame(columns=['Accuracy', 'F1-Score', 'Precision', 'Recall'])
        df_results_sd = pd.DataFrame(columns=['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall'])

        for model_name in ['SVC']:
            df_cv = pd.DataFrame(columns=metrics)
            # df_sd = pd.DataFrame(columns=metrics)
            n_total = 0

            for i, (train_index, test_index) in enumerate(skf.split(x, y)):
                train_x, test_x = x.iloc[train_index,:], x.iloc[test_index, :]
                train_y, test_y = y[train_index], y[test_index]

                test_x = test_x.drop('FAS', axis=1)

                # print(len(train_y) / (len(train_y) + len(test_y)))

                model = create_model(model_name, train_x, train_y)
                pred_y = model.predict(test_x)

                if model_name == 'SVR':
                    pred_y = [map_function(x) for x in pred_y]

                report = classification_report(test_y, pred_y, output_dict=True)
                l_performance = [accuracy_score(test_y, pred_y), report['macro avg']['f1-score'],
                                 report['macro avg']['precision'], report['macro avg']['recall']]

                df_cv = pd.concat([df_cv, pd.DataFrame(dict(zip(metrics, l_performance)), index=[df_cv.shape[0]])], axis=0)
                n_total +=1

            # print(df_cv)
            # exit()
            print(n_total)
            print(df_cv)
            exit()

            df_sd = pd.DataFrame(df_cv.apply(stdev, axis=0)).T
            # df_sd['Model'] = model_name
            df_sd.index = [df_results_sd.shape[0]]

            df_cv = pd.DataFrame(df_cv.apply(mean, axis=0)).T
            # df_cv['Model'] = model_name
            df_cv.index = [df_results.shape[0]]

            df_results = pd.concat([df_results, df_cv], axis=0)
            df_results_sd = pd.concat([df_results_sd, df_sd], axis=0)

    df_results = pd.DataFrame(df_results.apply(mean, axis=0)).T
    df_results['N_Fold'] = n_fold

    df_results_sd = pd.DataFrame(df_results_sd.apply(mean, axis=0)).T
    df_results_sd['N_Fold'] = n_fold

    df_fold = pd.concat([df_fold, df_results], axis=0)
    df_fold_sd = pd.concat([df_fold_sd, df_results_sd], axis=0)

df_fold['Meaning'] = 'Mean'
df_fold_sd['Meaning'] = 'SD'
df_fold = pd.concat([df_fold, df_fold_sd], axis=0)
print(df_fold)

# df_fold.to_csv('Paper_Data/robust_acc.csv', index=False)
# exit()


def svm_plot(X, y, type='2D'):
    from sklearn.svm import SVC
    import numpy as np
    import matplotlib.pyplot as plt
    # from sklearn import svm

    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    clf = create_model('SVC', X, y)
    print(accuracy_score(y, clf.predict(X)))

    if type == '2D':
        fig, ax = plt.subplots()
        # title for the plots
        title = ('Decision surface of linear SVC ')
        # Set-up grid for plotting.
        # print(X)
        # print(X.iloc[:, :1].columns)
        X0, X1 = X.iloc[:, 0], X.iloc[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        print(clf.coef_)
        print(clf.intercept_)
        # y = y.map({0: 'No Fatigue', 1: 'Substancial Fatigue'})

        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.6)
        l_fas = ['No Fatigue', 'Substantial Fatigue']
        col_fas = ['tab:blue', 'tab:red']

        y = pd.Series(y)

        for fas in y.unique():
            idx = y[y == fas].index
            ax.scatter(X.iloc[idx, 0], X.iloc[idx, 1], c=col_fas[int(fas)], cmap=plt.cm.coolwarm, s=30, edgecolors='k', label=l_fas[int(fas)])
        ax.set_ylabel('$\delta_{C3}$/$\delta_{A2}$')
        ax.set_xlabel('$\delta_{A1}$/$\delta_{A2}$')
        # ax.set_xticks(())
        # ax.set_yticks(())
        ax.legend(loc='upper center')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        # plt.show()

        plt.savefig('Paper_Data/svmVIS.pdf', bbox_inches='tight')


    elif type == '2D2':
        # clf = LinearSVC(C=C, loss="hinge", random_state=42, dual="auto").fit(X, y)
        # obtain the support vectors through the decision function
        decision_function = clf.decision_function(X)
        # we can also calculate the decision function manually
        # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
        # The support vectors are the samples that lie within the margin
        # boundaries, whose size is conventionally constrained to 1
        support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
        support_vectors = X[support_vector_indices]

        plt.subplot(1, 2, i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(clf, X, ax=ax, grid_resolution=50, plot_method="contour", colors="k",
                                               levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, linewidth=1, facecolors="none", edgecolors="k")

    elif type == '3D':
        # make it binary classification problem
        X = np.array(X)
        y = np.array(y)

        X = X[np.logical_or(y == 0, y == 1)]
        Y = y[np.logical_or(y == 0, y == 1)]

        # print(X)
        # print(y)

        # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
        # Solve for w3 (z)
        z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

        tmp = np.linspace(0, 2, 30)
        x, y = np.meshgrid(tmp, tmp)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], 'ob')
        ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'sr')
        ax.plot_surface(x, y, z(x, y))
        ax.view_init(30, 60)
        plt.show()


svm_plot(x.reset_index(drop=True).drop('FAS', axis=1), y, '2D')
