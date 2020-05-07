# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt

# Import dataset
df = pd.read_excel('/Users/davidsousa/Documents/SportsDS/datasets/players_clt.xlsx')

# Excluding outliers
df['Outlier'] = 0
df.loc[df['fouls_per_game'] > 5.9, 'Outlier'] = 1 # this nonstandard behavior is due to the player possessing only 1 match
df = df.loc[df['Outlier'] == 0]
df.drop(columns='Outlier', inplace=True)

# Select variables
X = df[['clearances_per_game', 'interceptions_per_game', 'duelswon_per_game',  'blocks_per_game',
        'fouls_per_game', 'completepasses_per_game', 'smartpasses_per_game', 'shots_per_game', 'crosses_per_game']]

y = df['Role']

# Split into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                    random_state=15, shuffle=True, stratify=y)

# Data standardization
scaler = StandardScaler().fit(X_train)
scaler_X_train = pd.DataFrame(scaler.transform(X_train))
scaler_X_val = pd.DataFrame(scaler.transform(X_val))

# Neural Network Model
model = MLPClassifier(random_state= 15, max_iter=500)

# Definition of the parameter space
parameter_space = {
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': list(np.logspace(-5, 3, 5)),
    'learning_rate_init': list(np.linspace(0.00001, 0.1, 5)),
    'warm_start': [True, False],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'early_stopping': [True, False]}

# Defining the grid search in which the hyper parameter tuning will be conducted
clf = GridSearchCV(model, parameter_space, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=21),
                   scoring='accuracy', verbose=1, n_jobs=-1)
clf.fit(scaler_X_train, y_train)

# Find best parameters
clf.best_params_

def NN_structure(min_neurons,max_neurons,n_layers):
    l = []
    for i in range(min_neurons,max_neurons+1):
        l.append(i)
    nnstructure = list(combinations_with_replacement(l,n_layers))
    return nnstructure

# Maintaining the previous parameters fixed, find NN architecture
model = MLPClassifier(random_state=15, max_iter=500, activation='tanh', solver='adam',
                      alpha=1e-05, learning_rate='constant', learning_rate_init=0.0750025, warm_start=True)

parameter_space = {
    'hidden_layer_sizes': NN_structure(10, 50, 2)
}

clf = GridSearchCV(model, parameter_space, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=21),
                   scoring='accuracy', verbose=1, n_jobs=-1)
clf.fit(scaler_X_train, y_train)
clf.best_params_
clf.best_score_

# Predictions
labels_train = clf.predict(scaler_X_train)
labels_val = clf.predict(scaler_X_val)

# Visually represent the results
def metrics(y_train, pred_train, y_val, pred_val, target_names):
    """
    Function that creates classification_report and confusion matrix for the training and validation set.

    :param y_train: training set ground truth
    :param pred_train: predicted labels on the training set
    :param y_val: validation set ground truth
    :param pred_val: predicted labels on the validation set
    :param target_names: role names

    Returns:
        - Classification report and confusion matrix
    """
    print('___________________________________________________________________________________________________________')
    print('                                                     TRAIN                                                 ')
    print('-----------------------------------------------------------------------------------------------------------')
    print(classification_report(y_train, pred_train, target_names=target_names))
    print(confusion_matrix(y_train, pred_train))

    print('___________________________________________________________________________________________________________')
    print('                                                VALIDATION                                                 ')
    print('-----------------------------------------------------------------------------------------------------------')
    print(classification_report(y_val, pred_val, target_names=target_names))
    print(confusion_matrix(y_val, pred_val))


metrics(y_train, labels_train, y_val, y_val, np.unique(y_train))

def plot_cm(confusion_matrix: np.array, class_names: list):
    """
    Function that creates a confusion matrix plot.

    :param confusion_matrix: confusion matrix that will be plotted
    :param class_names: labels of the classes

    Returns:
        - Plot of the Confusion Matrix
    """

    fig, ax = plt.subplots()
    plt.imshow(confusion_matrix, cmap=plt.get_cmap('cividis'))
    plt.colorbar()

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    plt.ylabel('Targets', fontweight='bold')
    plt.xlabel('Predictions', fontweight='bold')
    plt.ylim(top=len(class_names) - 0.5)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-0.5)  # adjust the bottom leaving top unchanged
    plt.tight_layout()
    return plt.show()

plot_cm(confusion_matrix(y_val, labels_val), np.unique(y_val))