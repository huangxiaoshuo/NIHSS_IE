import os
import sys
sys.path.append('..')
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from main.configs import base
import numpy as np

class Evaluation(object):
    def __init__(self, config):
        self.random_seed = config.random_seed
        self.output_path = config.output_path
        self.folds = config.folds
        self.init_model()

    def init_model(self):
        self.models = {
            'dt': DecisionTreeClassifier(random_state=self.random_seed),
            'rf': RandomForestClassifier(random_state=self.random_seed),
            'svm-rbf': SVC(kernel='rbf', random_state=self.random_seed)
        }
    
    def full_evaluation(self, args, X, y, test_X, test_y, test_R_FN_y):
        os.makedirs(self.output_path, exist_ok=True)
        X, y = shuffle(X, y, random_state = self.random_seed)
        y = y.reshape((X.shape[0],)) # 1d array was expected, from (4045,1) to (4045,)
        test_y = test_y.reshape((test_X.shape[0],))
        self.evaluation(args, X, y, test_X, test_y, test_R_FN_y)
        pass

    def evaluation(self, args, X, y, test_X, test_y, test_R_FN_y):

        test_y = np.append(test_y, np.array(test_R_FN_y))
        for classifier_name, model in self.models.items():
            print('Classifier: ', classifier_name)
            model.fit(X, y)
            prediction = model.predict(test_X)
            prediction = np.append(prediction, np.array([0]*len(test_R_FN_y)))

            print('accuracy: ', accuracy_score(test_y, prediction))
            print('precison: ', precision_score(test_y, prediction))
            print('recall:', recall_score(test_y, prediction))
            print('f1 score: ', f1_score(test_y, prediction))
            test_predict_y_path = base.config['data_dir'] / f'End2End_Test/cached/{args.source}_predict_y_{classifier_name}.txt'
            if not test_predict_y_path.exists():
                np.savetxt(test_predict_y_path, prediction, fmt='%d')