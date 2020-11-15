import sys
sys.path.append('..')
from argparse import ArgumentParser
from pathlib import Path
from main.common import tools
from main.common.tools import logger
from main.configs.base import config
from main.svm.data import Data
from main.svm.configuration import Configuration
from main.svm.features import Features
from main.svm.evaluation import Evaluation
from main.configs import base
import numpy as np

def main():
    parser = ArgumentParser()
    args = parser.parse_args()
    args.do_train = False
    args.do_test = False
    args.do_e2e_test = True

    if args.do_train:
        for fold_i in range(0,5):
            args.fold = fold_i
            run_train(args)

    # for golden standard, select the best model to test , Fold-2 and report Feature distance and (measure + measureRange + D)
    if args.do_test:
        args.fold = 2
        args.test_data_path = config['data_dir'] / f'RE_Test/re_test.pkl'
        args.test_features_cached_file = config['data_dir'] / f'RE_Test/cached/'
        run_test(args)
        pass

    # should one-pass
    if args.do_e2e_test:
        args.fold = 2
        # for bert model e2e test
        args.source = 'bert'
        args.test_data_path = config['data_dir'] / 'End2End_Test' / f're_e2e_test_by_fold_3.pkl'
        args.test_features_cached_file = config['data_dir'] / f'End2End_Test/cached/test_fold_3_features/'
        # for CRF model e2etest
        #args.source = 'crf'
        #args.test_data_path = config['data_dir'] / 'End2End_Test' / f'end2end_test_prediction_result_9_24.pkl'
        #args.test_features_cached_file = config['data_dir'] / f'End2End_Test/cached/end2end_test_prediction_result_9_24/'
        run_test(args)

def run_train(args):
    tools.init_logger(log_file = config['log_dir'] / 'svm_classifier.log')
    logger.info('**********Start Training*********')
    logger.info('Loading configuration.')
    re_config = Configuration()
    
    logger.info('Loading train data.')
    train_data_path = config['data_dir'] / f'RE_Train/re_train_fold_{args.fold}.pkl'
    train_data = Data(train_data_path)

    logger.info('Creating or loading features of train data')
    train_features_cached_file = config['data_dir'] / f'RE_Train/cached/cached_re_train_fold_{args.fold}_features'
    features = Features(re_config)
    features.get_feature_vetor(train_data.data, train_features_cached_file)
    train_X = features.X
    train_y = features.y

    logger.info('Loading valid data.')
    valid_data_path = config['data_dir'] / f'RE_Train/re_valid_fold_{args.fold}.pkl'
    valid_data = Data(valid_data_path)
    
    logger.info('Creating or loading features of valid data')
    valid_features_cached_file = config['data_dir'] / f'RE_Train/cached/cached_re_valid_fold_{args.fold}_features'
    features.get_feature_vetor(valid_data.data, valid_features_cached_file)
    valid_X = features.X
    valid_y = features.y

    model_eval = Evaluation(re_config)
    model_eval.full_evaluation(train_X, train_y, valid_X, valid_y)

def run_test(args):
    tools.init_logger(log_file = config['log_dir'] / 'svm_classifier_test.log')
    logger.info('**********Start Testing*********')
    logger.info('Loading configuration.')
    re_config = Configuration()
    
    logger.info('Loading train data.')
    train_data_path = config['data_dir'] / f'RE_Train/re_train_fold_{args.fold}.pkl'
    train_data = Data(train_data_path)

    logger.info('Creating or loading features of train data')
    train_features_cached_file = config['data_dir'] / f'RE_Train/cached/cached_re_train_fold_{args.fold}_features'
    features = Features(re_config)
    features.get_feature_vetor(train_data.data, train_features_cached_file)
    train_X = features.X
    train_y = features.y

    train_X_features_with_hadmid_path = base.config['data_dir'] / f'RE_Train/cached/cached_re_train_fold_{args.fold}_features/train_X_all_features.txt'
    if not train_X_features_with_hadmid_path.exists():
       np.savetxt(train_X_features_with_hadmid_path, features._X, fmt='%d')


    logger.info('Loading test data.')
    test_data_path = args.test_data_path
    test_data = Data(test_data_path)
    
    logger.info('Creating or loading features of test data')
    test_features_cached_file = args.test_features_cached_file
    features.get_feature_vetor(test_data.data, test_features_cached_file)
    test_X = features.X
    test_y = features.y

    test_R_FN_y = len(features.R_FN_X) * [1]

    test_X_features_with_hadmid_path = base.config['data_dir'] / f'End2End_Test/cached/X_all_features_{args.source}.txt'
    if not test_X_features_with_hadmid_path.exists():
       np.savetxt(test_X_features_with_hadmid_path, features._X, fmt='%d')

    model_eval = Evaluation(re_config)
    model_eval.full_evaluation(args, train_X, train_y, test_X, test_y, test_R_FN_y)

if __name__ == "__main__":
    main()