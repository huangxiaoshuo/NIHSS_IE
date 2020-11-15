import sys
sys.path.append('..')
from main.configs import base

class Configuration(object):
    def __init__(self):
        self.folds = 5
        self.random_seed = 2020
        self.output_path = base.config['result_dir'] / 're_classifier'
        self.re_labels = { 'Has_Value': 1, '0': 0 }
        self.vocab = base.config['data_dir'] / 'RE_Train/train_vocab.txt'
        self.pretrain_embeddings = base.config['checkpoint_dir'] / 'bert_lstm_crf_bio_fold_0'
        self.features_candidate = {
                                    'e1_token': {'enable':False, 'value':[]}, 'e2_token': {'enable':False, 'value':[]}, ## 3
                                    'e1_type': {'enable':False, 'value':[]}, 'e2_type': {'enable':False, 'value':[]}, ## 2
                                    'e1_index': {'enable':False, 'value':[]}, 'e2_index': {'enable':False, 'value':[]}, ## 4
                                    'e1_length':{'enable':False, 'value':[]}, 'e2_length': {'enable':False, 'value':[]}, ## 5
                                    'distance_e1e2': {'enable':True, 'value':[]}, 'distance_range': {'enable':False, 'value':[]}, ## 1
                                    'measurement': {'enable':False, 'value':[]}, 'measurement_range': {'enable':False, 'value':[]}, ## 6, 7
                                    'e1_left1_token': {'enable':False, 'value':[]}, 'e1_right1_token': {'enable':False, 'value':[]}, ## 8
                                    'e2_left1_token': {'enable':False, 'value':[]}, 'e2_right1_token': {'enable':False, 'value':[]}, ## 8
                                    'e1_embeddings': {'enable':False, 'value':[]}, 'e2_embeddings': {'enable':False, 'value':[]}
                                }

        self.entity_type = {
                            'NIHSS': 0, 'Measurement': 1,
                            '1a_LOC': 3, '1b_LOCQuestions': 4, '1c_LOCCommands': 5,
                            '2_BestGaze': 6,  '3_Visual': 7,
                            '4_FacialPalsy': 8,
                            '5_Motor': 10, '5a_LeftArm': 11, '5b_RightArm': 12,
                            '6_Motor': 13, '6a_LeftLeg': 14, '6b_RightLeg': 15,
                            '7_LimbAtaxia': 16, '8_Sensory': 17,
                            '9_BestLanguage': 18, '10_Dysarthria':19,
                            '11_ExtinctionInattention': 20
                            }
        
        