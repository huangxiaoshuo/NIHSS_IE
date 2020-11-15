import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import logging
import pickle
from main.common.tools import logger
from main.common.tools import load_pickle
from main.common.tools import save_pickle
from main.svm import seq_processor
import torch
from pytorch_transformers.modeling_bert import BertModel
from sklearn import preprocessing

logger = logging.getLogger()

class Features(object):
    def __init__(self, config):
        self.vocab = self.get_vocab(config.vocab)
        self.features_candidate = config.features_candidate
        self.entity_type = config.entity_type
        self.pretrain_model = config.pretrain_embeddings
        self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        self.re_labels = config.re_labels
        self.X = []
        self.y = []
    
    def init_model(self):
        logger.info(f'loading pretrain model from {self.pretrain_model}')
        model = BertModel.from_pretrained(self.pretrain_model)
        model.to(self.device)
        return model

    def get_vocab(self, vocab_path):
        df_vocab = pd.read_csv(vocab_path, sep='\t', header=None, names=['idx', 'token'])
        vocab_list = df_vocab.token.values.tolist()
        return vocab_list

    def get_feature_vetor(self, dataset, cached_file):
        _X, R_FN_X = self.feature_vector(dataset, cached_file)
        self._X = _X
        self.R_FN_X = R_FN_X
        len_embedding = 20
        if _X.shape[1] > len_embedding:
            position = _X.shape[1] - ((_X.shape[1] // len_embedding) * len_embedding )
            scale_X = _X[:, 4: position]
            remained_X = _X[:, position:]
            self.X = np.hstack([preprocessing.scale(scale_X), remained_X])
            self.y = _X[:, 3:4]
        else:
            scale_X = _X[:, 4:]
            self.X = preprocessing.scale(scale_X)
            self.y = _X[:, 3:4]

    def feature_vector(self, dataset, cached_file):
        vocab_path = self.pretrain_model / 'vocab.txt'
        f_name = ''
        for key, _ in self.features_candidate.items():
            if self.features_candidate[key]['enable']:
                 f_name += '_'+ key

        cached_file.mkdir(exist_ok=True)
        cached_file = cached_file / f'{f_name}.pkl'
        

        if cached_file.exists():
            logger.info('Loading features from cached file %s', cached_file)
            features = load_pickle(cached_file)

            FN_cached_file = str(cached_file).replace('.pkl', '_Relation_FN.pkl')
            logger.info('Loading R_FN_X from cached file %s', FN_cached_file)
            R_FN_X = load_pickle(FN_cached_file)
            return features, R_FN_X
        else:
            X = []
            R_FN_X = []
            count = 0
            for data in dataset:
                # data = dataset[61] # HADM_ID=138653 just for test
                count += 1
                print(f'total #{count}, generating features of hadm_id: {data["HADM_ID"]}')
                assert len(data['token']) == len(data['tags'])
                assert len(data['token']) == len(data['code'])

                # bert_input_example = processer.create_examples([data], example_type='train')
                # bert_input_feature = processer.create_features(bert_input_example, max_seq_len = 512)
                # embeddings =  self.get_bert_embeddings(bert_input_feature)
                embeddings = torch.zeros((1, len(data['token']),768))

                single_sample = {'token' : data['token'], 'code': data['code'], 'tags': data['tags']}
                df_data = pd.DataFrame(single_sample)
                df_entities = pd.DataFrame(data['entities'], columns=['code', 'type', 'start', 'end'])

                re_pairs = data['relations']
            
                for re_pair in re_pairs:
                    re_example = []
                    re_FN = []
                    e1_code = re_pair[0]
                    e2_code = re_pair[1]
                    y = self.re_labels[re_pair[3]] # {'Has_Value': 1, '0': 0}

                    if (not len(df_data[df_data['code'] == e1_code]['token'].index)>0) or (not len(df_data[df_data['code'] == e2_code]['token'].index)>0):
                        re_FN.append(data['HADM_ID'])
                        re_FN.append(int(e1_code.replace('T','')))
                        re_FN.append(int(e2_code.replace('T','')))
                        re_FN.append(y)
                        R_FN_X.append(re_FN)
                        continue

                    try:
                        e1_bert_output_start_idx = bert_input_feature[0].code_position[e1_code][0]
                        e1_bert_output_end_idx   = bert_input_feature[0].code_position[e1_code][1]
                        e2_bert_output_start_idx = bert_input_feature[0].code_position[e2_code][0]
                        e2_bert_output_end_idx   = bert_input_feature[0].code_position[e2_code][1]
                    except:
                        e1_bert_output_start_idx = 0
                        e1_bert_output_end_idx   = 0
                        e2_bert_output_start_idx = 0
                        e2_bert_output_end_idx   = 0
                        #continue

                    e1_embeddings = embeddings[0][0][e1_bert_output_start_idx : e1_bert_output_end_idx + 1].cpu().detach().numpy() # (4, 768)
                    e1_embeddings = np.mean(e1_embeddings, axis= 0 ) # (768, )
                    e2_embeddings = embeddings[0][0][e2_bert_output_start_idx : e2_bert_output_end_idx + 1].cpu().detach().numpy() # (4, 768)
                    e2_embeddings = np.mean(e2_embeddings, axis= 0 ) # (768, )

                    e1_token = df_data[df_data['code'] == e1_code]['token']
                    e2_token = df_data[df_data['code'] == e2_code]['token']

                    try:
                        try:
                            e1_token_id = np.mean([ self.vocab.index(token.lower()) for token in e1_token])
                        except:
                            e1_token_id = self.vocab.index('UNK')
                        try:
                            e2_token_id = np.mean([ self.vocab.index(token.lower()) for token in e2_token])
                        except:
                            e2_token_id = self.vocab.index('UNK')
                        e1_type = self.entity_type[df_entities[df_entities['code'] == e1_code]['type'].values.tolist()[0]]
                        e2_type = self.entity_type[df_entities[df_entities['code'] == e2_code]['type'].values.tolist()[0]]
                    except:
                        continue

                    e1_index = df_entities[df_entities['code'] == e1_code]['start'].values.tolist()[0]
                    e2_index = df_entities[df_entities['code'] == e2_code]['start'].values.tolist()[0]
                    e1_index_end = df_entities[df_entities['code'] == e1_code]['end'].values.tolist()[0]
                    e2_index_end = df_entities[df_entities['code'] == e2_code]['end'].values.tolist()[0]
                    e1_length = e1_index_end - e1_index + 1
                    e2_length = e2_index_end - e2_index + 1
                    distance_e1e2 = e2_index - e1_index_end - 1
                    distance_range = 2
                    
                    measurement = e2_token.values.tolist()[0]
                    measurement = int(measurement) if measurement.isdigit() else 0
                    measurement_range = 1 if measurement > 4 else 0

                    e1_left1_token_temp = ' ' if type(df_data.iloc[e1_index - 1]['token']) == float else df_data.iloc[e1_index - 1]['token'].lower()
                    try:
                        e1_left1_token = self.vocab.index(e1_left1_token_temp)
                    except:
                        e1_left1_token = self.vocab.index('UNK')
                    
                    try:
                        e1_right1_token = self.vocab.index( df_data.iloc[e1_index_end + 1]['token'].lower())
                    except:
                        e1_right1_token = self.vocab.index('UNK')
                    
                    try:
                        e2_left1_token = self.vocab.index( df_data.iloc[e2_index - 1]['token'].lower())
                    except:
                        e2_left1_token = self.vocab.index('UNK')
                    
                    try:
                        e2_right1_token = self.vocab.index( df_data.iloc[e2_index_end + 1]['token'].lower())
                    except:
                        e2_right1_token = self.vocab.index('UNK')


                    self.features_candidate['e1_embeddings']['value'] = e1_embeddings.tolist()
                    self.features_candidate['e2_embeddings']['value'] = e2_embeddings.tolist()
                    self.features_candidate['e1_token']['value'] = e1_token_id
                    self.features_candidate['e2_token']['value'] = e2_token_id
                    self.features_candidate['e1_type']['value'] = e1_type
                    self.features_candidate['e2_type']['value'] = e2_type
                    self.features_candidate['e1_index']['value'] = e1_index
                    self.features_candidate['e2_index']['value'] = e2_index
                    self.features_candidate['e1_length']['value'] = e1_length
                    self.features_candidate['e2_length']['value'] = e2_length
                    self.features_candidate['distance_e1e2']['value'] = distance_e1e2
                    self.features_candidate['distance_range']['value'] = distance_range
                    self.features_candidate['measurement']['value'] = measurement
                    self.features_candidate['measurement_range']['value'] = measurement_range

                    self.features_candidate['e1_left1_token']['value'] = e1_left1_token
                    self.features_candidate['e1_right1_token']['value'] = e1_right1_token
                    self.features_candidate['e2_left1_token']['value'] = e2_left1_token
                    self.features_candidate['e2_right1_token']['value'] = e2_right1_token

                    # re_example = []
                    re_example.append(data['HADM_ID'])
                    temp_e1_code = e1_code.replace('TFP_','') if 'TFP_' in e1_code else e1_code.replace('T','')
                    temp_e2_code = e2_code.replace('TFP_','') if 'TFP_' in e2_code else e2_code.replace('T','')
                    re_example.append(temp_e1_code)
                    re_example.append(temp_e2_code)
                    re_example.append(y)

                    for key, _ in self.features_candidate.items():
                        if self.features_candidate[key]['enable']:
                            if 'embeddings' in key:
                                re_example.extend(self.features_candidate[key]['value'])
                            else:
                                re_example.append(self.features_candidate[key]['value'])
                    
                    X.append(re_example)

            X = np.array(X, dtype = np.float)
            logger.info('Saving features into cached file %s', cached_file)
            save_pickle(X, cached_file)

            FN_cached_file = str(cached_file).replace('.pkl', '_Relation_FN.pkl')
            save_pickle(R_FN_X, FN_cached_file)
            np.savetxt(FN_cached_file+'.txt', R_FN_X, fmt='%d')
            return X, R_FN_X

    def get_bert_embeddings(self, input_features):
        self.model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([input_features[0].input_ids], dtype=torch.long).to(self.device)
            segment_ids = torch.tensor([input_features[0].segment_ids], dtype=torch.long).to(self.device)
            input_mask = torch.tensor([input_features[0].input_mask], dtype=torch.long).to(self.device)
            outputs = self.model(input_ids, segment_ids, input_mask)
        return outputs