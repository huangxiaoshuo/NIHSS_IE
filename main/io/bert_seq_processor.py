import torch
import sys
sys.path.append('..')
from main.common import tools
from main.common import progressbar
from pytorch_transformers import BertTokenizer
from torch.utils.data import TensorDataset
import math
import pandas as pd

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, code=None, relations = None, hadm_id = None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.code = code
        self.relations = relations
        self.hadm_id = hadm_id

class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, input_len, code, new_tokens, relations, hamd_id, code_position):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_len = input_len
        self.code = code
        self.new_tokens = new_tokens
        self.relations = relations
        self.hamd_id = hamd_id
        self.relation_pair_NER = []
        self.code_position = code_position

class BertProcessor(object):
    def __init__(self, vocab_path, do_lower_case, min_freq_words=None):
        self.tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=do_lower_case)

    def get_labels(self):
        return ['[CLS]', '[SEP]', 'O',
                'B-NIHSS', 'B-Measurement', 'B-1a_LOC', 'B-1b_LOCQuestions', 'B-1c_LOCCommands', 'B-2_BestGaze', 'B-3_Visual', 'B-4_FacialPalsy', 'B-5_Motor', 'B-5a_LeftArm', 'B-5b_RightArm', 'B-6_Motor', 'B-6a_LeftLeg', 'B-6b_RightLeg', 'B-7_LimbAtaxia', 'B-8_Sensory', 'B-9_BestLanguage', 'B-10_Dysarthria', 'B-11_ExtinctionInattention',
                'I-NIHSS', 'I-Measurement', 'I-1a_LOC', 'I-1b_LOCQuestions', 'I-1c_LOCCommands', 'I-2_BestGaze', 'I-3_Visual', 'I-4_FacialPalsy', 'I-5_Motor', 'I-5a_LeftArm', 'I-5b_RightArm', 'I-6_Motor', 'I-6a_LeftLeg', 'I-6b_RightLeg', 'I-7_LimbAtaxia', 'I-8_Sensory', 'I-9_BestLanguage', 'I-10_Dysarthria', 'I-11_ExtinctionInattention'
                ]

    @classmethod
    def read_data(cls, input_file, quotechar=None):
        if 'pkl' in str(input_file):
            lines = tools.load_pickle(input_file)
        else:
            lines = input_file
        return lines

    def get_train(self, data_file):
        return self.read_data(data_file)

    def get_valid(self, data_file):
        return self.read_data(data_file)

    def get_test(self, data_file):
        return self.read_data(data_file)

    def create_examples(self, lines, example_type, cached_file):
        if cached_file.exists():
            tools.logger.info("Loading samples from cached files %s", cached_file)
            examples = torch.load(cached_file)
        else:
            pbar = progressbar.ProgressBar(n_total=len(lines), desc=f'create {example_type} samples')
            examples = []
            for i, line in enumerate(lines):
                hadm_id = line['HADM_ID']
                guid = '%s-%s-%d' % (example_type, hadm_id, i)          
                sentence = line['token'] # list
                sentence = [' ' if type(t)==float else t for t in sentence ]
                label = line['tags']  # list
                code = line['code'] # brat entity Tcode T1 T2
                relations = line['relations'] # brat relations golden standard
                # text_a: string. The untokenized text of the first sequence. For single
                # sequence tasks, only this sequence must be specified.
                text_a = ' '.join(sentence) # string
                text_b = None
                examples.append(InputExample(guid=guid, text_a=text_a,text_b=text_b, label=label, code = code, relations = relations, hadm_id = hadm_id))
                pbar(step=i)
            tools.logger.info("Saving examples into cached file %s", cached_file)
            torch.save(examples, cached_file)
        return examples

    def create_features(self, examples, max_seq_len, cached_file):
        if cached_file.exists():
            tools.logger.info('Loading features from cached file %s', cached_file)
            features = torch.load(cached_file)
        else:
            label_list = self.get_labels()
            label2id = {label:i for i, label in enumerate(label_list)}
            pbar = progressbar.ProgressBar(n_total=len(examples), desc='creating the specified features of examples')
            features = []
            for example_id, example in enumerate(examples):
                hamd_id = example.hadm_id
                text_list = example.text_a.split(' ')  # string
                idx_CR = [idx for idx, text in enumerate(text_list) if text == '<CRLF>']
                label_list = example.label
                code_list = example.code
                relation_list = example.relations

                new_tokens = []
                new_segment_ids =[]
                new_label_ids = []
                new_code = []

                new_tokens.append('[CLS]')
                new_segment_ids.append(0)
                new_label_ids.append(label2id['[CLS]'])
                new_code.append('0')
                
                for text, label, code in zip(text_list, label_list, code_list):
                    if text == '<CRLF>':
                        continue
                    else:
                        token_list = self.tokenizer.tokenize(text)
                        for idx, token in enumerate(token_list):
                            new_tokens.append(token)
                            new_segment_ids.append(0)
                            if idx == 0:
                                new_label_ids.append(label2id[label])
                                new_code.append(code)
                            elif label == 'O':
                                new_label_ids.append(label2id[label])
                                new_code.append(code)
                            else:
                                temp_l = 'I-'+label.split('-')[1]
                                new_label_ids.append(label2id[temp_l])
                                new_code.append(code)

                assert len(new_tokens) == len(new_segment_ids)
                assert len(new_tokens) == len(new_label_ids)
                assert len(new_tokens) == len(new_code)

                if len(new_tokens) >= max_seq_len :
                    new_tokens = new_tokens[0:(max_seq_len-1)]
                    new_segment_ids = new_segment_ids[0:(max_seq_len-1)]
                    new_label_ids = new_label_ids[0:(max_seq_len-1)]
                    new_code = new_code[0:(max_seq_len-1)]

                new_tokens.append('[SEP]')
                new_segment_ids.append(0)
                new_label_ids.append(label2id['[SEP]'])
                new_code.append('0')

                input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)
                input_mask = [1] * len(input_ids)
                input_len = len(new_label_ids)

                if len(input_ids) < max_seq_len:
                    pad_zero = [0] * (max_seq_len - len(input_ids))
                    input_ids.extend(pad_zero)
                    input_mask.extend(pad_zero)
                    new_segment_ids.extend(pad_zero)
                    new_label_ids.extend(pad_zero)
                    new_code.extend(['0']* len(pad_zero))

                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(new_segment_ids) == max_seq_len
                assert len(new_label_ids) == max_seq_len
                assert len(new_code) == max_seq_len

                df_temp = pd.DataFrame({'input_ids':input_ids, 'code':new_code})
                agg_fun = lambda s: ( max(s['code']), s.index.tolist()[0], s.index.tolist()[-1])
                groupby_code = df_temp.groupby('code').apply(agg_fun)
                code_position = {}
                for key, start, end in groupby_code:
                    if key != '0':
                        code_position[(start-1 , end-1)] = key
                    else:
                        continue

                if example_id < 2:
                    tools.logger.info('*** Examples: ***')
                    tools.logger.info("guid: %s" % (example.guid))
                    tools.logger.info("tokens: %s" % " ".join([str(x) for x in new_tokens]))
                    tools.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    tools.logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    tools.logger.info("segment_ids: %s" % " ".join([str(x) for x in new_segment_ids]))
                    tools.logger.info("old label name: %s " % " ".join(example.label))
                    tools.logger.info("new label ids: %s" % " ".join([str(x) for x in new_label_ids]))

                features.append(
                    InputFeature(
                        input_ids = input_ids,
                        input_mask = input_mask,
                        segment_ids = new_segment_ids,
                        label_id = new_label_ids,
                        input_len = input_len,
                        code = new_code,
                        new_tokens = new_tokens,
                        relations = relation_list, # golden standard
                        hamd_id = hamd_id,
                        code_position = code_position
                ))

                pbar(step=example_id)
            
            tools.logger.info('Saving features into cached file %s', cached_file)
            torch.save(features, cached_file)
        return features

    def create_dataset(self, features, is_sorted=False):
        if is_sorted:
            tools.logger.info('sorted data by the length of input')
            features = sorted(features, key=lambda x: x.input_len, reverse=True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask =  torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids =  torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids =  torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_input_lens =  torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_lens)

        return dataset