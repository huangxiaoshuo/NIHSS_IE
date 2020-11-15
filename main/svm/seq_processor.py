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
    def __init__(self, guid, text_a, text_b=None, label=None, code=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.code = code

class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, input_len, code, code_position):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_len = input_len
        self.code = code
        self.code_position = code_position

class BertProcessor(object):
    def __init__(self, vocab_path, do_lower_case, min_freq_words=None):
        self.tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=do_lower_case)

    def get_labels(self):
        return ['[CLS]', '[SEP]', 'O',
                'B-NIHSS', 'B-Measurement', 'B-TemporalConstraint', 'B-1a_LOC', 'B-1b_LOCQuestions', 'B-1c_LOCCommands', 'B-2_BestGaze', 'B-3_Visual', 'B-4_FacialPalsy', 'B-56_Motor', 'B-5_Motor', 'B-5a_LeftArm', 'B-5b_RightArm', 'B-6_Motor', 'B-6a_LeftLeg', 'B-6b_RightLeg', 'B-7_LimbAtaxia', 'B-8_Sensory', 'B-9_BestLanguage', 'B-10_Dysarthria', 'B-11_ExtinctionInattention',
                'I-NIHSS', 'I-Measurement', 'I-TemporalConstraint', 'I-1a_LOC', 'I-1b_LOCQuestions', 'I-1c_LOCCommands', 'I-2_BestGaze', 'I-3_Visual', 'I-4_FacialPalsy', 'I-56_Motor', 'I-5_Motor', 'I-5a_LeftArm', 'I-5b_RightArm', 'I-6_Motor', 'I-6a_LeftLeg', 'I-6b_RightLeg', 'I-7_LimbAtaxia', 'I-8_Sensory', 'I-9_BestLanguage', 'I-10_Dysarthria', 'I-11_ExtinctionInattention'
               ]

    def create_examples(self, lines, example_type):
        examples = []
        for i, line in enumerate(lines):
            hadm_id = line['HADM_ID']
            guid = '%s-%s-%d' % (example_type, hadm_id, i)
            sentence = line['token'] # list
            sentence = [' ' if type(t)==float else t for t in sentence ]
            label = line['tags']  # list
            code = line['code']
            # text_a: string. The untokenized text of the first sequence. For single
            # sequence tasks, only this sequence must be specified.
            text_a = ' '.join(sentence) # string
            text_b = None
            examples.append(InputExample(guid=guid, text_a=text_a,text_b=text_b, label=label, code=code))
        return examples

    def create_features(self, examples, max_seq_len):
        label_list = self.get_labels()
        label2id = {label:i for i, label in enumerate(label_list)}

        features = []
        for example_id, example in enumerate(examples): # examples：
            text_list = example.text_a.split(' ')  # string
            label_list = example.label
            code_list = example.code

            new_tokens = [] # tokens
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
                    code_position[key] = (start, end)
                else:
                    continue

            features.append(
                InputFeature(
                    input_ids = input_ids,
                    input_mask = input_mask,
                    segment_ids = new_segment_ids,
                    label_id = new_label_ids,
                    input_len = input_len,
                    code = new_code,
                    code_position = code_position
            ))

        return features
