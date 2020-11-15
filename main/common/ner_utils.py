import torch
from collections import Counter
import numpy as np

def get_entity_bio(seq, id2label):
    chunks = []
    chunk = [-1, -1, -1]
    for idx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith('B-'):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[0] = tag.split('-')[1]
            chunk[1] = idx
            chunk[2] = idx
            if idx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:  
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = idx
            
            if idx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq, id2label, markup='bio'):
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)

class SeqEntityScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()
    
    def reset(self):
        self.origins = []
        self.founds = []
        self.rights =[]

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {'acc': round(precision, 4), 'recall':round(recall, 4), 'f1':round(f1, 4) }
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2label)
            pre_entities = get_entities(pre_path, self.id2label)
            temp_label_entities = []
            temp_pre_entities = []
            for label in label_entities:
                if '56_Motor' not in label[0] and 'Temporal' not in label[0]:
                    temp_label_entities.append(label)
                else:
                    continue
            for pre_label in pre_entities:
                if '56_Motor' not in pre_label[0] and 'Temporal' not in pre_label[0]:
                    temp_pre_entities.append(pre_label)
                else:
                    continue
 
            label_entities = temp_label_entities
            pre_entities = temp_pre_entities

            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities ])

def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i+j))
                break
    return S

if __name__ == "__main__":
    seq = ['B-NIHSS', 'I-NIHSS', 'O', 'B-1a_LOC']
    print(get_entities(seq, None))
