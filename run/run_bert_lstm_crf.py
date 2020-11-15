import sys
sys.path.append('..')
from argparse import ArgumentParser
from pathlib import Path
import torch
import datetime
from main.configs import base
from main.common import tools
from main.io.bert_seq_processor import BertProcessor
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from main.model.nn.bert_lstm_crf import BERTLSTMCRF
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from main.common import trainingmonitor
from main.common import modelcheckpoint
from main.train import NER_Trainer
from main.common import optimizater
from main.common import lrscheduler
from main import common
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def main():
    parser = ArgumentParser()
    parser.add_argument('--arch', default='bert_lstm_crf', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--data_name', default='nihss', type=str)
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'lookahead'])
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--checkpoint', default=0, type=int)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--annotation', default='bio', type=str)
    parser.add_argument('--gradient_accumulation_steps', default=1 , type=int)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--train_max_seq_len', default=512, type=int)
    parser.add_argument('--eval_max_seq_len', default=512, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--no_cuda', action='store_true' )
    parser.add_argument('--local_rank', default=-1, type=int )
    parser.add_argument('--sorted', default=1, type=int, help='1:True, 0:False')
    parser.add_argument('--n_gpu', default='0', type=str, help='"0,1,.." or "0" or "" ' )
    parser.add_argument('--warmup_proportion', default=0.05, type=float )
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--grad_clip', default=5.0, type=float)
    parser.add_argument('--mode', default='max', type=str)
    parser.add_argument('--monitor', default='valid_f1', type=str)
    parser.add_argument('--save_best', action='store_true')    
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--resume_path', default='', type=str)

    args = parser.parse_args()

    # For easily debugging, setting args here
    args.arch = 'bert_lstm_crf'
    args.do_train =  False
    args.do_test = False
    args.do_e2e_re = True
    args.do_lower_case = True
    args.data_name = 'new_nihss'
    args.optimizer = 'adam'
    args.epochs = 40
    args.checkpoint = 0
    args.fold = 1
    args.annotation = 'bio'
    args.gradient_accumulation_steps = 1
    args.train_batch_size = 12
    args.eval_batch_size = 50
    args.train_max_seq_len = 512
    args.eval_max_seq_len = 512
    args.learning_rate = 1e-4
    args.seed = 2023
    args.no_cuda = False
    args.local_rank = -1
    args.sorted = 1
    args.n_gpu = '0,1'
    args.warmup_proportion = 0.05
    args.weight_decay = 0.01
    args.grad_clip = 5.0
    args.fp16 = False
    args.mode = 'max'
    args.monitor = 'valid_f1'
    args.save_best = True
    
    
    # config pretrained model, device type, model arch and model dir
    args.device = torch.device(f'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # If training for the first time, load pretrained model from args.pretrain_model, and then,load it from checkpoint path
    args.pretrain_model = base.config['checkpoint_dir'] / f'bert-checkpoint-{args.checkpoint}'
    args.arch = args.arch + f'_{args.annotation}_fold_{args.fold}'
    
    # for saving trained model
    args.model_path = base.config['checkpoint_dir'] / args.arch
    args.model_path.mkdir(exist_ok=True) # 路径已存在时忽略。

    args.resume_path = '' 

    torch.save(args, base.config['checkpoint_dir'] / 'training_args.bin')
    tools.seed_everything(args.seed)
    tools.init_logger(log_file=base.config['log_dir'] / f'{args.arch}.log')
    tools.logger.info('Trainig device type: %s', args.device)
    tools.logger.info('Training/evaluation parameters: %s', args)

    if args.do_train:
        for i in range(0,5):
            args.fold = i
            args.arch = 'bert_lstm_crf'
            args.arch = args.arch + f'_{args.annotation}_fold_{args.fold}'
            args.model_path = base.config['checkpoint_dir'] / args.arch
            args.model_path.mkdir(exist_ok=True)
            torch.save(args, args.model_path / 'training_args.bin')
            run_train(args)
    
    if args.do_test:
        for fold_i in range(0,5):
            args.fold = fold_i
            args.arch = 'bert_lstm_crf'
            args.arch = args.arch + f'_{args.annotation}_fold_{args.fold}'
            args.model_path = base.config['checkpoint_dir'] / args.arch
            run_test(args)
        pass

    #using the best model
    if args.do_e2e_re:
        args.fold = 3
        args.arch = 'bert_lstm_crf'
        args.arch = args.arch + f'_{args.annotation}_fold_{args.fold}'
        args.model_path = base.config['checkpoint_dir'] / args.arch
        run_end2end_realtion_extration(args)
    pass

def run_train(args):
    vocab_path = args.pretrain_model / 'bert-base-cased-vocab.txt'
    processer = BertProcessor(vocab_path=vocab_path, do_lower_case=args.do_lower_case)
    processer.tokenizer.save_vocabulary(str(args.model_path))
    label_list = processer.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}

    train_data_path = base.config['data_dir'] / f'train/{args.data_name}_train_fold_{args.fold}.pkl'
    train_data = processer.get_train(train_data_path)
    train_examples_cached_file = base.config['data_dir'] / f'train/cached/cached_{args.data_name}_train_fold_{args.fold}_examples'
    train_examples = processer.create_examples(lines=train_data, example_type='train', cached_file=train_examples_cached_file)
    train_features_cached_file = base.config['data_dir'] / f'train/cached/cached_{args.data_name}_train_fold_{args.fold}_features_{args.train_max_seq_len}'
    train_features = processer.create_features(examples=train_examples, max_seq_len=args.train_max_seq_len,cached_file=train_features_cached_file)
    train_dataset = processer.create_dataset(train_features, is_sorted=args.sorted)
    
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_data_path = base.config['data_dir'] / f'train/{args.data_name}_valid_fold_{args.fold}.pkl'
    valid_data = processer.get_valid(valid_data_path)
    valid_examples_cached_file = base.config['data_dir'] / f'train/cached/cached_{args.data_name}_valid_fold_{args.fold}_examples'    
    valid_examples = processer.create_examples(lines=valid_data, example_type='valid',cached_file=valid_examples_cached_file)
    valid_features_cached_file = base.config['data_dir'] / f'train/cached/cached_{args.data_name}_valid_fold_{args.fold}_features_{args.eval_max_seq_len}'
    valid_features = processer.create_features(examples=valid_examples, max_seq_len=args.eval_max_seq_len,cached_file=valid_features_cached_file)
    valid_dataset = processer.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)

    tools.logger.info('initializing model')

    model = BERTLSTMCRF
    model = model.from_pretrained(args.pretrain_model, label2id=label2id, device=args.device)
    model = model.to(args.device) #2020-9-19

    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs )
    bert_param_optimizer = list(model.bert.named_parameters())
    lstm_param_optimizer = list(model.bilstm.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # n is the name of parameter, p contains the data.
    optimizer_grouped_parameters = [
         {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': args.learning_rate},
         {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': 0.001},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 0.001},
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': 0.001},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 0.001}
    ]

    if args.optimizer == 'adam':
        optimizer = optimizater.BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, 
                                         warmup=args.warmup_proportion, t_total=t_total)
    lr_scheduler = lrscheduler.BERTReduceLROnPlateau(optimizer, lr=args.learning_rate, mode=args.mode, factor=0.5, patience=5,
                                                     verbose=1, epsilon=1e-8, cooldown=0, min_lr=0, eps=1e-8)

    figure_dir = base.config['figure_dir']
    train_monitor = trainingmonitor.TrainingMonitor(file_dir=figure_dir, arch=args.arch)
    model_checkpoint = modelcheckpoint.ModelCheckpoint(checkpoint_dir=args.model_path, mode=args.mode,
                                                       monitor=args.monitor, arch=args.arch, save_best_only=args.save_best)

    tools.logger.info('****** Running Training Model ******')
    tools.logger.info(' Num examples = %d', len(train_examples))
    tools.logger.info(' Num epochs = %d', args.epochs)
    tools.logger.info(' Total train batch size = %d', args.train_batch_size * args.gradient_accumulation_steps *
                    (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    tools.logger.info(' Gradient accumulation steps = %d',  args.gradient_accumulation_steps)
    tools.logger.info(' Total optimization steps = %d', t_total)

    trainer = NER_Trainer.Trainer(n_gpu=args.n_gpu,
                                  model = model,
                                  logger = tools.logger,
                                  optimizer = optimizer,
                                  lr_scheduler = lr_scheduler,
                                  label2id = label2id,
                                  training_monitor = train_monitor,
                                  fp16 = args.fp16,
                                  resume_path = args.resume_path,
                                  grad_clip = args.grad_clip,
                                  model_checkpoint = model_checkpoint,
                                  gradient_accumulation_steps = args.gradient_accumulation_steps)
    
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader, epochs= args.epochs, seed=args.seed)

def run_test(args):
    from main.common import progressbar
    from main.common import ner_utils
    from main.common.tools import save_pickle
    args.resume_path = args.model_path
    processor = BertProcessor(args.resume_path / 'vocab.txt', args.do_lower_case)
    label_list = processor.get_labels() # all labels
    label2id = {label:i for i, label in enumerate(label_list)}
    id2label = {i:label for i, label in enumerate(label_list)}
    model = BERTLSTMCRF
    model = model.from_pretrained(args.resume_path, label2id=label2id, device=args.device)
    tools.logger.info(f'loaded model from {args.resume_path}')
    model.to(args.device)
    max_seq_len = args.eval_max_seq_len
    
    test_data_path = base.config['data_dir'] / 'test/new_nihss_ner_test.pkl'
    test_data = processor.get_test(test_data_path)
    test_examples_cached_file = base.config['data_dir'] / f'test/cached/cached_{args.data_name}_test_examples'
    test_examples = processor.create_examples(lines=test_data, example_type='test', cached_file=test_examples_cached_file)
    test_features_cached_file = base.config['data_dir'] / f'test/cached/cached_{args.data_name}_test_features_{args.eval_max_seq_len}'
    test_features = processor.create_features(examples=test_examples, max_seq_len=args.eval_max_seq_len, cached_file=test_features_cached_file)

    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    tools.logger.info('****** Running Testing Model ******')
    tools.logger.info(' Num test examples = %d', len(test_examples))

    pbar = progressbar.ProgressBar(n_total=len(test_dataloader),desc='Testing')

    entity_score = common.ner_utils.SeqEntityScore(id2label)
    entity_score.reset()
    test_loss = tools.AverageMeter()

    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, input_lens = batch
        input_lens = input_lens.cpu().detach().numpy().tolist()
        model.eval()
        with torch.no_grad():
            features, loss = model.forward_loss(input_ids, segment_ids, input_mask, label_ids, input_lens)
            tags, _ = model.crf._obtain_labels(features, id2label, input_lens)
        test_loss.update(val=loss.item(), n=input_ids.size(0))
        pbar(step=step, info={'loss':loss.item()})
        label_ids = label_ids.to('cpu').numpy().tolist()
        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == label2id['[SEP]']:
                    entity_score.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(id2label[label_ids[i][j]])
                    temp_2.append(tags[i][j])
    test_info, class_info = entity_score.result()
    info = {f'test_{key}': value for key, value in test_info.items()}
    info['test_loss'] = test_loss.avg
    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()

    logs = dict(**info)
    show_info = f'Test: ' + " -".join([f' {key}: {value:.4f}' for key, value in logs.items()])
    tools.logger.info(show_info)
    tools.logger.info("The entity scores of test data : ")

    result_path = base.config['result_dir'] / args.arch
    result_path.mkdir(exist_ok=True)
    result_file_path = result_path / f'{args.arch}_test_result_{str(datetime.date.today())}.txt'
    tools.logger.info(f'Saving test data to {result_file_path}')
    with open(str(result_file_path), 'a+') as f:
        content = show_info + '\n'
        f.write(content)

        for key, value in class_info.items():
            info = f'Entity: {key} \t' + "-\t".join([f' {key_}: {value_:.4f} ' for key_, value_ in value.items()])
            tools.logger.info(info)
            f.write(info + '\n')
    f.close()


# end2end relation extraction：
def run_end2end_realtion_extration(args):
    from main.common import progressbar
    from main.common import ner_utils
    from main.common.tools import save_pickle
    from random import choice
    import pandas as pd
    import copy
    
    args.resume_path = args.model_path
    processor = BertProcessor(vocab_path = args.resume_path / 'vocab.txt', do_lower_case=True)
    label_list = processor.get_labels() # all labels
    label2id = {label:i for i, label in enumerate(label_list)}
    id2label = {i:label for i, label in enumerate(label_list)}
    model = BERTLSTMCRF
    tools.logger.info(f'loading prtrained model from {args.resume_path}')
    model = model.from_pretrained(args.resume_path, label2id=label2id, device=args.device)
    model.to(args.device)

    test_data_path = base.config['data_dir'] / f'RE_Test/re_test.pkl'
    test_data = processor.get_test(test_data_path)
    
    test_examples_cached_file = base.config['data_dir'] / f'End2End_Test/cached_{args.data_name}_e2e_test_examples.pkl'
    test_examples = processor.create_examples(lines=test_data, example_type='test', cached_file=test_examples_cached_file)
    test_features_cached_file = base.config['data_dir'] / f'End2End_Test/cached_{args.data_name}_e2e_test_features.pkl'
    test_features = processor.create_features(examples=test_examples, max_seq_len=args.eval_max_seq_len, cached_file=test_features_cached_file)

    test_datasets = test_features
    
    pbar = progressbar.ProgressBar(n_total=len(test_datasets),desc='Testing End2End relation extraction performace')

    entity_score = common.ner_utils.SeqEntityScore(id2label)
    entity_score.reset()

    ner_output_samples = []
    for step, one_sample in enumerate(test_datasets):
        entity_score.reset()
        hadm_id = one_sample.hamd_id
        input_ids = torch.tensor([one_sample.input_ids], dtype=torch.long)
        input_mask =  torch.tensor([one_sample.input_mask], dtype=torch.long)
        segment_ids =  torch.tensor([one_sample.segment_ids], dtype=torch.long)
        label_ids =  torch.tensor([one_sample.label_id], dtype=torch.long)
        input_lens =  torch.tensor([one_sample.input_len], dtype=torch.long)
        true_codes = one_sample.code
        relations = one_sample.relations 
        code_position = one_sample.code_position 
        new_tokens = one_sample.new_tokens[1:-1]


        batch = (input_ids, input_mask, segment_ids, label_ids, input_lens)
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, input_lens = batch
        input_lens = input_lens.cpu().detach().numpy().tolist()
        model.eval()

        with torch.no_grad():
            features, loss = model.forward_loss(input_ids, segment_ids, input_mask, label_ids, input_lens)
            tags, _ = model.crf._obtain_labels(features, id2label, input_lens)

        label_ids = label_ids.to('cpu').numpy().tolist()
        pbar(step=step, info={'loss':loss.item()})

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == label2id['[SEP]']:
                    entity_score.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(id2label[label_ids[i][j]])
                    temp_2.append(tags[i][j])

        relation_pairs_NER = []
        relation_pairs_dict = {}

        count_y1 = 0
        for relation in relations:
            if relation[3] == 'Has_Value':
                relation_pairs_NER.append(relation)
                relation_pairs_dict.setdefault(relation[0],[]).append(relation[1])
                count_y1 += 1

        found_entities = []
        found_e1_T_code = []
        found_e2_T_code = []
        TFN_idx = 1
        for entity in entity_score.founds:
            key = (entity[1], entity[2])
            e_type = entity[0]
            if key in code_position.keys():
                T_code = code_position[key]
                found_entities.append((T_code, e_type, entity[1], entity[2]))
            else:
                T_code = 'TFP_' + str(TFN_idx)
                found_entities.append((T_code, e_type, entity[1], entity[2]))
                TFN_idx += 1
            if e_type == 'Measurement':
                found_e2_T_code.append(T_code)
            else:
                found_e1_T_code.append(T_code)

        pred_tags = tags[0][1:-1] 
        assert len(new_tokens) == len(pred_tags) 
        pred_codes = ['0'] * len(new_tokens)
        for e in found_entities:
            start = e[2] 
            end = e[3]
            for idx in range(start, end+1):
                pred_codes[idx] = e[0]

        b = 1
        count_y0 = 0
        relations_temp = copy.deepcopy(relation_pairs_dict)
        while count_y0 <= (count_y1 + b) and len(found_e1_T_code) > 1:
            e1_random = choice(found_e1_T_code)
            try:
                e1_random_correspond_e2 = relations_temp[e1_random]
                other_e2_codes = list(set(found_e2_T_code).difference(set(e1_random_correspond_e2)))
                e2_random = choice(other_e2_codes)
            except:
                if len(found_e2_T_code) > 0:
                    e2_random = choice(found_e2_T_code)
                else:
                    count_y0 += 1
                    continue

            if e1_random not in relations_temp.keys():
                relations_temp[e1_random] = [e2_random]
                relation_pairs_NER.append((e1_random, e2_random, 'RTN_'+str(count_y0),'0'))
                count_y0 += 1
                continue
            elif e2_random not in relations_temp[e1_random]:
                value_list = copy.deepcopy(relations_temp[e1_random])
                value_list.append(e2_random)
                relations_temp[e1_random] = value_list
                relation_pairs_NER.append((e1_random, e2_random, 'RTN_'+str(count_y0),'0'))
                count_y0 += 1
                continue
            else:
                count_y0 += 1
                continue

        single_sample = {'token' :new_tokens, 'codes': pred_codes, 'tags': pred_tags}
        df_temp = pd.DataFrame(single_sample)
        current_row = 0
        while current_row <= df_temp.shape[0] - 1:
            i = 0
            temp_token = ''
            current_token = df_temp.iloc[current_row][0]
            if not current_token.startswith('##'):
                current_row += 1
                continue
            else:
                while (current_row + i) <= (df_temp.shape[0] - 1) and df_temp.iloc[current_row + i][0].startswith('##'):
                    temp_token += df_temp.iloc[current_row + i][0].replace('##','')
                    i += 1
                start_word_piece_position = current_row - 1
                df_temp.iloc[start_word_piece_position][0] += temp_token
                current_row += i
        
        df_temp = df_temp[df_temp['token'].str.startswith('##') == False]
        new_tokens = df_temp['token'].values.tolist()
        pred_tags = df_temp['tags'].values.tolist()
        pred_codes = df_temp['codes'].values.tolist()
        
        df_temp = df_temp.reset_index(drop=True)
        agg_fun = lambda s: ( max(s['codes']), s['tags'].iloc[0], s.index.tolist()[0], s.index.tolist()[-1])
        groupby_code = df_temp.groupby('codes').apply(agg_fun)
        new_found_entities = []
        for key, e_type, start, end in groupby_code:
            if key != '0':
                e_type = e_type.split('-')[1]
                new_found_entities.append((key, e_type, start, end))
            else:
                continue

        sample = {'HADM_ID': hadm_id,
                  'token': new_tokens,
                  'tags': pred_tags,
                  'relations': relation_pairs_NER,
                  'entities': new_found_entities,
                  'code': pred_codes}
        ner_output_samples.append(sample)

    content = str(ner_output_samples)
    file_path = base.config['data_dir'] / 'End2End_Test/re_e2e_test.txt'
    with open(file_path, 'w+') as new_f:
        new_f.writelines(content)
    new_f.close()

    e2e_test_file_path = base.config['data_dir'] / f'End2End_Test/re_e2e_test_by_fold_{args.fold}.pkl'
    tools.logger.info(f'Saving e2e_test_file into {e2e_test_file_path}')
    save_pickle(ner_output_samples, e2e_test_file_path)

if __name__ == "__main__":
    main()