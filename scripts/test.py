import sys, os
sys.path.append("../src")

import copy
import info
from model import Transformer
from data import CustomDataset

from tokenizers import Tokenizer

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import numpy as np

import time
import argparse
from tqdm import tqdm
from collections import Counter

def get_ngram(n, sentence):
    ngrams = []
    for i in range(len(sentence)-n+1):
        ngrams += [" ".join(sentence[i:i+n])]
    return ngrams

def length_penalty(sentence, alpha = 0.6):
    return ((5+len(sentence))**alpha)/((5+1)**alpha)

def limit_length_repeat_penalty(src_length, score, predict_sen):
    min_length = 0.5*src_length
    max_length = 1.5*src_length
    if len(predict_sen) < min_length:
        return 100000
    if len(predict_sen) > max_length:
        return 100000
    if len(predict_sen) >= 5:
        predict_sen = [str(step.item()) for step in predict_sen]
        predict_ngram = Counter(get_ngram(4, predict_sen))
        for k, v in predict_ngram.items():
            if v >= 2:
                return 100000
    return score

def beam_search(src_sens, model, device, beam_size, boundary=None):
    predict_sentences = []
    model.eval()
    with torch.no_grad():
        for src_sen in tqdm(src_sens[:boundary]):
            stop_flag = False
            src_leng = len(src_sen)
            src_sen = src_sen + [0]*(info.max_len - src_leng)
            encoder_input = torch.LongTensor(src_sen).reshape(1,-1).to(device) # L,
            
            pad_src_mask, attn_src_mask = model.get_mask(encoder_input)       
            encoder_output = model.encoder(encoder_input, attn_src_mask) #N, L, D
            decoder_input = torch.LongTensor([1]).reshape(1,-1).to(device) #N, L
            
            beam = [(0.0, [decoder_input])]
            
            completed_sequences = []
            max_gen_length = round(src_leng*1.5)
            
            for time_step in range(max_gen_length):
                new_beam = []
                for score, sequence in beam:
                    sequence_tensor = torch.cat(sequence, dim = -1)
                    pad_tgt_mask, attn_tgt_mask = model.get_mask(sequence_tensor)
                    src_tgt_mask = (torch.bmm(pad_tgt_mask.unsqueeze(2).float(), pad_src_mask.unsqueeze(1).float()) == 0) #N. TL, SL
                    decoder_output = model.decoder(x=sequence_tensor, src_tgt_masked_info=src_tgt_mask,
                                                   tgt_masked_info=attn_tgt_mask, encoder_output=encoder_output)
                    predict = model.outputlayer(decoder_output).softmax(dim=-1)[:,-1,:]
                    probabilities, candidates = predict.topk(beam_size)
                    for i in range(beam_size):
                        candidate = candidates.squeeze()[i].unsqueeze(0).unsqueeze(0) #
                        prob = probabilities.squeeze()[i]
                        new_sequence = sequence + [candidate]
                        # print(probabilities)
                        # print(prob)
                        # print(beam)
                        # print(new_beam)
                        new_score = (score - torch.log(prob + 1e-7)).item()
                        
                        if candidate.item() == 2: #when search the eos token
                            completed_sequences.append((score, sequence))
                            if len(completed_sequences) >= beam_size:
                                stop_flag = True
                                break
                        else:
                            new_beam.append((new_score, new_sequence))
                        
                    if stop_flag:
                        break
                    
                if stop_flag:
                    break
                
                beam = sorted(new_beam, key=lambda x:x[0])[:beam_size-len(completed_sequences)]
                
            completed_sequences.extend(beam)
            completed_sequences = list(map(list, completed_sequences))
            for ind in range(len(completed_sequences)):
                completed_sequences[ind][0] = limit_length_repeat_penalty(len(src_sen), completed_sequences[ind][0], completed_sequences[ind][1])                    
            completed_sequences = sorted(completed_sequences, key=lambda x: x[0]/(length_penalty(x[1])))[0]
            best_score, best_sequence = completed_sequences
            best_sequence = [step.item() for step in best_sequence]
            predict_sentences.append(best_sequence + [2])
            
    return predict_sentences

def bleu_score(predict, tgt, boundary = None):
    total_bleu_score = 0
    for predict_sentence, tgt_sentence in zip(predict, tgt[:boundary]):
        predict_sentence = list(map(str,predict_sentence))
        tgt_sentence = list(map(str,tgt_sentence))
        n_bleu = dict()
        for n in range(1,5):
            correct = 0
            predict_ngram = Counter(get_ngram(n, predict_sentence)) #예측
            tgt_ngram = Counter(get_ngram(n, tgt_sentence)) #정답
            total = sum(predict_ngram.values()) 
            if total == 0:
                n_bleu[n] = 0
                continue
            for pdt_n, pdt_n_c in predict_ngram.items(): #예측
                if pdt_n in tgt_ngram.keys(): #정답 안에 예측이 있다면
                    tgt_n_c = tgt_ngram[pdt_n] #정답의 개수
                    if tgt_n_c >= pdt_n_c:
                        correct += pdt_n_c #예측을 더함
                    else :
                        correct += tgt_n_c #정답이 더 작으면 정답을 더함
            n_bleu[n] = correct/total
        brevity_penalty = 1
        if len(tgt_sentence) > len(predict_sentence):
            brevity_penalty = np.exp(1 - (len(tgt_sentence)/max(len(predict_sentence),1)))
        bleu = brevity_penalty*np.exp(sum(np.log(max(bs, 1e-7)) for bs in n_bleu.values()) / 4)
        total_bleu_score += bleu
    total_bleu_score /= len(predict)
    total_bleu_score *= 100
    return total_bleu_score

def perplexity(model, test_dataset, device):
    test_cost = 0
    test_ppl = 0
    num = 0
    model.eval()
    test_dataloader = DataLoader(test_dataset, info.batch_size, shuffle=False)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for src_data, tgt_data, src_len, tgt_len in test_dataloader:
            src_data = src_data.to(info.device)
            tgt_data = tgt_data.to(info.device)
            predict = model.forward(src_input=src_data, tgt_input=tgt_data[:,:-1])
            loss = loss_function(predict, tgt_data[:,1:].reshape(-1))
            test_cost += loss.detach().cpu().item()
            test_ppl += torch.exp(loss.detach()).cpu().item()
            num += 1
    test_cost /= num
    test_ppl /= num
    print('='*10, f"Test Loss: {test_cost:<10.4f} Test Ppl: {test_ppl:<10.2f}", '='*10)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.manual_seed(42) 
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", choices=["base", "big"])
    parser.add_argument("--lang", default="ende", choices=["ende", "enfr"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--version")
    parser.add_argument("--avg_ckp", action=argparse.BooleanOptionalAction, help='average check point or not') # --avg_ckp True, --no-avg_ckp False
    option = parser.parse_args()

    name = option.lang + option.version + "_" + option.model
    hyper_params = getattr(info, f"{option.model}_hyper_params")
    info.device = option.device

    tokenizer_path = f"../data/{option.lang}_WMT14_Tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    if option.lang == "ende":
        src_test_data_path = "../data/test/test_en.txt"
        tgt_test_data_path = "../data/test/test_de.txt"
        test_dataset = CustomDataset(tokenizer=tokenizer, src_path=src_test_data_path, tgt_path=tgt_test_data_path)
    elif option.lang == "enfr":
        src_test_data_path = "../data/test/test_enfr_en.txt"
        tgt_test_data_path = "../data/test/test_enfr_fr.txt"
        test_dataset = CustomDataset(tokenizer=tokenizer, src_path=src_test_data_path, tgt_path=tgt_test_data_path)
    
    setup(0,1)
    if option.avg_ckp :
        save_points = ["94000", "95500", "97000", "98500", "100000"]
        model_info = torch.load(f"./save_model/{save_points[0]}_{name}_CheckPoint.pth", map_location=info.device)
        
        avg_state_dict = copy.deepcopy(model_info['model_state_dict'])
        for poi in save_points[1:]:
            model_info = torch.load(f"./save_model/{poi}_{name}_CheckPoint.pth", map_location=info.device)
            for key in avg_state_dict:
                avg_state_dict[key] += model_info['model_state_dict'][key]

        for key in avg_state_dict:
            avg_state_dict[key] /= 5
        
        model = model_info['model'].to(info.device)
        model.load_state_dict(avg_state_dict)
        model = model.module
    else :
        model_info = torch.load(f"./save_model/{name}_CheckPoint.pth", map_location=info.device)
        model = model_info['model'].to(info.device)
        model.load_state_dict(model_info['model_state_dict'])
        model = model.module


    model.eval()
    beam_predict = beam_search(test_dataset.src, model, option.device, beam_size=4)
    beam_bleu_score = bleu_score(beam_predict, test_dataset.tgt)
    print("="*100)
    print(tokenizer.decode(beam_predict[0]))
    print(tokenizer.decode(test_dataset.tgt[0][1:-1]))
    print("="*100)
    print(' '.join(tokenizer.decode(beam_predict[-1])))
    print(' '.join(tokenizer.decode(test_dataset.tgt[-1][1:-1])))
    print("="*50)
    print(f"{name} beam bleu score : {beam_bleu_score:.2f}")
    with open(f"{name}_predict.txt", "w") as file:
        for line in beam_predict:
            file.write(line + "\n")