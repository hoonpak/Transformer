import info
import multiprocessing
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

# class CustomDataset(Dataset):
#     def __init__(self, tokenizer, src_path, tgt_path):
#         manager = multiprocessing.Manager()
#         self.src = manager.list()
#         self.tgt = manager.list()

#         process_src = multiprocessing.Process(target=self.load_and_set_data, args=(self.src, tokenizer, src_path, "src data", 1))
#         process_tgt = multiprocessing.Process(target=self.load_and_set_data, args=(self.tgt, tokenizer, tgt_path, "tgt data", 0))

#         process_src.start()
#         process_tgt.start()

#         process_src.join()
#         process_tgt.join()
        
#         self.filtering()

#         self.length = len(self.src)
    
#     def get_tokenized_data_from_text_file(self, tokenizer, path, desc, position):
#         sentences = []
#         with open(path, "r") as file:
#             lines = file.readlines()
#             for line in tqdm(lines, desc=f"{desc} tokenizing & loading", position=position, leave=True):
#                 tokenized_line = tokenizer.encode(line).ids
#                 sentences.append(tokenized_line)
#         return sentences

#     def load_and_set_data(self, data_list, tokenizer, path, desc, position):
#         print(f"Prepare {desc}")
#         data = self.get_tokenized_data_from_text_file(tokenizer=tokenizer, path=path, desc=desc, position=position)
#         data_list.extend(data)
        
#     def get_filter_index(self, length_list, max_length):
#         filter_index = []
#         for i, s in enumerate(length_list):
#             if s > max_length:
#                 filter_index.append(i)
#         return filter_index
        
#     def filtering(self):
#         src_length_list = list(map(len, self.src))
#         print("src length")
#         tgt_length_list = list(map(len, self.tgt))
#         print("tgt length")
#         src_filter_index = self.get_filter_index(src_length_list, (info.max_len-2))
#         print("src filtering index")
#         tgt_filter_index = self.get_filter_index(tgt_length_list, (info.max_len-2))
#         print("tgt filtering index")
#         total_filter_index = set(src_filter_index + tgt_filter_index)
#         self.src = [sen for i, sen in enumerate(self.src) if i not in total_filter_index]
#         print("src filtering")
#         self.tgt = [sen for i, sen in enumerate(self.tgt) if i not in total_filter_index]
#         print("tgt filtering")

#     def __len__(self):
#         return self.length
        
#     def __getitem__(self, index):
#         src_sen = self.src[index]
#         tgt_sen = self.tgt[index]
#         src_leng = len(src_sen)
#         tgt_leng = len(tgt_sen)
#         padded_src_sen = src_sen + [0]*(info.max_len - src_leng)
#         padded_tgt_sen = tgt_sen + [0]*(info.max_len - tgt_leng)
#         return [torch.LongTensor(padded_src_sen), torch.LongTensor(padded_tgt_sen), (src_leng+tgt_leng)/2]

class CustomDataset(Dataset):
    def __init__(self, tokenizer, src_path, tgt_path):
        self.src = []
        self.tgt = []
        self.get_tokenized_data_from_text_file(tokenizer=tokenizer,src_path=src_path, tgt_path=tgt_path)
        self.length = len(self.src)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        src_sen = self.src[index]
        tgt_sen = self.tgt[index]
        src_leng = len(src_sen)
        tgt_leng = len(tgt_sen)
        padded_src_sen = src_sen + [0]*(info.max_len - src_leng) # 128
        padded_tgt_sen = tgt_sen + [0]*(info.max_len - tgt_leng + 1) # 129
        return [torch.LongTensor(padded_src_sen), torch.LongTensor(padded_tgt_sen), (src_leng+tgt_leng)/2]
    
    def get_tokenized_data_from_text_file(self, tokenizer, src_path, tgt_path):
        src_file = open(src_path, "r")
        tgt_file = open(tgt_path, "r")
        src_lines = src_file.readlines()
        tgt_lines = tgt_file.readlines()
        
        # for idx in tqdm(range(len(src_lines)), desc="data tokenizing & loading"):
            # src_tokenized_line = tokenizer.encode(src_lines[idx]).ids
            # tgt_tokenized_line = tokenizer.encode(tgt_lines[idx]).ids
        for src_line, tgt_line in tqdm(zip(src_lines, tgt_lines), desc="data tokenizing & loading"):
            src_tokenized_line = tokenizer.encode(src_line).ids
            tgt_tokenized_line = tokenizer.encode(tgt_line).ids
            if (len(src_tokenized_line) > (info.max_len-2)) | (len(tgt_tokenized_line) > (info.max_len-2)):
                continue
            self.src.append(src_tokenized_line) 
            self.tgt.append(tgt_tokenized_line) 