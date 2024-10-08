import info
import multiprocessing
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

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
        return [torch.LongTensor(padded_src_sen), torch.LongTensor(padded_tgt_sen), src_leng, tgt_leng]
    
    def get_tokenized_data_from_text_file(self, tokenizer, src_path, tgt_path):
        with open(src_path, "r") as file:
            src_lines = file.readlines()
        with open(tgt_path, "r") as file:
            tgt_lines = file.readlines()
        
        for src_line, tgt_line in tqdm(zip(src_lines, tgt_lines), desc="data tokenizing & loading"):
            if (len(src_line.strip()) == 0)|(len(tgt_line.strip()) == 0):
                continue
            src_tokenized_line = tokenizer.encode(src_line).ids
            tgt_tokenized_line = tokenizer.encode(tgt_line).ids
            if (len(src_tokenized_line) > info.max_len) | (len(tgt_tokenized_line) > info.max_len):
                continue
            self.src.append(src_tokenized_line) 
            self.tgt.append(tgt_tokenized_line)

class CustomENFRDataset(Dataset):
    def __init__(self, tokenizer, src_path, tgt_path):
        self.src = []
        self.tgt = []
        self.tokenizer = tokenizer
        self.get_filtered_data_from_text_file(src_path=src_path, tgt_path=tgt_path)
        self.length = len(self.src)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        src_sen = self.tokenizer.encode(self.src[index]).ids
        tgt_sen = self.tokenizer.encode(self.tgt[index]).ids
        src_leng = len(src_sen)
        tgt_leng = len(tgt_sen)
        padded_src_sen = src_sen + [0]*(info.max_len - src_leng) # 128
        padded_tgt_sen = tgt_sen + [0]*(info.max_len - tgt_leng + 1) # 129
        return [torch.LongTensor(padded_src_sen), torch.LongTensor(padded_tgt_sen), src_leng, tgt_leng]
    
    # def get_filtered_data_from_text_file(self, src_path, tgt_path):
    #     src_file = open(src_path, "r")
    #     tgt_file = open(tgt_path, "r")
        
    #     data_size = 40842333
    #     for tmp_idx in tqdm(range(data_size), desc="Filtering..."):
    #         src_line = src_file.readline()
    #         tgt_line = tgt_file.readline()
    #         if len(src_line) == 0:
    #             if len(tgt_line) != 0:
    #                 print("WARNING! Please make sure the data is well paired.")
    #             break
    #         tmp_src_ids = self.tokenizer.encode(src_line).ids
    #         tmp_tgt_ids = self.tokenizer.encode(tgt_line).ids
    #         if (len(tmp_src_ids) > info.max_len) | (len(tmp_tgt_ids) > info.max_len) :
    #             continue                   
    #         self.src.append(src_line)
    #         self.tgt.append(tgt_line)
            
    #     src_file.close()
    #     tgt_file.close()

    def get_filtered_data_from_text_file(self, src_path, tgt_path):
        with open(src_path, "r") as file :
            src_lines = file.readlines()
        with open(tgt_path, "r") as file :
            tgt_lines = file.readlines()
        
        total_lines = zip(src_lines, tgt_lines)
        # data_size = 40842333
        for src_line, tgt_line in tqdm(total_lines, desc="Filtering..."):
            if (len(src_line.strip()) == 0)|(len(tgt_line.strip()) == 0):
                continue
            tmp_src_ids = self.tokenizer.encode(src_line).ids
            tmp_tgt_ids = self.tokenizer.encode(tgt_line).ids
            if (len(tmp_src_ids) > 128) | (len(tmp_tgt_ids) > 128) :
                continue                   
            self.src.append(src_line)
            self.tgt.append(tgt_line)

# def collate_fn(batch):
#     src_sen, tgt_sen, src_len, tgt_len = zip(*batch)
    
#     src_sen = torch.stack(src_sen, dim=0)  # (batch_size, src_seq_len)
#     tgt_sen = torch.stack(tgt_sen, dim=0)  # (batch_size, tgt_seq_len)

#     src_sen = src_sen[:, :max(src_len)]
#     tgt_sen = tgt_sen[:, :max(tgt_len)]

#     return src_sen, tgt_sen, src_len, tgt_len

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