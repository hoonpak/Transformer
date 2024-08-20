import sys, os
sys.path.append("../src")
import time
import argparse

import info
from model import Transformer
from data import CustomDataset
from utils import lrate

from tokenizers import Tokenizer

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="base", choices=["base", "big"])
parser.add_argument("--device", default="cuda:3")
parser.add_argument("--version")
option = parser.parse_args()

name = option.version + "_" + option.model
hyper_params = getattr(info, f"{option.model}_hyper_params")
info.device = option.device

tokenizer_path = "../data/ende_WMT14_Tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

file_path = '../data/ende_training_custom_dataset.pt'

if os.path.exists(file_path):
    training_dataset = torch.load(file_path)
else:
    src_train_data_path = "../data/training/training_en.txt"
    tgt_train_data_path = "../data/training/training_de.txt"
    training_dataset = CustomDataset(tokenizer=tokenizer, src_path=src_train_data_path, tgt_path=tgt_train_data_path)
    torch.save(training_dataset, "ende_training_custom_dataset.pt")
    
src_test_data_path = "../data/test/test_en.txt"
tgt_test_data_path = "../data/test/test_de.txt"
test_dataset = CustomDataset(tokenizer=tokenizer, src_path=src_test_data_path, tgt_path=tgt_test_data_path)

train_dataloader = DataLoader(dataset=training_dataset, batch_size=info.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=info.batch_size, shuffle=False)

model = Transformer(N=hyper_params["N"], vocab_size=vocab_size, pos_max_len=info.max_len,
                    d_model=hyper_params["d_model"], head=hyper_params["head"], d_k=hyper_params["d_k"],
                    d_v=hyper_params["d_v"], d_ff=hyper_params["d_ff"], drop_rate=hyper_params["dropout_rate"]).to(info.device)
criterion = CrossEntropyLoss(label_smoothing=hyper_params['label_smoothing']).to(info.device)
optim = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
lr_update = LambdaLR(optimizer=optim, lr_lambda=lambda step: lrate(step, hyper_params['d_model'], info.warmup_step))

writer = SummaryWriter(log_dir=f"./runs/{name}")
iter = 0
train_loss = 0
step = 0
token_counts = 0
train_flag = False
st = time.time()

while True:
    for src_data, tgt_data, n_tokens in train_dataloader:
        token_counts += n_tokens.sum()
        src_data = src_data.to(info.device)
        tgt_data = tgt_data.to(info.device)
        
        model.train()
        predict = model.forward(src_input=src_data, tgt_input=tgt_data[:,:-1])
        loss = criterion(predict, tgt_data[:,1:].reshape(-1))
        loss.backward()
        
        if token_counts >= 24000:
            optim.step()
            lr_update.step()
            optim.zero_grad()
            token_counts = 0
            if step == hyper_params["train_steps"]:
                train_flag = True
                break
            step += 1
        
        train_loss += loss.detach().cpu().item()
        iter += 1
    
        if iter % 5000 == 0:
            train_loss /= 5000
            print(f"Step: {step:<10} Iter: {iter:<10} Train Loss: {train_loss:<10.4f} Time:{(time.time()-st)/3600:>6.4f} Hour")
            writer.add_scalars('loss', {'train_loss':train_loss}, iter)
            writer.flush()
            train_loss = 0

        if step % 5000 == 0:
            test_cost = 0
            num = 0
            model.eval()
            with torch.no_grad():
                for src, src_len, tgt in test_dataloader:
                    src = src.to(info.device)
                    tgt = tgt.to(info.device)
                    predict = model.forward(src, src_len, tgt)
                    loss = criterion(predict, tgt[:,1:].reshape(-1))
                    test_cost += loss.detach().cpu().item()
                    num += 1
            test_cost /= num
            print('='*10, f"Step: {step:<10} Test Loss: {test_cost:<10.4f} Time:{(time.time()-st)/3600:>6.4f} Hour", '='*10)
            writer.add_scalars('cost', {'test_cost':test_cost}, iter)
            writer.flush()
            model.train()

        if step % 10000 == 0:
            torch.save({'step': step,
                        'model': model,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        }, f"./save_model/{name}_CheckPoint.pth")
            
    if train_flag:
        torch.save({'step': step,
                    'model': model,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, f"./save_model/{name}_CheckPoint.pth")
        break

model.eval()
with torch.no_grad():
    for src, src_len, tgt in test_dataloader:
        src = src.to(info.device)
        tgt = tgt.to(info.device)
        predict = model.forward(src, src_len, tgt)
        loss = criterion(predict, tgt[:,1:].reshape(-1))
        test_cost += loss.detach().cpu().item()
        num += 1
test_cost /= num
print('#'*10, f"Step: {step:<10} Test Loss: {test_cost:<10.4f} Time:{(time.time()-st)/3600:>6.4f} Hour", '#'*10)
writer.add_scalars('cost', {'test_cost':test_cost}, iter)
writer.flush()
            
