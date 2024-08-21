import sys, os
sys.path.append("../src")
import time
import argparse

import info
from model import Transformer
from data import CustomDataset, collate_fn
from utils import lrate

from tokenizers import Tokenizer

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="base", choices=["base", "big"])
parser.add_argument("--device", default="cuda:0")
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
    print("load saved dataset")
    training_dataset = torch.load(file_path)
else:
    # src_train_data_path = "../data/test/test_en.txt"
    # tgt_train_data_path = "../data/test/test_de.txt"
    src_train_data_path = "../data/training/training_en.txt"
    tgt_train_data_path = "../data/training/training_de.txt"
    training_dataset = CustomDataset(tokenizer=tokenizer, src_path=src_train_data_path, tgt_path=tgt_train_data_path)
    torch.save(training_dataset, "../data/ende_training_custom_dataset.pt")
    
src_test_data_path = "../data/test/test_en.txt"
tgt_test_data_path = "../data/test/test_de.txt"
test_dataset = CustomDataset(tokenizer=tokenizer, src_path=src_test_data_path, tgt_path=tgt_test_data_path)

train_dataloader = DataLoader(dataset=training_dataset, batch_size=info.batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=info.batch_size, shuffle=False, collate_fn=collate_fn)

model = Transformer(N=hyper_params["N"], vocab_size=vocab_size, pos_max_len=info.max_len,
                    d_model=hyper_params["d_model"], head=hyper_params["head"], d_k=hyper_params["d_k"],
                    d_v=hyper_params["d_v"], d_ff=hyper_params["d_ff"], drop_rate=hyper_params["dropout_rate"]).to(info.device)
with torch.no_grad():
    model.share_embedding.weight[0].fill_(0)
criterion = CrossEntropyLoss(label_smoothing=hyper_params['label_smoothing'], ignore_index=info.PAD).to(info.device)
optim = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
lr_update = LambdaLR(optimizer=optim, lr_lambda=lambda step: lrate(step, hyper_params['d_model'], info.warmup_step))

writer = SummaryWriter(log_dir=f"./runs/{name}")
iter = 0
train_loss = 0
step = 0
# step = 990
token_counts = 0
train_flag = False
test_flag = False
st = time.time()

while True:
    for src_data, tgt_data, n_tokens in train_dataloader:
        torch.cuda.empty_cache()
        token_counts += n_tokens
        src_data = src_data.to(info.device)
        tgt_data = tgt_data.to(info.device)
        
        model.train()
        predict = model.forward(src_input=src_data, tgt_input=tgt_data[:,:-1])
        loss = criterion(predict, tgt_data[:,1:].reshape(-1))
        loss.backward()
        train_loss += loss.detach().cpu().item()
        iter += 1
        
        if token_counts >= 24000:
            model.share_embedding.weight.grad[0].fill_(0)
            optim.step()
            lr_update.step()
            optim.zero_grad()
            if step == hyper_params["train_steps"]:
                train_flag = True
                break
            train_loss /= iter
            print(f"Step: {step:<8} Iter: {iter:<4} Token Num: {token_counts:<9} lr: {optim.param_groups[0]['lr']:<9.1e} Train Loss: {train_loss:<8.4f} Time:{(time.time()-st)/3600:>6.4f} Hour")
            writer.add_scalars('loss', {'train_loss':train_loss}, step)
            writer.flush()
            token_counts = 0
            iter = 0
            train_loss = 0
            step += 1
            test_flag = True
            # with torch.no_grad():
            #     print(model.share_embedding.weight[0].sum().detach().cpu().item())
            #     print(model.outputlayer.weight[0].sum().detach().cpu().item())
            #     print(model.encoder.emb_layer.embedding.weight[0].sum().detach().cpu().item())
            #     print(model.decoder.emb_layer.embedding.weight[0].sum().detach().cpu().item())

        if (step % 1000 == 0) & (test_flag):
            test_cost = 0
            num = 0
            model.eval()
            with torch.no_grad():
                for src_data, tgt_data, n_tokens in test_dataloader:
                    src_data = src_data.to(info.device)
                    tgt_data = tgt_data.to(info.device)
                    predict = model.forward(src_input=src_data, tgt_input=tgt_data[:,:-1])
                    loss = criterion(predict, tgt_data[:,1:].reshape(-1))
                    test_cost += loss.detach().cpu().item()
                    num += 1
            test_cost /= num
            print('='*10, f"Step: {step:<8} Test Loss: {test_cost:<10.4f} Time:{(time.time()-st)/3600:>6.4f} Hour", '='*10)
            writer.add_scalars('cost', {'test_cost':test_cost}, step)
            writer.flush()
            torch.cuda.empty_cache()
            model.train()
            test_flag = False

        if ((step+1) % 10000 == 0) | (step in [100000, 98500, 97000, 95500, 94000]):
            if step in [100000, 98500, 97000, 95500, 94000]:
                torch.save({'step': step,
                            'model': model,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optim.state_dict(),
                            }, f"./save_model/{step}_{name}_CheckPoint.pth")
            else:
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
            
