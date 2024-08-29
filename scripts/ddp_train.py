import sys, os
sys.path.append("../src")
import time
import argparse

import info
from model import Transformer
from data import CustomDataset, CustomENFRDataset
from utils import lrate

from tokenizers import Tokenizer

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    torch.manual_seed(42) 
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, tokenizer, hyper_params, name, lang, test_dataloader):
    setup(rank, world_size)
    device = f"cuda:{rank}"
    # print(device)
    hyper_params["device"] = device
    
    file_path = f'../data/{lang}_training_custom_dataset.pt'
    if os.path.exists(file_path):
        print("load saved dataset")
        training_dataset = torch.load(file_path)
    else:
        if lang == "ende":
            src_train_data_path = "../data/training/training_en.txt"
            tgt_train_data_path = "../data/training/training_de.txt"
            training_dataset = CustomDataset(tokenizer=tokenizer, src_path=src_train_data_path, tgt_path=tgt_train_data_path)
            torch.save(training_dataset, f"../data/{lang}_training_custom_dataset.pt")
        elif lang == "enfr":
            src_train_data_path = "../data/training/training_enfr_en.txt"
            tgt_train_data_path = "../data/training/training_enfr_fr.txt"
            training_dataset = CustomENFRDataset(tokenizer=tokenizer, src_path=src_train_data_path, tgt_path=tgt_train_data_path)
            torch.save(training_dataset, f"../data/{lang}_training_custom_dataset.pt")
    
    if rank == 0:
        print("Training dataset size: ",len(training_dataset.src),len(training_dataset.tgt))
    
    vocab_size = tokenizer.get_vocab_size()
    model = Transformer(N=hyper_params["N"], vocab_size=vocab_size, pos_max_len=info.max_len,
                    d_model=hyper_params["d_model"], head=hyper_params["head"], d_k=hyper_params["d_k"],
                    d_v=hyper_params["d_v"], d_ff=hyper_params["d_ff"], drop_rate=hyper_params["dropout_rate"], device=hyper_params["device"]).to(device)
    # save_info = torch.load(f"/home/user19/bag/5.Transformer/scripts/save_model/{name}_CheckPoint.pth", map_location=device)
    # model = save_info['model'].to(device)
    # model.load_state_dict(save_info['model_state_dict'])
    model = DDP(model, device_ids=[rank])
    
    criterion = CrossEntropyLoss(label_smoothing=hyper_params['label_smoothing']).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    # optim.load_state_dict(save_info['optimizer_state_dict'])
    lr_update = LambdaLR(optimizer=optim, lr_lambda=lambda step: lrate(step, hyper_params['d_model'], info.warmup_step))

    train_sampler = DistributedSampler(training_dataset, shuffle=True)
    train_dataloader = DataLoader(dataset=training_dataset, batch_size=32, sampler=train_sampler)

    step = 0
    # step = save_info['step']
    epoch = 0
    iteration = 0
    avg_iter = 0
    train_loss = 0
    token_counts = 0
    # step_threshold = 24100
    iter_threshold = round(20/world_size)
    train_flag = False
    test_flag = False
    
    if rank == 0:
        st = time.time()
        writer = SummaryWriter(log_dir=f"./runs/{name}")
        
    while True:
        train_sampler.set_epoch(epoch)
        for src_data, tgt_data, src_len, tgt_len in train_dataloader:
            # torch.cuda.empty_cache()
            token_counts += (sum(src_len) + sum(tgt_len))//2
            src_data = src_data.to(device)
            tgt_data = tgt_data.to(device)
            
            model.train()
            predict = model.forward(src_input=src_data, tgt_input=tgt_data[:,:-1])
            loss = criterion(predict, tgt_data[:,1:].reshape(-1))
            loss.backward()
            train_loss += loss.detach().cpu().item()
            iteration += 1

            # token_counts_tensor = torch.tensor([token_counts], device='cuda')
            # dist.all_reduce(token_counts_tensor, op=dist.ReduceOp.SUM)
            
            if iteration == iter_threshold:
                optim.step()
                lr_update.step()
                optim.zero_grad()
                
                avg_iter += iteration
                if rank == 0:
                    if (step % 20 == 0) & (step != 0):
                        train_loss /= avg_iter
                        print(f"Step: {epoch}/{step:<8} Iter: {iteration:<4} Token Num: {token_counts:<7} lr: {optim.param_groups[0]['lr']:<9.1e} Train Loss: {train_loss:<8.4f} Time:{(time.time()-st)/3600:>6.4f} Hour")
                        writer.add_scalars('loss', {'train_loss':train_loss}, step)
                        writer.flush()
                        train_loss = 0
                        avg_iter = 0
                token_counts = 0
                iteration = 0
                step += 1
                test_flag = True
            
                if step == hyper_params["train_steps"]:
                    train_flag = True
                    break

            if rank == 0:
                if (step % 2500 == 0) & (test_flag):
                    test_cost = 0
                    test_ppl = 0
                    num = 0
                    model.eval()
                    with torch.no_grad():
                        for src_data, tgt_data, src_len, tgt_len in test_dataloader:
                            src_data = src_data.to(device)
                            tgt_data = tgt_data.to(device)
                            predict = model.forward(src_input=src_data, tgt_input=tgt_data[:,:-1])
                            loss = criterion(predict, tgt_data[:,1:].reshape(-1))
                            test_cost += loss.detach().cpu().item()
                            test_ppl += torch.exp(loss.detach()).cpu().item()
                            num += 1
                    test_cost /= num
                    test_ppl /= num
                    print('='*10, f"Step: {epoch}/{step:<8} Test Loss: {test_cost:<10.4f} Test Ppl: {test_ppl:<8.2f}  Time:{(time.time()-st)/3600:>6.4f} Hour", '='*10)
                    writer.add_scalars('cost', {'test_cost':test_cost}, step)
                    writer.flush()
                    model.train()
                    # torch.cuda.empty_cache()
                    test_flag = False
                    
                if ((step+1) % 10000 == 0) | ((step) in [98500, 97000, 95500, 94000]):
                    if step in [98500, 97000, 95500, 94000]:
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
            if rank == 0:
                torch.save({'step': step,
                            'model': model,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optim.state_dict(),
                            }, f"./save_model/{step}_{name}_CheckPoint.pth")
            break
        epoch += 1
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", choices=["base", "big"])
    parser.add_argument("--lang", default="ende", choices=["ende", "enfr"])
    parser.add_argument("--version")
    option = parser.parse_args()
    
    info.device = "cuda"
    
    lang = option.lang
    name = lang + option.version + "_" + option.model
    hyper_params = getattr(info, f"{option.model}_hyper_params")
    
    tokenizer_path = f"../data/{lang}_WMT14_Tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # file_path = f'../data/{lang}_training_custom_dataset.pt'
    # if os.path.exists(file_path):
    #     print("load saved dataset")
    #     training_dataset = torch.load(file_path)
    # else:
    #     if lang == "ende":
    #         src_train_data_path = "../data/training/training_en.txt"
    #         tgt_train_data_path = "../data/training/training_de.txt"
    #         training_dataset = CustomDataset(tokenizer=tokenizer, src_path=src_train_data_path, tgt_path=tgt_train_data_path)
    #         torch.save(training_dataset, f"../data/{lang}_training_custom_dataset.pt")
    #     elif lang == "enfr":
    #         src_train_data_path = "../data/training/training_enfr_en.txt"
    #         tgt_train_data_path = "../data/training/training_enfr_fr.txt"
    #         training_dataset = CustomENFRDataset(tokenizer=tokenizer, src_path=src_train_data_path, tgt_path=tgt_train_data_path)
    #         torch.save(training_dataset, f"../data/{lang}_training_custom_dataset.pt")
            
    # print("Training dataset size: ",len(training_dataset.src),len(training_dataset.tgt))
    
    if lang == "ende":
        src_test_data_path = "../data/test/test_en.txt"
        tgt_test_data_path = "../data/test/test_de.txt"
    elif lang == "enfr":
        src_test_data_path = "../data/test/test_enfr_en.txt"
        tgt_test_data_path = "../data/test/test_enfr_fr.txt"
        
    test_dataset = CustomDataset(tokenizer=tokenizer, src_path=src_test_data_path, tgt_path=tgt_test_data_path)    
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=info.batch_size, shuffle=False)

    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp,
             args=(world_size, tokenizer, hyper_params, name, lang, test_dataloader),
             nprocs=world_size,
             join=True)
    