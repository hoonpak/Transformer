import torch

def lrate(step_num, d_model, warmup_steps):
    if step_num == 0:
        step_num = 1
    with torch.no_grad():
        step1 = torch.pow(torch.tensor(d_model),-0.5)
        step2 = torch.min(torch.tensor((torch.pow(torch.tensor(step_num),-0.5), step_num*torch.pow(torch.tensor(warmup_steps),-1.5))))
        learning_rate = step1*step2
    return learning_rate

