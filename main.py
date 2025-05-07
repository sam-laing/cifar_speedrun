"""
modded out the cifar speedrun of Keller Jordan with a goal of
understanding the dynamics of the muon optimizer
"""

#############################################
#                  Setup                    #
#############################################

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
from math import ceil
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)  


from utils import (
    CifarLoader, CifarNet, 
    Muon, zeropower_via_newtonschulz5,
    print_columns, print_training_details, evaluate, logging_columns_list
)
############################################
#                Training                  #
############################################




def main(
        run, model,
        newtonschulz_steps=5,
        seed=None
        ):
    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)


    path = "/fast/slaing/data/vision/cifar10/"
    test_loader = CifarLoader(path, train=False, batch_size=2000, seed=seed)
    train_loader = CifarLoader(
        path, train=True, batch_size=batch_size, 
        aug=dict(flip=True, translate=2), seed=seed)
    
    if run == "warmup":
        generator = torch.Generator(device=train_loader.labels.device)
        generator.manual_seed(seed if seed is not None else 0)
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), 
                                        device=train_loader.labels.device,
                                        generator=generator)

    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(8 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers and learning rate schedulers
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
        ]
    
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
        ]
    
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
        dict(params=norm_biases,         lr=bias_lr, weight_decay=wd/bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)
        ]
    
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=True
        )
    optimizer2 = Muon(
        filter_params, lr=0.24, momentum=0.6, nesterov=True, 
        steps=newtonschulz_steps, eps=1e-7
        )
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0
    def start_timer():
        starter.record()
    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    # Initialize the whitening layer using training images
    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for epoch in range(ceil(total_train_steps / len(train_loader))):

        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:]+optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc



if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")

    base_seed = 99
    print_columns(logging_columns_list, is_head=True)
    main("warmup", model, seed=base_seed)
    


    acc_dict = {} # keyed by newtonschulz_steps, values are lists of accuracies/std dev
    for ns_steps in range(0, 5):
        print("Newton-Schulz steps: %d" % ns_steps)
        accs = torch.tensor([
            main(run, model, newtonschulz_steps=ns_steps, seed=base_seed+run) 
            for run in range(10)
        ])
        print("Mean: %.4f    Std: %.4f" % (accs.mean(), accs.std()))
        acc_dict[ns_steps] = accs.mean().item(), accs.std().item()
    

    # save the results to a file
    import json
    import os
    #save acc_dict to a json file
    with open(f"/home/slaing/cifar_speedrun/plots/det_accs_dict_{base_seed}.json", "w") as f:
        json.dump(acc_dict, f)

    # create a plot of the results
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.array(list(acc_dict.keys()))
    y = np.array([acc_dict[k][0] for k in x])
    yerr = np.array([acc_dict[k][1] for k in x])
    plt.errorbar(x, y, yerr=yerr, fmt="o")
    plt.xticks(x)
    plt.xlabel("Newton-Schulz steps")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Newton-Schulz steps")
    plt.grid()
    plots_dir = "/home/slaing/cifar_speedrun/plots/"
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.savefig(os.path.join(plots_dir, f"muon_accuracy_vs_steps_fixed_seed{base_seed}.png"))
    plt.close()
  
    """
    # no need to log model etc 
    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, accs=accs), log_path)
    print(os.path.abspath(log_path))
    
    """  

    