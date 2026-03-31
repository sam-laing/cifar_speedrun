"""
Script to compare orthogonalization (SVD) vs Newton-Schulz iteration (ns_steps=3)
by training with 10 different seeds and tracking loss curves
"""

#############################################
#                  Setup                    #
#############################################

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  #for deterministic behavior

import sys
import json
import numpy as np
from math import ceil

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

#use this to switch between SVD and NS
ORTHOGONALIZE = False
NS_STEPS = 3

def main(run, model, seed=None):
    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size
    
    #track training losses
    train_losses = []

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
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        generator = torch.Generator(device=train_loader.labels.device)
        generator.manual_seed(seed if seed is not None else 0)
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), 
                                      device=train_loader.labels.device,
                                      generator=generator)
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
        steps=NS_STEPS, eps=1e-7, orthogonalize=ORTHOGONALIZE
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
            
            #calculate and track loss
            loss = F.cross_entropy(outputs, labels, label_smoothing=0.2 )
            print(loss)
            train_losses.append(loss.item())
            
            loss.backward()
            
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

    return tta_val_acc, train_losses

if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")

    base_seed = 42  # Different base seed for this experiment
    print_columns(logging_columns_list, is_head=True)
    
    # Warmup run
    main("warmup", model, seed=base_seed)
    
    # Run with 10 different seeds
    method_name = "orthogonalize" if ORTHOGONALIZE else f"ns_steps_{NS_STEPS}"
    all_accuracies = []
    all_losses = []
    
    #run 10 seeds
    for run in range(1):
        seed = base_seed + run
        print(f"Running seed {seed} with {method_name}")
        accuracy, losses = main(run, model, seed=seed)
        all_accuracies.append(accuracy)
        all_losses.append(losses)
        
        #save individual run losses
        os.makedirs("results", exist_ok=True)
        with open(f"results/x_losses_{method_name}_seed{seed}.json", "w") as f:
            json.dump(losses, f)
    
    #convert lists to numpy arrays for easier calculations
    all_accuracies = np.array(all_accuracies)
    
    #make all loss lists the same length (if they differ slightly)
    min_length = min(len(losses) for losses in all_losses)
    all_losses = np.array([losses[:min_length] for losses in all_losses])
    
    #calculate stats
    mean_accuracy = all_accuracies.mean()
    std_accuracy = all_accuracies.std()
    mean_losses = all_losses.mean(axis=0)
    std_losses = all_losses.std(axis=0)
    
    #save results
    results = {
        "method": method_name,
        "mean_accuracy": float(mean_accuracy),
        "std_accuracy": float(std_accuracy),
        "mean_losses": mean_losses.tolist(),
        "std_losses": std_losses.tolist(),
        "individual_accuracies": all_accuracies.tolist(),
        "ns_steps": NS_STEPS,
        "orthogonalize": ORTHOGONALIZE
    }
    
    with open(f"results/x_summary_{method_name}.json", "w") as f:
        json.dump(results, f)
    
    print(f"\nResults for {method_name}:")
    print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Results saved to results/summary_{method_name}.json")
    
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        steps = np.arange(len(mean_losses))
        plt.plot(steps, mean_losses, label=f'Mean Loss ({method_name})')
        plt.fill_between(steps, 
                         mean_losses - std_losses,
                         mean_losses + std_losses, 
                         alpha=0.3)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title(f'Training Loss: {method_name.replace("_", " ").title()}')
        plt.legend()
        plt.savefig(f"results/loss_curve_{method_name}.png")
        print(f"Plot saved to results/loss_curve_{method_name}.png")
    except ImportError:
        print("Matplotlib not available, skipping plot generation")
        """