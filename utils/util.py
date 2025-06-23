
import os
import torch
import shutil
import logging
from collections import OrderedDict
import numpy as np
from sklearn.decomposition import PCA
import joblib

from datasets import pca_dataset


def save_checkpoint(args, state, is_best, filename):
    model_path = os.path.join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(args.save_dir, "best_model.pth"))
        if not args.use_indoor_datasets:
            shutil.copyfile(model_path, "checkpoints/best_model.pth")


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location="cuda")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r1 = checkpoint["best_r1"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@1 = {best_r1:.1f}")
    # if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        # shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r1, start_epoch_num, not_improved_num


def split_and_assign_qkv_parameters(model, pretrained_dict):

    for block_name, block in model.named_children():
        if 'blocks' in block_name:
            for layer_name, layer in block.named_children():
                for sublayer_name, sublayer in layer.named_children():
                    if 'attn' in sublayer_name:
                        qkv_weight_key = f'{block_name}.{layer_name}.attn.qkv.weight'
                        qkv_bias_key = f'{block_name}.{layer_name}.attn.qkv.bias'
                    
                        qkv_weight = pretrained_dict[qkv_weight_key]
                        qkv_bias = pretrained_dict.get(qkv_bias_key, None)
                        
                        dim = qkv_weight.size(0) // 3
                        
                        q_weight, k_weight, v_weight = qkv_weight.split(dim, dim=0)
                        if qkv_bias is not None:
                            q_bias, k_bias, v_bias = qkv_bias.split(dim)
                        else:
                            q_bias = k_bias = v_bias = None

                        sublayer.q.weight.data = q_weight
                        sublayer.k.weight.data = k_weight
                        sublayer.v.weight.data = v_weight
                        
                        if q_bias is not None:
                            sublayer.q.bias.data = q_bias
                            sublayer.k.bias.data = k_bias
                            sublayer.v.bias.data = v_bias

def compute_pca(args, model, pca_dataset_folder, full_features_dim, pca_file_path = "./logs/pca.pkl"):
    
    try:
        pca = joblib.load(pca_file_path)
        print("Loaded PCA from file.")
        return pca
    except FileNotFoundError:
        print("PCA file not found, computing PCA.")
        
    model = model.eval()
    pca_ds = pca_dataset.PCADataset(args, args.datasets_folder, pca_dataset_folder)
    dl = torch.utils.data.DataLoader(pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    with torch.no_grad():
        for i, images in enumerate(dl):
            if i*args.infer_batch_size >= len(pca_features):
                break
            features = model(images.to("cuda")).cpu().numpy()
            pca_features[i*args.infer_batch_size : (i*args.infer_batch_size)+len(features)] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    
    joblib.dump(pca, pca_file_path)
    print("PCA computed and saved to file.")
    
    return pca

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")


def print_trainable_layers(model):
    """
    Prints the name of trainable parameters in the model.
    """
    layer_names = []
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            layer_names.append(name)

    logging.info(", ".join(layer_names))