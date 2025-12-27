import os, random
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_metric_learning import losses, miners, distances

from utils import util, parser, commons, test
from network import STVGLNet
from datasets import gsv_cities, maze_with_text
from backbone import setup_cfg

torch.backends.cudnn.benchmark = True  # Provides a speedup
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = os.path.join("logs", args.save_dir, args.backbone + "_" + args.aggregation, "gsv_cities", start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs")

#### Creation of Datasets
logging.debug(f"Loading gsv_cities and {args.dataset_name} from folder {args.datasets_folder}")

val_ds = maze_with_text.MazeTextDataset(args, floor=3)
logging.info(f"Val set: {val_ds}")

#### Initialize model
cfg = setup_cfg(args)
model = STVGLNet(cfg)
model = model.to("cuda")

model = torch.nn.DataParallel(model)

util.print_trainable_parameters(model)
util.print_trainable_layers(model)

#### Setup Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
if not args.resume:
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.2, total_iters=4000)
criterion = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=distances.CosineSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=distances.CosineSimilarity())
if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, _, best_r1, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@1 {best_r1:.1f}")
    if args.aggregation == "netvlad" and args.use_linear:
        best_r1 = 0
    if args.use_indoor_datasets:
        best_r1 = start_epoch_num = not_improved_num = 0
else:
    best_r1 = start_epoch_num = not_improved_num = 0

#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    epoch_num += 1
    logging.info(f"Start training epoch: {epoch_num:02d}")

    if args.use_indoor_datasets:
        random_datasets = gsv_cities.INDOOR_DATASETS.copy()
    else: 
        random_datasets = gsv_cities.TRAIN_CITIES.copy()
    random.shuffle(random_datasets)
    train_ds = gsv_cities.GSVCitiesDataset(args, cities=(random_datasets))
    train_dl = DataLoader(train_ds, batch_size= args.train_batch_size, num_workers=args.num_workers, pin_memory= True)
    epoch_start_time = datetime.now()
    epoch_losses=[]

    model.train()
    for batch_idx, (places, labels) in enumerate(tqdm(train_dl, ncols=100, desc=f"Epoch {epoch_num}/{args.epochs_num}")):

        BS, N, ch, h, w = places.shape
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        optimizer.zero_grad()

        if not args.use_amp16:
            features = model(images)
            miner_outputs = miner(features, labels)
            loss = criterion(features, labels, miner_outputs)
            loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                features = model(images)
                miner_outputs = miner(features, labels)
                loss = criterion(features, labels, miner_outputs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        if not args.resume:
            scheduler.step()

        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)

        del loss, features, miner_outputs, images, labels, batch_loss

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
        f"average epoch loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

    
    is_best = recalls[0] > best_r1
    
    if is_best:
        logging.info(f"Improved: previous best R@1 = {best_r1:.1f}, current R@1 = {(recalls[0]):.1f}")
        best_r1 = (recalls[0])
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@1 = {best_r1:.1f}, current R@1 = {(recalls[0]):.1f}")

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r1": best_r1,
        "not_improved_num": not_improved_num
    }, is_best, filename=f"last_model.pth")

    if not_improved_num == args.patience:
        logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
        break

logging.info(f"Best R@1: {best_r1:.1f}")
logging.info(f"Trained for {epoch_num:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

# update test

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args.resume = f"{args.save_dir}/best_model.pth"

cfg = setup_cfg(args)
model = STVGLNet(cfg)

model = model.to("cuda")

model = util.resume_model(args, model)

model = torch.nn.DataParallel(model)

for i in range(6):
    if i <= 4:
        test_ds = maze_with_text.MazeTextDataset(args, floor=i+1)
    else:
        test_ds = maze_with_text.MazeTextDataset(args)
    logging.info(f"Test set: {test_ds}")
    
    ######################################### TEST on TEST SET #########################################
    recalls, recalls_str = test.test(args, test_ds, model)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
