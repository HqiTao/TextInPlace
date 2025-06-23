import os, sys
import logging
from datetime import datetime

from utils import parser, commons, util, test
from network import STVGLNet_test
from datasets import maze_with_text
from backbone import setup_cfg
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = os.path.join("logs", args.save_dir, args.backbone + "_" + args.aggregation, \
                             args.dataset_name, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir, console="info")
commons.make_deterministic(args.seed)
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

cfg = setup_cfg(args)
model = STVGLNet_test(cfg)
model = model.to("cuda")

if args.resume != None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)

for i in range(6):
    if i <= 4:
        # Single floor experiment
        test_ds = maze_with_text.MazeTextDataset(args, floor=i+1)
    else:
        # All floor experiment
        test_ds = maze_with_text.MazeTextDataset(args)
    logging.info(f"Test set: {test_ds}")

    ######################################### TEST on TEST SET #########################################
    recalls, recalls_str, recalls_rerank, recalls_str_rerank = test.test_text_rerank(args, test_ds, model, use_llm=args.use_llm)
    logging.info(f"Test w/o text based rerank")
    logging.info(f"Recalls on {test_ds}: {recalls_str}")

    logging.info(f"Test with text based rerank")
    logging.info(f"Recalls on {test_ds}: {recalls_str_rerank}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
