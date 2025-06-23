import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cross-domain Switch-aware Re-parameterization for Visual Geo-Loclization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Ablation parameters
    parser.add_argument("--resize_test_imgs", action="store_true",
                        help="traing or testing")
    parser.add_argument("--use_lora", action="store_true",
                        help="low rank adaption")
    parser.add_argument("--use_extra_datasets", action="store_true",
                        help="extra datasets")
    parser.add_argument("--use_indoor_datasets", action="store_true",
                        help="extra datasets")
    parser.add_argument("--num_hiddens", type=int, default=3,
                        help="channel attention")
    parser.add_argument("--use_linear", action="store_true",
                        help="")
    parser.add_argument("--linear_dim", type=int, default=256,
                    help="linear_dim")
    # Training parameters
    parser.add_argument("--use_amp16", action="store_true",
                        help="use Automatic Mixed Precision")
    parser.add_argument("--margin", type=float, default=0.1,
                    help="margin for the triplet loss")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--infer_batch_size", type=int, default=64,
                        help="Batch size for inference.")
    parser.add_argument("--epochs_num", type=int, default=4,
                        help="number of epochs to train")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00006, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="dinov2_vitb14",
                        choices=["ResNet50", "ResNet101", "ResNet152",
                                 "dinov2_vitb14", "dinov2_vits14", "dinov2_vitl14", "dinov2_vitg14"], help="_")
    parser.add_argument("--aggregation", type=str, default="cosgem", choices=["salad", "netvlad", "cosgem", "convap", "g2m", "boq", "mixvpr"])
    parser.add_argument("--trainable_layers", type=str, default="8, 9, 10, 11",
                    help="Comma-separated list of layer indexes to be trained")
    parser.add_argument("--features_dim", type=int, default=768,
                        help="features_dim")
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=16, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[320, 320], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=5, help="_")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                    help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                    help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10], nargs="+",
                        help="Recalls to be computed, such as R@1.")
    parser.add_argument('--precision_values', type=int, default=[1, 5, 10], nargs="+",
                        help="Precisions to be computed, such as P@1.")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--save_dir", type=str, default="",
                        help="Folder name of the current run (saved in ./logs/)")
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="set != 0 if you want to save predictions for each query")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                    choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                    help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted"])
    parser.add_argument("--neg_samples_num", type=int, default=100,
                    help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="set to true if you want to save predictions only for "
                        "wrongly predicted queries")
    parser.add_argument("--get_text", action="store_true", help="get text")
    parser.add_argument("--use_llm", action="store_true",
                    help="set to true if you want to use a llm for text-based reranking")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default="../datasets", help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="MazewithText", help="Relative path of the dataset")
    parser.add_argument("--queries_name", type=str, default=None,
                        help="Path with images to be queried")
    parser.add_argument("--pca_dataset_folder", type=str, default="pitts30k/images/train",
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    
    args = parser.parse_args()

    return args