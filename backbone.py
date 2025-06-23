import torch
from torch import nn
import torchvision
import argparse
import aggregations as aggregations
from adet.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.features_dim = args.features_dim
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
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
        "--confidence-threshold",
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
    return parser


class Backbone(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cpu_device = torch.device("cpu")

        # build text modle and load pretrain params (frozen)
        self.textmodel = build_model(cfg)
        if cfg.MODEL.WEIGHTS:
            checkpointer = DetectionCheckpointer(self.textmodel)
            checkpointer.load(cfg.MODEL.WEIGHTS)
        for param in self.textmodel.parameters():
            param.requires_grad = False

        # build resnet50
        weights = "IMAGENET1K_V1"
        resnet = torchvision.models.resnet50(weights=weights)

        self.frozen_layers = nn.Sequential(
            self.textmodel.dptext_detr.backbone[0].backbone.stem,
            self.textmodel.dptext_detr.backbone[0].backbone.stages[0],
        )

        self.unfrozen_layers = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
        )

    def inference(self, x):
        batch_size, height, width = x.shape[0], x.shape[-2], x.shape[-1]
        batched_inputs = [{'image': x[i], 'height': height, 'width': width} for i in range(batch_size)]

        predictions, frozen_features = self.textmodel(batched_inputs)

        return predictions, frozen_features

    def forward(self, x):
        x = self.frozen_layers(x)
        torch.cuda.synchronize()
        x = self.unfrozen_layers(x)
        return x


if __name__=="__main__":
    args = get_parser().parse_args()

    cfg = setup_cfg(args)

    model = Backbone(cfg).cuda()
    x = torch.randn(1, 3, 480, 640).to('cuda')
    out = model(x)
    print(out.shape) # 1,1024,30,40
