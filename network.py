from torch import nn
import aggregations as aggregations
from backbone import Backbone


class STVGLNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = Backbone(cfg)

        self.aggregation = aggregations.BoQ(
                    in_channels=1024,
                    proj_channels=512,
                    num_queries=64,
                    num_layers=2,
                    row_dim=cfg.features_dim//512,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


class STVGLNet_test(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = Backbone(cfg)
        self.aggregation = aggregations.BoQ(
                    in_channels=1024,
                    proj_channels=512,
                    num_queries=64,
                    num_layers=2,
                    row_dim=16384//512,
        )

    def forward(self, x):
        predictions, frozen_features = self.backbone.inference(x)
        
        return predictions, frozen_features

    def vpr_branch(self, frozen_features):
        feat = self.backbone.unfrozen_layers(frozen_features)
        desc = self.aggregation(feat)
        return desc