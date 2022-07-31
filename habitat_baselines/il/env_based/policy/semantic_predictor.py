import sys
import time
import torch
import numpy as np
import time
import os

from torch import nn

from habitat import Config, logger
from habitat_baselines.il.env_based.policy.rednet import load_rednet

# ShapeConv imports
from habitat_baselines.il.env_based.policy.shapeconv.config import ShapeConvConfig
from habitat_baselines.il.env_based.policy.shapeconv import build_model
from scripts.utils.hm3d_utils import mask_shapeconv_new_cats, shapeconv_hm3d_to_rednet_mapping

# sys.path.insert(0, "/srv/flash1/rramrakhya6/spring_2022/AnalyzeSemanticDataset/dependencies/ShapeConv/")

# from rgbd_seg.utils import Config as ShapeConvConfig
# from rgbd_seg.models import build_model


class SemanticPredictor(nn.Module):
    r"""A wrapper over semantic predictor network.
    """

    def __init__(self, model_config: Config, device):
        super().__init__()
        self.model_config = model_config
        self.predictor = None
        self.device = device

        self.MAX_DEPTH = 5.0
        self.MIN_DEPTH = 0.5

        self.is_shapeconv = None
        self.only_rednet_cats = model_config.SEMANTIC_PREDICTOR.SHAPECONV.only_rednet_cats
        if model_config.SEMANTIC_PREDICTOR.name == "shapeconv":
            self.is_shapeconv = True
            self.shapeconv_config = ShapeConvConfig.fromfile(model_config.SEMANTIC_PREDICTOR.SHAPECONV.config)
            self.predictor = build_model(self.shapeconv_config["inference"]["model"])

            # Load checkpoint
            checkpoint = torch.load(model_config.SEMANTIC_PREDICTOR.SHAPECONV.pretrained_weights, map_location="cpu")
            self.predictor.load_state_dict(checkpoint["state_dict"])

            self.predictor.to(self.device)
            logger.info("Initializing ShapeConv semantic predictor...")

            self.category_map = np.zeros(25, dtype=np.int8)
            for i in range(self.category_map.shape[0]):
                self.category_map[i] = i

            for key, value in shapeconv_hm3d_to_rednet_mapping.items():
                self.category_map[key] = value
            self.category_map = torch.tensor(self.category_map, device=device)
        else:
            # Default to RedNet predictor
            self.predictor = load_rednet(
                self.device,
                ckpt=model_config.SEMANTIC_PREDICTOR.REDNET.pretrained_weights,
                resize=True, # since we train on half-vision
                num_classes=model_config.SEMANTIC_PREDICTOR.REDNET.num_classes
            )
            logger.info("Initializing RedNet semantic predictor...")
        self.predictor.eval()

        # self.eval()

    def forward(self, observations):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]
        depth_obs = observations["depth"].clone()
        x = None
        batch_end_time, inf_end_time, softmax_time = 0,0,0
        if self.is_shapeconv:
            with torch.no_grad():
                st_time = time.time()

                # unnormalize dept
                depth_obs = ((depth_obs * (self.MAX_DEPTH - self.MIN_DEPTH)) + self.MIN_DEPTH) * 1000.0

                rgbd_frame = [rgb_obs, depth_obs]
                rgbd_frame = torch.cat(rgbd_frame, dim=-1).permute(0, 3, 1, 2)
                batch_end_time = time.time() - st_time

                st_time = time.time()
                semantic_frame = self.predictor(rgbd_frame)
                inf_end_time = time.time() - st_time

                st_time = time.time()
                semantic_frame = semantic_frame.detach().softmax(dim=1)
                x = (torch.max(semantic_frame, dim=1)[1]).float()

                # Mask out categories not used in rednet training to background
                if self.only_rednet_cats:
                    x = self.category_map[x.long()].float()
                softmax_time = time.time() - st_time
        else:
            x = self.predictor(rgb_obs, depth_obs)
        # logger.info("bef ret shape: {}, {}".format(x.shape, x.dtype))
        return x, batch_end_time, inf_end_time, softmax_time
