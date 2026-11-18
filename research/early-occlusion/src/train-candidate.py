"""Fine-tune the three-score V18 model on a user-supplied, session-split dataset."""

import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.loss import v8PoseLoss
from ultralytics.utils.tal import TaskAlignedAssigner

from protocol import ROOT


class PresenceAssigner(TaskAlignedAssigner):
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        player_scores = pd_scores.amax(dim=-1, keepdim=True).expand_as(pd_scores)
        return super().forward(player_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)


class PresenceBCE(nn.BCEWithLogitsLoss):
    def forward(self, logits, targets):
        teams = targets[..., :2]
        return super().forward(logits, torch.cat((teams, teams.amax(dim=-1, keepdim=True)), dim=-1))


class PresencePoseLoss(v8PoseLoss):
    def __init__(self, model):
        super().__init__(model)
        base = self.assigner
        self.assigner = PresenceAssigner(topk=base.topk, topk2=base.topk2, num_classes=self.nc,
                                         alpha=base.alpha, beta=base.beta, eps=base.eps, stride=base.stride)
        self.bce = PresenceBCE(reduction='none')


class PresenceTrainer(PoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None or weights.model[-1].nc != 3:
            raise ValueError('Expected three-score pose weights')
        model = PoseModel(deepcopy(weights.yaml), nc=3, ch=3,
                          data_kpt_shape=weights.kpt_shape, verbose=verbose)
        model.load_state_dict(weights.float().state_dict(), strict=True)
        model.names = dict(weights.names)
        return model


BASE_POSTPROCESS = PoseValidator.postprocess


def init_criterion(model):
    return PresencePoseLoss(model)


def team_only_validation(validator, predictions):
    rows = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
    if rows.shape[1] == 10:
        rows = torch.cat((rows[:, :6], rows[:, 7:]), dim=1)
    return BASE_POSTPROCESS(validator, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_yaml', type=Path)
    parser.add_argument('--weights', type=Path, default=ROOT/'checkpoints/v18.pt')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=3)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    with (patch.object(PoseModel, 'init_criterion', init_criterion),
          patch.object(PoseValidator, 'postprocess', team_only_validation)):
        YOLO(str(args.weights.resolve()), task='pose').train(
            trainer=PresenceTrainer, data=str(args.data_yaml.resolve()), project=str(args.project.resolve()),
            name='presence-candidate', epochs=args.epochs, patience=args.epochs, batch=args.batch, imgsz=1280,
            device=args.device, workers=0, freeze=10, optimizer='AdamW', cls_pw=0., lr0=.00004, lrf=.3,
            warmup_epochs=1., warmup_bias_lr=.0002, amp=args.device!='cpu', cache=False, save_period=2,
            mosaic=0., mixup=0., close_mosaic=0, scale=.08, translate=.04, fliplr=0., hsv_s=.25, hsv_v=.18, seed=42,
        )


if __name__ == '__main__':
    main()
