# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import warnings
# torchvisionのpretrainedに関する警告を無視
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

import numpy as np
import sys
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from custom_module import print_detailed_param_status, print_simplified_param_status, TerminalLogger, _print_fusion_params
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models.backbone import print_parameter_status

### changes here - EarlyStoppingクラスの追加
class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -100000
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        '''Saves model when validation score increase.'''
        if self.verbose:
            self.trace_func(f'Validation score increased ({self.val_score_max:.6f} --> {val_score:.6f}).  Saving model ...')
        
        # モデルの保存（DDP対応）
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            torch.save({'model': model.module.state_dict()}, self.path)
        else:
            torch.save({'model': model.state_dict()}, self.path)
            
        self.val_score_max = val_score


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    ## changes here
    parser.add_argument('--lr_depth_encoder', default=1e-4, type=float,
                        help="Learning rate for depth encoder")
    parser.add_argument('--depth_path', type=str, default=None,
                        help='path to depth images folder')
    parser.add_argument('--use_depth', action='store_true',
                        help='use depth images for training')
    parser.add_argument('--debug', action='store_true',
                        help='use a small subset of the dataset for debugging')
    parser.add_argument('--val_split', action='store_true',
                        help='use split validation set (train: 4000, test: 1000)')
    parser.add_argument('--patience', default=5, type=int,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--use_sharefusion', action='store_true',
                        help='Use ShareFusion architecture for RGB-D fusion')
    parser.add_argument('--use_ar_fusion', action='store_true',
                        help='Use AR-Fusion method for attention fusion in ShareFusion modules')
    parser.add_argument('--use_learnable_param', action='store_true',
                        help='Use learnable fusion weights in ShareFusion modules')

    return parser


def main(args):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    head_name = "class_embed"

    param_dicts = [
        # 1. other parameters
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                       if "backbone" not in n 
                       and "_depth" not in n 
                       and "input_proj" not in n 
                       and head_name not in n
                       and "alpha" not in n  
                       and "beta" not in n   
                       and p.requires_grad],
            "lr": args.lr
        },
        # 2. Backbone
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                       if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        # 3. Depth
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                       if ("_depth" in n or "input_proj_depth" in n) and p.requires_grad],
            "lr": args.lr * 1.0,
        },
        # 4. Class Head
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                       if head_name in n and p.requires_grad],
            "lr": args.lr * 10.0, 
        },
        # 5. fusion parameters
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                       if ("alpha" in n or "beta" in n) and p.requires_grad],
            "lr": args.lr * 10.0, 
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    data_loader_test = build_dataset(image_set='test', args=args) if args.eval else None

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_test = DistributedSampler(dataset_val, shuffle=False) if data_loader_test is not None else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_val) if data_loader_test is not None else None

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_val, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) if data_loader_test is not None else None

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        # model_without_ddp.load_state_dict(checkpoint['model'])

        pretrained_dict = checkpoint['model']

        if args.use_sharefusion:

            model_dict = model_without_ddp.state_dict()
            # ---------------------------------------------------------
            # Checkpointの "self_attn.xxx" -> "self_attn.rgb_attn.xxx" に変換
            # ---------------------------------------------------------
            new_state_dict = {}
            renamed_count = 0
            for k, v in pretrained_dict.items():
                # EncoderのSelf-Attention部分のみ変換対象
                if "transformer.encoder.layers" in k and "self_attn" in k:
                    # 例: ...self_attn.in_proj_weight -> ...self_attn.rgb_attn.in_proj_weight
                    # ※ Decoderは構造を変えていないので変換してはいけない
                    if "rgb_attn" not in k: # すでに変換済みでなければ
                        new_key = k.replace("self_attn.", "self_attn.rgb_attn.")
                        # print(k, "->", new_key)
                        new_state_dict[new_key] = v
                        renamed_count += 1
                    else:
                        new_state_dict[k] = v
                else:
                    # それ以外（Backbone, Decoder, Input_projなど）はそのまま
                    new_state_dict[k] = v
            print(f"Renamed {renamed_count} keys for RGB-Stream compatibility.")

            # ---------------------------------------------------------
            # 3. 重みのロード (RGBストリームの復元)
            # ---------------------------------------------------------
            # strict=False は必須 (Depth側の重みがまだ無いため)
            missing, unexpected = model_without_ddp.load_state_dict(new_state_dict, strict=False)
            
            # 重要なログ確認: RGB側の主要パーツが "missing" になっていないか？
            rgb_missing = [k for k in missing if "rgb_attn" in k or "backbone" in k]
            if len(rgb_missing) > 0:
                print("\n[WARNING] Some RGB weights were NOT loaded! Check these keys:")
                for k in rgb_missing[:5]: print(f" - {k}")
            else:
                print(">>> RGB Stream weights loaded successfully.")

            # ---------------------------------------------------------
            # 4. Depthストリームの初期化 (Transfer Learning)
            #    ロードされたばかりの rgb_attn の重みを depth_attn にコピー
            # ---------------------------------------------------------
            if args.use_depth:
                print(">>> Initializing Depth Stream from RGB weights...")
                
                encoder = model_without_ddp.transformer.encoder
                for i, layer in enumerate(encoder.layers):
                    # Attentionのコピー (rgb_attn -> depth_attn)
                    if hasattr(layer.self_attn, "rgb_attn") and hasattr(layer.self_attn, "depth_attn"):
                        # in_proj_weight / bias
                        if hasattr(layer.self_attn.rgb_attn, "in_proj_weight"):
                            layer.self_attn.depth_attn.in_proj_weight.data.copy_(
                                layer.self_attn.rgb_attn.in_proj_weight.data
                            )
                            layer.self_attn.depth_attn.in_proj_bias.data.copy_(
                                layer.self_attn.rgb_attn.in_proj_bias.data
                            )
                        
                        # out_proj.weight / bias
                        layer.self_attn.depth_attn.out_proj.weight.data.copy_(
                            layer.self_attn.rgb_attn.out_proj.weight.data
                        )
                        layer.self_attn.depth_attn.out_proj.bias.data.copy_(
                            layer.self_attn.rgb_attn.out_proj.bias.data
                        )
                    
                    # FFN (Feed Forward) のコピー (linear1, linear2 -> linear1_depth, linear2_depth)
                    # ※ sharefusion.py の EncoderLayer_RGBD 定義に基づく
                    layer.linear1_depth.weight.data.copy_(layer.linear1.weight.data)
                    layer.linear1_depth.bias.data.copy_(layer.linear1.bias.data)
                    layer.linear2_depth.weight.data.copy_(layer.linear2.weight.data)
                    layer.linear2_depth.bias.data.copy_(layer.linear2.bias.data)

                    # Norm Layers のコピー
                    layer.norm1_depth.weight.data.copy_(layer.norm1.weight.data)
                    layer.norm1_depth.bias.data.copy_(layer.norm1.bias.data)
                    layer.norm2_depth.weight.data.copy_(layer.norm2.weight.data)
                    layer.norm2_depth.bias.data.copy_(layer.norm2.bias.data)

                # input_proj (CNN -> Transformerのつなぎ) のコピー
                # ※ detr.py で定義されているはず
                if hasattr(model_without_ddp, "input_proj_depth"):
                    model_without_ddp.input_proj_depth.weight.data.copy_(
                        model_without_ddp.input_proj.weight.data
                    )
                    model_without_ddp.input_proj_depth.bias.data.copy_(
                        model_without_ddp.input_proj.bias.data
                    )

                print(">>> Depth Stream initialized.")
                print("="*60 + "\n")

        else:
            print(">>> Mode: Standard DETR (Loading directly)")
            # 名前変換なしでそのままロード
            # strict=Falseにしておく（input_proj_depthなど余計なものがあるかもしれないため）
            missing, unexpected = model_without_ddp.load_state_dict(pretrained_dict, strict=False)
            
            # 本来あるべきRGBの重みが消えてないかチェック
            if len(missing) > 0:
                print(f"    Missing keys: {len(missing)}")
                # ここで 'self_attn' 系がMissingになっていなければ成功
            else:
                print(">>> All weights loaded successfully.")
        

        # パラメータを固定
        for param in model_without_ddp.input_proj.parameters():
            param.requires_grad = False
        for param in model_without_ddp.backbone.parameters():
            param.requires_grad = False
        for param in model_without_ddp.transformer.decoder.parameters():
            param.requires_grad = False
        for name, param in model_without_ddp.transformer.encoder.named_parameters():
            # Depth関連のキーワードが含まれているかチェック
            is_depth_param = ("_depth" in name) or ("depth_attn" in name)
            is_fusion = ("alpha" in name) or ("beta" in name)

            if is_depth_param or is_fusion:
                # Depth用 or fusion parameter -> 学習 (Trainable)
                param.requires_grad = True
            else:
                # RGB用 (linear1, norm1, rgb_attnなど) -> 固定 (Frozen)
                param.requires_grad = False

            
    print_simplified_param_status(model_without_ddp)
    print_parameter_status(model_without_ddp)

        
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=str(Path(args.output_dir) / 'best_model.pth'))

    print("Start training")
    start_time = time.time()
    if args.use_learnable_param:
        _print_fusion_params(model_without_ddp)
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, args=args)                                                                                                                 
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        ## changes here
        test_stats = {}
        coco_evaluator = None
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, epoch=epoch, args=args
            )
            if coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
                # mAP (IoU=0.50:0.95) を監視対象にする
                val_mAP = coco_evaluator.coco_eval['bbox'].stats[0]
                
                # ベストモデル更新チェック & ストップ判定
                early_stopping(val_mAP, model_without_ddp)
                
                if early_stopping.early_stop:
                    print("Early stopping triggered!")
                    break

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    
    if data_loader_test is not None:
        print("Start testing on test set")
        test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_test, base_ds, device, args.output_dir, epoch=epoch
                )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)       
        log_file_path = os.path.join(args.output_dir, 'terminal_log.txt')
        sys.stdout = TerminalLogger(log_file_path)
        sys.stderr = TerminalLogger(log_file_path) 
        
        print(f"Logging to: {log_file_path}")
    main(args)
