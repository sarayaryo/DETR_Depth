import argparse
import torch
from pathlib import Path
import json
import sys
import os
from datetime import datetime

from datasets import build_dataset
from models import build_model
from engine import evaluate
import util.misc as utils
from torch.utils.data import DataLoader


class TestLogger:
    """ターミナルとログファイルの両方に出力するクラス"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def get_args_parser():
    parser = argparse.ArgumentParser('DETR Test Script', add_help=False)
    
    # モデル設定（学習時と同じ設定を使用）
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    
    # データセット設定
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--depth_path', type=str, default=None)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='推論時のバッチサイズ（学習時より大きくても可）')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--val_split', action='store_true',
                        help='use split validation set (train: 4000, val: 500, test: 500)')
    parser.add_argument('--debug', action='store_true',
                        help='use a small subset of the dataset for debugging')
    
    # RGB-D設定（学習時と同じ設定を使用）
    parser.add_argument('--use_depth', action='store_true')
    parser.add_argument('--use_sharefusion', action='store_true')
    parser.add_argument('--use_ar_fusion', action='store_true')
    parser.add_argument('--use_learnable_param', action='store_true')
    
    # 推論設定
    parser.add_argument('--weights', type=str, required=True,
                        help='学習済みモデルのパス (e.g., best_model.pth)')
    parser.add_argument('--output_dir', default='./test_results',
                        help='結果の保存先')
    parser.add_argument('--device', default='cuda',
                        help='使用デバイス')
    
    # Loss係数（評価時は不要だが、build_model内で参照される）
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--aux_loss', action='store_false')
    parser.add_argument('--frozen_weights', type=str, default=None)
    parser.add_argument('exp_name', type=str, nargs='?', default='test')
    
    return parser


def main(args):
    device = torch.device(args.device)
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ログファイルの設定
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f'test_log_{args.exp_name}.txt'
    logger = TestLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    # 詳細出力は `detailed_prints()` にまとめました（呼び出さない限り出力されません）

    def detailed_prints(model_info=None, dataset_info=None, load_info=None, eval_info=None):
        # ここに元の詳細出力を入れておく（将来的に呼び出して利用可能）
        if model_info is not None:
            print(model_info)
        if load_info is not None:
            print(load_info)
        if dataset_info is not None:
            print(dataset_info)
        if eval_info is not None:
            print(eval_info)
    
    # モデルの構築
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # 学習済み重みの読み込み
    # Loading weights (detailed message moved to detailed_prints)
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        # Weights loaded (details available via `detailed_prints`)
    else:
        print("Warning: 'model' key not found in checkpoint, loading directly...")
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    # Building test dataset (progress output suppressed)
    # main.pyと同じように、valセットを構築してからtestとして使用
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='val', args=args)
    
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(
        dataset_test,
        args.batch_size,
        sampler=sampler_test,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers
    )
    
    from datasets import get_coco_api_from_dataset
    # base_dsはdataset_valから取得（main.pyと同じ）
    base_ds = get_coco_api_from_dataset(dataset_val)
    
    # Test dataset loaded (details available via `detailed_prints`)
    
    # 推論実行
    # Running inference on test dataset (progress output suppressed)
    
    test_stats, coco_evaluator = evaluate(
        model, criterion, postprocessors,
        data_loader_test, base_ds, device,
        args.output_dir
    )
    
    # Only output the concise AP summary line to terminal and log
    if coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
        stats = coco_evaluator.coco_eval['bbox'].stats
        summary_line = (f"AP: {stats[0]:.4f} | AP50: {stats[1]:.4f} | AP75: {stats[2]:.4f} | "
                        f"APS: {stats[3]:.4f} | APM: {stats[4]:.4f} | APL: {stats[5]:.4f}")
        print(summary_line)
        
        # 結果をJSONで保存
        results_dict = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'weights': args.weights,
            'dataset': args.coco_path,
            'batch_size': args.batch_size,
            'num_images': len(dataset_test),
            'metrics': {
                'AP': float(stats[0]),
                'AP50': float(stats[1]),
                'AP75': float(stats[2]),
                'APS': float(stats[3]),
                'APM': float(stats[4]),
                'APL': float(stats[5]),
                'AR1': float(stats[6]),
                'AR10': float(stats[7]),
                'AR100': float(stats[8]),
                'ARS': float(stats[9]),
                'ARM': float(stats[10]),
                'ARL': float(stats[11])
            },
            'full_stats': test_stats
        }
        
        results_file = output_dir / f'test_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
    else:
        # If evaluator missing, do not print verbose warnings

        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DETR Test Inference',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)