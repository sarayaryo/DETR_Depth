def _print_fusion_params(model):
    import torch
    import numpy as np
    """Share-Fusionパラメータ(alpha, beta)の平均値を1行でコンパクトに表示する"""
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module
    
    # Transformerエンコーダ層へのアクセス
    transformer = getattr(model, 'transformer', None)
    if transformer is None: return
    encoder = getattr(transformer, 'encoder', None)
    if encoder is None: return
    layers = getattr(encoder, 'layers', [])

    alpha_vals = []
    beta_vals = []

    for layer in layers:
        attn = getattr(layer, 'self_attn', None)
        if attn is None: continue

        # Alpha値の取得 (学習可能ならsigmoidを通す)
        if hasattr(attn, 'alpha'):
            val = attn.alpha
            if isinstance(val, (torch.nn.Parameter, torch.Tensor)):
                alpha_vals.append(torch.sigmoid(val).item())
            else:
                alpha_vals.append(val)

        # Beta値の取得
        if hasattr(attn, 'beta'):
            val = attn.beta
            if isinstance(val, (torch.nn.Parameter, torch.Tensor)):
                beta_vals.append(torch.sigmoid(val).item())
            else:
                beta_vals.append(val)

    # 1行で表示
    msgs = []
    if alpha_vals:
        avg_alpha = sum(alpha_vals) / len(alpha_vals)
        msgs.append(f"Avg Alpha: {avg_alpha:.3f}")
    if beta_vals:
        avg_beta = sum(beta_vals) / len(beta_vals)
        msgs.append(f"Avg Beta: {avg_beta:.3f}")
    
    if msgs:
        print(f" [Fusion] {' | '.join(msgs)} (over {len(alpha_vals)} layers)")

def print_detailed_param_status(model):
    print("\n" + "="*90)
    print(f"{'Parameter Name':<60} | {'Status':<10} | {'Shape':<15}")
    print("-" * 90)
    
    trainable_count = 0
    frozen_count = 0
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        # パラメータ数をカウント
        n_p = param.numel()
        total_params += n_p
        
        # ステータス判定
        if param.requires_grad:
            status = "TRAINABLE"
            trainable_count += 1
            trainable_params += n_p
            # 色付きで見やすく（Jupyterや一部ターミナル用）
            status_str = f"\033[92m{status}\033[0m" 
        else:
            status = "FROZEN"
            frozen_count += 1
            status_str = f"\033[90m{status}\033[0m" # グレーアウト
            
        print(f"{name:<60} | {status_str:<10} | {str(list(param.shape)):<15}")

    print("-" * 90)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Trainable Layers: {trainable_count}")
    print(f"Frozen Layers   : {frozen_count}")
    print("="*90 + "\n")

def print_simplified_param_status(model):
    print("\n" + "="*80)
    print(f"{'Module Name':<40} | {'Status':<15} | {'Details'}")
    print("-" * 80)
    
    # チェックするモジュールのリスト
    modules_to_check = [
        ('Backbone (ResNet)', model.backbone),
        ('Input Proj (RGB)', model.input_proj),
        ('Input Proj (Depth)', getattr(model, 'input_proj_depth', None)),
        ('Transformer Encoder', model.transformer.encoder),
        ('Transformer Decoder', model.transformer.decoder),
        ('Class/BBox Head', model.class_embed), # bbox_embed含む
    ]

    total_trainable_params = 0

    for name, module in modules_to_check:
        if module is None: continue
        
        trainable = 0
        frozen = 0
        total = 0
        
        for param in module.parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
            else:
                frozen += n
        
        # ステータス判定
        if trainable == 0 and total > 0:
            status = "\033[90mALL FROZEN\033[0m" # グレー
        elif trainable == total:
            status = "\033[92mALL TRAINABLE\033[0m" # 緑
        else:
            status = "\033[93mPARTIAL\033[0m" # 黄色（一部学習）

        # 詳細（パラメータ数）
        details = f"{trainable:,} / {total:,} params"
        print(f"{name:<40} | {status:<15} | {details}")
        total_trainable_params += trainable

    print("-" * 80)
    print(f"Total Trainable Params: {total_trainable_params:,}")
    print("="*80 + "\n")

class TerminalLogger(object):
    def __init__(self, filename):
        import sys
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8') # 追記モード

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # すぐにファイルに書き込む

    def flush(self):
        # python 3 compatibility
        self.terminal.flush()
        self.log.flush()
