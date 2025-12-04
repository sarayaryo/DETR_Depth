import argparse, os, sys
from PIL import Image
import torch
import matplotlib.pyplot as plt

# === あなたのDETRクローンを import パスに追加 ===
DETR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DETR_DIR)

from models import build_model
from datasets import transforms as T
from util import box_ops

# ---- COCOのクラス（公式ノートの定義に合わせて N/A を含む91要素）----
CLASSES = [
    'N/A','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','N/A','stop sign','parking meter','bench','bird','cat','dog','horse',
    'sheep','cow','elephant','bear','zebra','giraffe','N/A','backpack','umbrella','N/A',
    'N/A','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
    'baseball glove','skateboard','surfboard','tennis racket','bottle','N/A','wine glass','cup','fork','knife',
    'spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','N/A','dining table','N/A','N/A',
    'toilet','N/A','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','N/A','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush'
]

# ---- 描画色（matplotlib のタブカラーパレットを繰返し利用）----
COLORS = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])

from argparse import Namespace
import numpy as np
from scipy.ndimage import zoom

def visualize_attention(image, attention_weights, layer_idx, query_idx, results=None, out_path=None):
    """
    特定のObject QueryのAttention Mapを可視化
    
    Parameters:
        image: PIL Image (元画像)   
        attention_weights: tensor [Layers, Batch, Queries, H, W] 
                           (例: torch.Size([6, 1, 100, 20, 15])
        query_idx: 可視化したいクエリのインデックス (0-99)
        layer_idx: 可視化したいレイヤー (-1で最終層)
        out_path: 保存先パス
    """
    # 特定レイヤー・クエリのAttentionを取得
    print(f"attention_weights shape: {attention_weights.shape}")
    attn = attention_weights[0, query_idx].cpu().numpy()  # [H*W]
    print(f"Visualizing Attention for Query {query_idx} at Layer {layer_idx} - Attention Shape: {attn.shape}")
    
    # 特徴マップのサイズを推定 (通常 H/32, W/32)
    h = int(attn.shape[0])
    w = int(attn.shape[1])

    print(f"Reshaping attention to ({h}, {w})")
    attn = attn.reshape(h, w)
    
    # 画像サイズにリサイズ
    img_w, img_h = image.size
    zoom_factor = (img_h / h, img_w / w)
    attn_resized = zoom(attn, zoom_factor, order=1)
    
    # 正規化 [0, 1]
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 元画像
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image')
    axes[0].axis('off')
    if results is not None:
        # ロジットとボックスを取得 (Batch index 0 を指定)
        prob = results['pred_logits'][0, query_idx].softmax(-1)
        top_score, top_label = prob[:-1].max(0) # 背景クラスを除く最大値
        box = results['pred_boxes'][0, query_idx]

        # 座標変換 (cx, cy, w, h) -> (xmin, ymin, w, h) absolute
        cx, cy, bw, bh = box.cpu().detach().numpy()
        cx, cy, bw, bh = cx * img_w, cy * img_h, bw * img_w, bh * img_h
        xmin = cx - 0.5 * bw
        ymin = cy - 0.5 * bh

        import matplotlib.patches as patches
        
        # BBoxの作成 (赤枠)
        rect = patches.Rectangle((xmin, ymin), bw, bh, linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        
        # ラベルとスコアの表示
        label_text = f"{CLASSES[top_label.item()]}: {top_score.item():.2f}"
        axes[0].text(xmin, ymin - 5, label_text, color='white', fontsize=12,
                     bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))
    
    # Attention Map
    axes[1].imshow(image)
    axes[1].imshow(attn_resized, cmap='jet', alpha=0.6, interpolation='nearest')
    axes[1].set_title(f'Attention Map (Query {query_idx}, Layer {layer_idx})')
    axes[1].axis('off')
    
    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention map: {out_path}")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import cv2  # OpenCVを使用 (pip install opencv-python)
import math

def visualize_multi_layer_attention(image, attention_weights, query_idx, results=None, out_path=None):
    """
    全レイヤーのAttentionを一度に可視化（ピクセル状スタイル）
    
    Parameters:
        image: PIL Image
        attention_weights: tensor [Batch, Queries, H, W] 
        query_idx: クエリインデックス
        out_path: 保存先パス
    """
    # -------------------------------------------------------
    # 1. グリッド設定 (レイヤー数に応じて行・列を動的に決める)
    # -------------------------------------------------------
    num_layers = attention_weights.shape[0]
    
    # 例: 6層なら 2行3列, 1層なら 1行1列
    ncols = 3
    nrows = math.ceil(num_layers / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8))
    
    # axesが1つの場合や1次元の場合でもループできるように平坦化
    if num_layers == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    # -------------------------------------------------------
    # 2. 各レイヤーの可視化
    # -------------------------------------------------------
    img_w, img_h = image.size
    print(f"attention_weights shape: {attention_weights.shape}")

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        # [Layers, Batch, Queries, H, W] -> Batchインデックス0を指定
        attn = attention_weights[layer_idx, 0, query_idx].cpu().detach().numpy()
        print(f"attention_weights shape: {attn.shape}")
        
        # 特徴マップのサイズ取得 (20x15など、長方形でもOK)
        h = int(attn.shape[0])
        w = int(attn.shape[1])
        
        img_w, img_h = image.size
        zoom_factor = (img_h / h, img_w / w)
        attn_resized = zoom(attn, zoom_factor, order=1)
        
        # 正規化 [0, 1]
        attn_min, attn_max = attn_resized.min(), attn_resized.max()
        attn_resized = (attn_resized - attn_min) / (attn_max - attn_min + 1e-8)
        
        # 描画
        ax.imshow(image)
        # interpolation='nearest' で描画
        ax.imshow(attn_resized, cmap='jet', alpha=0.6, interpolation='nearest')
        
        ax.set_title(f'Layer {layer_idx}')
        ax.axis('off')

    # 余ったサブプロット（空白）を非表示にする
    for j in range(num_layers, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-layer attention: {out_path}")
    
    plt.show()

def load_model(weights_path: str, device: str = 'cuda'):
    # 公式 main.py のデフォルトを踏襲（DETR-R50）
    args = Namespace(
        # 環境
        device=device,
        dataset_file='coco',

        # Backbone / Transformer 形状
        backbone='resnet50',
        dilation=False,
        position_embedding='sine',
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        dim_feedforward=2048,
        enc_layers=6,
        dec_layers=6,
        lr_backbone=1e-5,
        pre_norm=False,

        # DETR クエリなど
        num_queries=100,
        aux_loss=False,   # 推論だけなら False でOK（学習時の補助出力を使わない）
        masks=False,      # セグメントしない

        # Matcher & 損失（推論では使わないが build() が参照）
        set_cost_class=1,
        set_cost_bbox=5,
        set_cost_giou=2,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        eos_coef=0.1,
        mask_loss_coef=1,
        dice_loss_coef=1,

        # その他
        frozen_weights=None
    )

    model, _, postprocessors = build_model(args)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device).eval()
    return model, postprocessors


def get_transform():
    # 公式ノートと同じ前処理
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

def rescale_bboxes(out_bbox: torch.Tensor, size):
    # (cx,cy,w,h) -> (xmin,ymin,xmax,ymax) へ変換し、画像サイズにスケール
    img_w, img_h = size
    b = box_ops.box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b * scale

def plot_results(pil_img, prob, boxes, out_path=None, score_thr=0.7):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    # prob: [N, num_classes], boxes: [N, 4] (xyxy)
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        cl = p.argmax().item()
        score = p[cl].item()
        if score < score_thr:
            continue
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{CLASSES[cl]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    plt.show()

def run(img_path: str, weights_path: str, out_path: str = None, device='cuda', score_thr=0.7, visualize_attn=False):
    # 画像読み込み
    im = Image.open(img_path).convert('RGB')
    
    max_side = 640
    w, h = im.size
    scale = max_side / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    im = im.resize(new_size, Image.Resampling.LANCZOS)

    # モデル準備
    model, postprocessors = load_model(weights_path, device=device)

    # 前処理
    transform = get_transform()
    tensor, _ = transform(im, None)

    # 推論
    with torch.no_grad():
        outputs = model([tensor.to(device)])

    # ソフトマックスでクラス確率（背景クラスは除外）
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # バウンディングボックスを画像座標へ
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0].cpu(), im.size)

    ## Attention
    attention_weights = outputs['decoder_attention']

    print("=== Attention Weights Info ===")
    print(outputs['decoder_attention'].shape)  # デバッグ用

    # 可視化
    plot_results(im, probas.cpu(), bboxes_scaled.cpu(), out_path=out_path, score_thr=score_thr)

    ## choose layer to visualize -1 -? 5
    layer_idx = 1

        # ===== Attention Weights の取り出し =====
    if visualize_attn and 'decoder_attention' in outputs:
        attention_weights_layer = outputs['decoder_attention'][layer_idx]  # [num_layers, num_queries, H*W]
        
        # 検出スコアが高いクエリTop3を可視化
        top_scores, top_indices = probas.max(-1).values.topk(3)
        
        print("\n=== Top 3 Detected Objects ===")
        for rank, (score, idx) in enumerate(zip(top_scores, top_indices)):
            cls_id = probas[idx].argmax().item()
            print(f"Rank {rank+1}: Query {idx.item()} - {CLASSES[cls_id]} (score: {score:.3f})")
            
            # 各クエリのAttention Map可視化
            attn_out = out_path.replace('.png', f'_attn_query{idx.item()}.png') if out_path else None
            visualize_attention(im, attention_weights_layer, layer_idx, idx.item(), results=outputs, out_path=attn_out)
            
            # 全レイヤーの可視化（オプション）
            multi_out = out_path.replace('.png', f'_attn_multilayer_query{idx.item()}.png') if out_path else None
            visualize_multi_layer_attention(im, attention_weights, idx.item(), results=outputs, out_path=multi_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="入力画像（COCO val2017 の *.jpg 推奨）")
    parser.add_argument("--weights", required=True, help="detr-r50 の .pth")
    parser.add_argument("--out", default="vis_pred.png", help="保存先パス（PNG）")
    parser.add_argument("--cpu", action="store_true", help="CPUで実行（デバッグ用）")
    parser.add_argument("--thr", type=float, default=0.7, help="描画する最小スコア閾値")
    parser.add_argument("--visualize", action="store_true", help="Attentionマップを可視化")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run(args.img, args.weights, out_path=args.out, device=device, score_thr=args.thr, visualize_attn=args.visualize)
