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

def run(img_path: str, weights_path: str, out_path: str = None, device='cuda', score_thr=0.7):
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

    # 可視化
    plot_results(im, probas.cpu(), bboxes_scaled.cpu(), out_path=out_path, score_thr=score_thr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="入力画像（COCO val2017 の *.jpg 推奨）")
    parser.add_argument("--weights", required=True, help="detr-r50 の .pth")
    parser.add_argument("--out", default="vis_pred.png", help="保存先パス（PNG）")
    parser.add_argument("--cpu", action="store_true", help="CPUで実行（デバッグ用）")
    parser.add_argument("--thr", type=float, default=0.7, help="描画する最小スコア閾値")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run(args.img, args.weights, out_path=args.out, device=device, score_thr=args.thr)
