1. how to run visualize.py
python visualize.py --img datasets\test\humans.jpg --weights weights\detr-r50-e632da11.pth --out result.png --thr 0.7 
## if visualize attentionMap
python visualize.py --img datasets\test\humans.jpg --weights weights\detr-r50-e632da11.pth --out result.png --thr 0.7 --visualize

python main.py --coco_path "S:/coco/coco2017" --depth_path "S:/coco/coco2017_depth" --use_depth --resume "../weights/detr-r50-e632da11.pth" --output_dir outputs/sanity_check --epochs 10 --lr 1e-4 --lr_backbone 1e-5 --batch_size 16 --num_workers 0 --debug --val_split

### debugを消し，本格学習
python main.py --coco_path "S:/coco/coco2017" --depth_path "S:/coco/coco2017_depth" --use_depth --resume "../weights/detr-r50-e632da11.pth" --output_dir outputs/depth-30_bs-8 --epochs 30 --lr 1e-4 --lr_backbone 1e-5 --batch_size 4 --num_workers 0  --val_split


### サーバー上で動かすとき(/workspace/DETR, coco想定)
python DETR/detr/main.py --coco_path "/workspace/coco2017" --depth_path "/workspace/Dataset/coco2017_depth" --use_depth --resume "/workspace/DETR/weights/detr-r50-e632da11.pth" --output_dir outputs/depth-30_bs-8 --epochs 30 --lr 1e-4 --lr_backbone 1e-5 --batch_size 4 --num_workers 0  --val_split

### sharefusion
--use_sharefusion True

##最新版
python DETR/detr/main.py --coco_path "/workspace/coco/coco2017" --depth_path "/workspace/coco/coco2017_depth" --use_depth --resume "/workspace/DETR/weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputsoutputs/[TEST]sharefusion_alpha0.5_beta0.5_ep50_bs4 --epochs 50 --lr 1e-4 --lr_backbone 1e-5 --batch_size 4 --num_workers 8  --val_split --use_sharefusion True --device cuda

##ローカルテスト用
python detr/main.py --coco_path "S:/coco/coco2017" --depth_path "S:/coco/coco2017_depth" --use_depth --resume "weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputs/test --debug --epochs 10 --lr 1e-4 --lr_backbone 1e-5 --batch_size 2 --num_workers 0  --val_split --use_sharefusion True
