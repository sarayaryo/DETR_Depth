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
python DETR/detr/main.py --coco_path "/workspace/coco/coco2017" --depth_path "/workspace/coco/coco2017_depth" --use_depth --resume "/workspace/DETR/weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputsoutputs/[TEST]sharefusion_alpha0.5_beta0.5_ep50_bs4 --epochs 50 --lr 1e-4 --lr_backbone 1e-5 --batch_size 4 --num_workers 8  --val_split --use_sharefusion --device cuda

##ローカルテスト用
python detr/main.py --coco_path "S:/coco/coco2017" --depth_path "S:/coco/coco2017_depth" --use_depth --resume "weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputs/test --debug --epochs 1 --lr 1e-4 --lr_backbone 1e-5 --batch_size 2 --num_workers 0  --val_split --use_sharefusion

python detr/main.py --coco_path "S:/coco/coco2017" --depth_path "S:/coco/coco2017_depth" --use_depth --resume "weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputs/test --debug --epochs 1 --lr 1e-4 --lr_backbone 1e-5 --batch_size 2 --num_workers 0  --val_split  --use_learnable_param

##ローカルテスト用 RGBのみ
python detr/main.py --coco_path "S:/coco/coco2017" --depth_path "S:/coco/coco2017_depth" --resume "weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputs/test --debug --epochs 1 --lr 1e-4 --lr_backbone 1e-5 --batch_size 2 --num_workers 0  --val_split 

##share-fusion 本番
python DETR/detr/main.py --coco_path "/workspace/coco/coco2017" --depth_path "/workspace/coco/coco2017_depth" --use_depth --resume "/workspace/DETR/weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputs/[PE]sharefusion_alpha0.5_beta0.5_ep50_bs4_dec-frozen --epochs 50 --lr 1e-4 --lr_backbone 1e-5 --batch_size 4 --num_workers 8  --val_split --use_sharefusion --device cuda　


python DETR/detr/main.py --coco_path "/workspace/coco/coco2017" --depth_path "/workspace/coco/coco2017_depth" --use_depth --resume "/workspace/DETR/weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputs/[PE2]sharefusion_alphalearn_betalearn_ep50_bs8-4*2_dec-frozen --epochs 50 --lr 1e-4 --lr_backbone 1e-5 --batch_size 4 --num_workers 8  --val_split --use_sharefusion --device cuda --use_learnable_param

python DETR/detr/main.py --coco_path "/workspace/coco/coco2017" --depth_path "/workspace/coco/coco2017_depth" --use_depth --resume "/workspace/DETR/weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputs/test --epochs 5 --lr 1e-4 --lr_backbone 1e-5 --batch_size 4 --num_workers 8  --val_split --use_sharefusion --device cuda --use_learnable_param --debug

# share-fusion
python DETR/detr/main.py --coco_path "/workspace/coco/coco2017" --depth_path "/workspace/coco/coco2017_depth" --use_depth --resume "/workspace/DETR/weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputs/[PE2]sharefusion_alpha0.0_beta0.5_ep50_bs-4*2_dec-frozen --epochs 50 --lr 1e-4 --lr_backbone 1e-5 --batch_size 4 --num_workers 8  --val_split --use_sharefusion --device cuda 

# alphabeta learn 0.001
python DETR/detr/main.py --coco_path "/workspace/coco/coco2017" --depth_path "/workspace/coco/coco2017_depth" --use_depth --resume "/workspace/DETR/weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputs/[PE3]sharefusion_alphalearn_betalearn_*10_ep50_bs8-4*2_dec-frozen --epochs 50 --lr 1e-4 --lr_backbone 1e-5 --batch_size 4 --num_workers 8  --val_split --use_sharefusion --device cuda --use_learnable_param

## AR-fusion
python DETR/detr/main.py --coco_path "/workspace/coco/coco2017" --depth_path "/workspace/coco/coco2017_depth" --use_depth --resume "/workspace/DETR/weights/detr-r50-e632da11.pth" --output_dir DETR/detr/outputs/[PE3]ARfusion_alphalearn_betalearn_*10_ep50_bs8-4*2_dec-frozen --epochs 50 --lr 1e-4 --lr_backbone 1e-5 --batch_size 4 --num_workers 8  --val_split --use_sharefusion --use_ar_fusion --device cuda --use_learnable_param

# ####################################
# 推論時
# ####################################


# ###### ローカル環境
# バッチサイズ4*2で推論（学習時の実効バッチサイズと同じ）
python test.py --weights "outputs\[PE2]latefusion_alpha0.0_beta0.0_ep50_bs8-4x2_dec-frozen\best_model.pth" --coco_path "S:/coco/coco2017" --depth_path "S:/coco/coco2017_depth" --batch_size 2 --num_workers 0 --use_depth --use_sharefusion --output_dir test_results --val_split --lr_backbone 1e-5 --exp_name latefusion_alpha0.0_beta0.0

--use_sharefusion
--use_ar_fusion --use_learnable_param