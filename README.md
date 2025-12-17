1. how to run visualize.py
python visualize.py --img datasets\test\humans.jpg --weights weights\detr-r50-e632da11.pth --out result.png --thr 0.7 
## if visualize attentionMap
python visualize.py --img datasets\test\humans.jpg --weights weights\detr-r50-e632da11.pth --out result.png --thr 0.7 --visualize

python main.py --coco_path "S:/coco/coco2017" --depth_path "S:/coco/coco2017_depth" --use_depth --resume "../weights/detr-r50-e632da11.pth" --output_dir outputs/sanity_check --epochs 10 --lr 1e-4 --lr_backbone 1e-5 --batch_size 16 --num_workers 0 --debug --val_split