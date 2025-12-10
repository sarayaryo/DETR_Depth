1. how to run visualize.py
python visualize.py --img datasets\test\humans.jpg --weights weights\detr-r50-e632da11.pth --out result.png --thr 0.7 
## if visualize attentionMap
python visualize.py --img datasets\test\humans.jpg --weights weights\detr-r50-e632da11.pth --out result.png --thr 0.7 --visualize