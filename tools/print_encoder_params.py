#!/usr/bin/env python3
import os
import importlib.util

# transformer.py をパッケージ初期化を起こさず直接読み込む
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
transformer_path = os.path.join(project_root, 'detr', 'models', 'transformer.py')
spec = importlib.util.spec_from_file_location('tr_mod', transformer_path)
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
Transformer = tr.Transformer


def main():
    model = Transformer()
    total = 0
    trainable = 0

    print("Parameters of model.encoder:")
    for name, p in model.encoder.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
        print(f"{name}: {tuple(p.size())}, requires_grad={p.requires_grad}, numel={num}")

    print(f"Total params: {total}, Trainable params: {trainable}")


if __name__ == '__main__':
    main()
