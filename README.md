# 简介
此分支为 [SpecularityNet-PSD](https://github.com/jianweiguo/SpecularityNet-PSD) 的torch更新

# 依赖
示例 `pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple`
```
opencv-python
tensorboardX
scikit-image
dominate
matplotlib
```

# 训练
```sh
CUDA_VISIBLE_DEVICES=1 python train_specularitynet.py --name refined --inet refined --iters 1 --suffix iters1 --enhance de --freq 0.25 --noise True --lambda_coarse 0.5 --lambda_detect 1.0 --batchSize 4 --nThreads 24 --fliplr 0.5 --flipud 0.5
```

# 测试
```sh
CUDA_VISIBLE_DEVICES=0 python test_specularitynet.py -r --name refined --inet refined --iters 1 --suffix iters1 --enhance de --batchSize 16 --nThreads 32
```