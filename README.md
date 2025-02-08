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
tqdm
```

# 训练
```sh
python train_specularitynet.py --name refined --inet refined --iters 1 --suffix iters1 --enhance de --freq 0.25 --noise True --lambda_coarse 0.5 --lambda_detect 1.0 --batchSize 1 --nThreads 2 --fliplr 0.5 --flipud 0.5
```

**显存相关**：调整batchSize
**内存相关**：调整nThreads
**PSD数据集需要修改路径，修改后**：
```sh
├───PSD_Test
│   ├───PSD_Test_diffuse
│   ├───PSD_Test_group
│   └───PSD_Test_specular
├───PSD_Train
│   ├───nospec
│   ├───PSD_Train_group
│   │   ├───01
│   │   ├───aligned
│   │   └───unaligned
│   └───spec
└───PSD_val
    ├───PSD_val_diffuse
    ├───PSD_val_group
    └───PSD_val_specular
```


# 测试
```sh
python test_specularitynet.py -r --name refined --inet refined --iters 1 --suffix iters1 --enhance de --batchSize 2 --nThreads 2
```