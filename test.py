import time
from tqdm import tqdm

train_loader_tqdm = tqdm(
    range(10),
    desc=f'Train Epoch ',
    dynamic_ncols=True,
    bar_format='{l_bar}{bar:20}{r_bar}',
    # leave=False  # 确保进度条完成后被清理
)

for i, data in enumerate(train_loader_tqdm):
    # 模拟训练过程
    time.sleep(0.1)

    # 更新当前指标
    current_metrics = {'IPixel': 0.05, 'VGG': 3.44}
    
    # 动态更新进度条的附加信息（保留4位小数）
    formatted_metrics = {k: round(v, 4) for k, v in current_metrics.items()}
    train_loader_tqdm.set_postfix(formatted_metrics)