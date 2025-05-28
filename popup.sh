#对数据集添加弹窗
# 使用默认设置处理所有数据集
python popup.py

# 仅处理web数据集，添加80%的弹窗
python popup.py --dataset web --percentage 0.8

# 每个数据集快速测试10张图像
python popup.py --max 10