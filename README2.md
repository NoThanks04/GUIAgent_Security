# Robust GUI Agent 安全性评估

## 🌟项目概述

本项目旨在对GUI智能体平台进行全面的安全性检测与分析。  


我们设计并使用了多个针对性的高质量数据集，针对不同的GUI智能体模型进行性能评估。


GUIAgent有PLanning和Grouding两个阶段具体工作流如下：

```txt
用户高级指令+图片->planning->输出具体的元素描述

具体的元素描述+图片->grouding输出->精确坐标
```

通过递交特定的数据集和相应的操作指令，我们对模型的决策能力进行测试，评估其在不同攻击情境下的表现。通过计算并分析模型的决策成功率和攻击成功率，使不同模型在不同场景下的表现得以清晰展示，从而帮助我们更好地理解各个模型的安全性和鲁棒性。

## 🌟项目结构

```txt
Robust_GUI_Grounding/
├── DejaVuSansMono-Bold.ttf      # 弹窗生成器所需字体文件
├── README.md                     # 项目说明文档
├── README2.md                    # 额外说明文档
├── create_noisy.py               # 噪声图像生成脚本
├── download.sh                   # 原始数据集下载脚本
├── generation_config.json        # 生成配置文件
├── popup.py                      # 弹窗干扰生成器
├── popup.sh                      # 弹窗生成脚本
├── preprocessor_config.json      # 预处理配置文件
├── requirements.txt              # 项目依赖
│
├── datas/                        # 数据集目录
│   ├── Initial/                  # 原始数据集
│   │   ├── screenspotv2_desktop_ug_target.json  # 桌面目标定位数据
│   │   ├── screenspotv2_image/                 # 原始图像目录
│   │   ├── screenspotv2_mobile_ug_target.json  # 移动目标定位数据
│   │   ├── screenspotv2_web_ug_target.json     # 网页目标定位数据
│   │   ├── screenspot_desktop_v2.json          # 桌面平台数据
│   │   ├── screenspot_mobile_v2.json           # 移动平台数据
│   │   └── screenspot_web_v2.json              # 网页平台数据
│   │
│   └── Noisy/                    # 噪声干扰数据集
│       ├── color_jitter/         # 颜色抖动噪声图像
│       ├── contrast_adjusted/    # 对比度调整噪声图像
│       ├── gaussian_blur/        # 高斯模糊噪声图像
│       ├── gaussian_noise/       # 高斯噪声图像
│       ├── screenspot_desktop_v2.json          # 桌面平台噪声数据标注
│       ├── screenspot_mobile_v2.json           # 移动平台噪声数据标注
│       └── screenspot_web_v2.json              # 网页平台噪声数据标注
│
├── pop_result/                   # 弹窗干扰结果目录
│   ├── image/                    # 添加弹窗后的图像
│   ├── desktop_popup_metadata.json            # 桌面弹窗元数据
│   ├── mobile_popup_metadata.json             # 移动弹窗元数据
│   └── web_popup_metadata.json                # 网页弹窗元数据
│
├── result/                       # 测试结果输出目录
│
└── Uground/                      # Uground模型测试目录
    ├── check.sh                  # 攻击效果评估脚本
    ├── infer.sh                  # 基准推理脚本
    ├── noisy.sh                  # 噪声测试脚本
    ├── target.sh                 # 目标攻击脚本
    ├── untarget.sh               # 非目标攻击脚本
    │
    ├── model/                    # Uground模型文件
    │   ├── config.json           # 模型配置
    │   ├── model.safetensors     # 模型权重
    │   ├── pytorch_model.bin     # PyTorch模型权重
    │   ├── tokenizer_config.json # 分词器配置
    │   └── ... (其他模型文件)
    │
    ├── result/                   # Uground测试结果
    │
    └── scripts/                  # 测试脚本
        ├── check_attack.py       # 攻击评估脚本
        ├── infer_noisy.py        # 噪声干扰测试脚本
        ├── infer_screenspot.py   # ScreenSpot推理脚本
        ├── target.py             # 目标攻击实现
        ├── untarget.py           # 非目标攻击实现
        │
        └── preprocess/           # 预处理工具
            ├── constants.py      # 常量定义
            ├── data_process.py   # 数据处理工具
            ├── params.py         # 参数设置
            ├── patch_image.py    # 图像补丁处理
            └── point_extract.py  # 点位提取工具
```

## 🌟主要功能模块

### 1. 数据处理与测试  

ScreenSpot推理测试 (`infer_screenspot.py`)

+ 使用原始ScreenSpot数据集对模型进行基准性能测试

+ 评估模型在正常条件下的UI元素定位与交互能力

噪声干扰测试 (`infer_noisy.py`)

+ 向图像添加不同类型的噪声（高斯噪声、椒盐噪声等）
+ 测试模型在图像噪声干扰下的鲁棒性

### 2. 攻击方法实现

目标攻击 (`target.py`)

+ 实现PGD等对抗攻击算法
+ 通过修改图像引导模型点击特定位置
+ 评估模型对有目标性对抗样本的防御能力

非目标攻击 (`untarget.py`)

+ 通过最大化原始与对抗特征差异，扰乱模型的视觉表示
+ 测试模型在无目标攻击下的行为一致性

攻击效果评估 (`check_attack.py)`
+ 计算ACC(准确率)：正常样本下的模型准确率
+ 计算ASR(攻击成功率)：在攻击样本下模型被成功干扰的比率分析不同模型在各类攻击下的表现差异

视觉干扰生成器 (`popup.py`)

我们开发了专用弹窗干扰生成器，可为GUI截图添加各种类型的弹窗干扰：

+ 警告弹窗
+ 广告弹窗
+ 系统通知弹窗
+ 普通确认弹窗
+ 提示弹窗

这些干扰用于测试模型在面对视觉干扰时是否能够保持正确操作，不被误导执行不必要或危险的操作。

## 🚀快速部署
✅🌟✨📑🌐

