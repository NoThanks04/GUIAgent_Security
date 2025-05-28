import json
import argparse
import os
import logging
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import time
from datetime import datetime
from transformers.generation import GenerationConfig
from preprocess.point_extract import extract_bbox, pred_2_point

# 配置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
parser = argparse.ArgumentParser()
# 定义命令行参数
parser.add_argument('--model_path', type=str, required=True)  # 模型路径
parser.add_argument('--screenspot_imgs', type=str, required=True)  # 图像目录路径
parser.add_argument('--screenspot_test', type=str, required=True)  # 测试数据集路径
parser.add_argument('--task', type=str, required=True)  # 任务类型（mobile/desktop/web或all）
parser.add_argument('--max_pixels', type=int, required=True)  # 最大像素数
parser.add_argument('--noise_type', type=str, default="all")  # 噪声类型（gaussian_noise/gaussian_blur/color_jitter/contrast_adjusted或all）
args = parser.parse_args()

# 创建结果保存目录
result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    logging.info(f"创建结果目录: {result_dir}")
else:
    logging.info(f"结果将保存到: {result_dir}")

# 定义支持的噪声类型
NOISE_TYPES = ["gaussian_noise", "gaussian_blur", "color_jitter", "contrast_adjusted"]

# 确定要处理的噪声类型
if args.noise_type.lower() == "all":
    noise_types = NOISE_TYPES
else:
    if args.noise_type not in NOISE_TYPES:
        logging.warning(f"不支持的噪声类型: {args.noise_type}，将使用所有噪声类型")
        noise_types = NOISE_TYPES
    else:
        noise_types = [args.noise_type]

# 输出配置信息
logging.info("="*50)
logging.info(f"开始执行噪声数据集推理任务 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logging.info(f"模型路径: {args.model_path}")
logging.info(f"图像目录: {args.screenspot_imgs}")
logging.info(f"测试数据集: {args.screenspot_test}")
logging.info(f"任务类型: {args.task}")
logging.info(f"噪声类型: {', '.join(noise_types)}")
logging.info(f"最大像素数: {args.max_pixels}")
logging.info("="*50)

# 记录开始时间
start_time = time.time()

# 加载模型和预处理器 - 添加local_files_only=True以仅使用本地文件
logging.info("正在加载模型和预处理器...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype="float16", device_map="cuda:0", local_files_only=True
).eval()  # 设置为评估模式
processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
model.generation_config = GenerationConfig.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
logging.info("模型加载完成！")
print("Load Success")
# 设置最小和最大像素
min_pixels=4*28*28
max_pixels=args.max_pixels*28*28
logging.info(f"最小像素: {min_pixels}, 最大像素: {max_pixels}")

# 确定要处理的任务类型
if args.task == "all":
    tasks = ["mobile", "desktop", "web"]
else:
    tasks = [args.task]
logging.info(f"将处理以下任务: {', '.join(tasks)}")

# 为每种噪声类型分别进行评测
for noise_type in noise_types:
    logging.info("="*50)
    logging.info(f"开始处理噪声类型: {noise_type}")
    logging.info("="*50)
    
    # 确认噪声数据集路径
    noise_img_path = os.path.join(args.screenspot_imgs, noise_type)
    if not os.path.exists(noise_img_path):
        logging.error(f"噪声数据集路径不存在: {noise_img_path}")
        continue
        
    logging.info(f"使用噪声数据集: {noise_img_path}")
    
    tasks_result = []  # 存储任务结果
    result = []  # 存储详细结果
    max_num_patchs = 0

    # 各类型样本总数统计
    all_samples_count = {}
    for task in tasks:
        # 构建数据集文件名
        dataset = "screenspot_" + task + "_v2.json"
        dataset_path = os.path.join(args.screenspot_test, dataset)
        
        # 检查文件是否存在
        if not os.path.exists(dataset_path):
            logging.error(f"数据集文件不存在: {dataset_path}")
            continue
            
        # 加载测试数据集
        logging.info(f"正在加载数据集: {dataset}")
        try:
            screenspot_data = json.load(open(dataset_path, 'r'))
            all_samples_count[task] = len(screenspot_data)
            logging.info(f"数据集 {task} 样本数: {len(screenspot_data)}")
        except Exception as e:
            logging.error(f"加载数据集出错: {str(e)}")
            continue

        print("Num of sample: " + str(len(screenspot_data)))

        # 初始化计数器和结果列表
        num_action = 0  # 总样本数
        corr_action = 0  # 正确预测数
        text_correct = []  # 文本元素预测结果
        icon_correct = []  # 图标元素预测结果
        num_wrong_format = 0  # 格式错误数
        task_start_time = time.time()
        
        logging.info(f"开始处理任务: {task} - 噪声类型: {noise_type}")
        for j, item in tqdm(enumerate(screenspot_data), desc=f"处理{task}-{noise_type}数据集", total=len(screenspot_data)):
            sample_start_time = time.time()
            num_action += 1
            filename = item["img_filename"]
            img_path = os.path.join(noise_img_path, filename)
            
            # 检查图片文件是否存在
            if not os.path.exists(img_path):
                logging.warning(f"图片文件不存在: {img_path}")
                num_wrong_format += 1
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                continue
                
            image = Image.open(img_path)
            instruction = item["instruction"]  # 获取指令
            bbox = item["bbox"]  # 获取边界框
            img_size = image.size
            
            if j % 10 == 0:  # 每处理10个样本输出一次详细信息
                logging.info(f"正在处理第 {j+1}/{len(screenspot_data)} 个样本 - 指令: '{instruction}'")
                
            # 转换边界框格式：[x, y, width, height] -> [x1, y1, x2, y2]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            # 归一化边界框坐标
            bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
            
            # 准备输入消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path, "min_pixels": min_pixels, "max_pixels": max_pixels},
                        {"type": "text", "text": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
Description: {instruction}
Answer:"""},
                    ],
                },
            ]
            # 应用聊天模板
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # 处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)
            # 预处理输入
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")  # 移至GPU
            
            # 生成模型输出
            generated_ids = model.generate(**inputs, max_new_tokens=64)  # 减少max_new_tokens

            # 裁剪生成的ID，仅保留新生成的部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            # 解码生成的文本
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]

            try:
                # 从输出文本中提取坐标
                if 'box' in output_text:
                    # 如果输出包含盒子坐标
                    pred_bbox = extract_bbox(output_text)
                    click_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
                    click_point = [item / 1000 for item in click_point]
                else:
                    # 直接提取点击坐标
                    click_point = pred_2_point(output_text)
                    click_point = [item / 1000 for item in click_point]
                
                # 检查预测点是否在目标边界框内
                if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                    corr_action += 1  # 预测正确
                    if item["data_type"] == 'text':
                        text_correct.append(1)
                    else:
                        icon_correct.append(1)
                    if j % 10 == 0:  # 每处理10个样本输出一次详细信息
                        logging.info(f"预测正确 - 点击坐标: ({click_point[0]:.3f}, {click_point[1]:.3f}) - 目标区域: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})")
                        logging.info(f"当前准确率: {corr_action/num_action:.4f}")
                else:
                    # 预测错误
                    if item["data_type"] == 'text':
                        text_correct.append(0)
                    else:
                        icon_correct.append(0)
                    if j % 10 == 0:  # 每处理10个样本输出一次详细信息
                        logging.info(f"预测错误 - 点击坐标: ({click_point[0]:.3f}, {click_point[1]:.3f}) - 目标区域: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})")
                        logging.info(f"当前准确率: {corr_action/num_action:.4f}")
                    with open(os.path.join(result_dir, f"temp_{noise_type}_{task}.txt"), 'a+') as f:
                        f.write(str(filename)+' '+str(click_point)+'\n')
                
                # 保存详细结果
                result.append({
                    "img_path": img_path, 
                    "text": instruction, 
                    "bbox": bbox, 
                    "pred": click_point,
                    "type": item["data_type"], 
                    "source": item["data_source"],
                    "correct": (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]),
                    "task": task,
                    "noise_type": noise_type
                })
                
                # 计算样本处理时间
                sample_process_time = time.time() - sample_start_time
                if j % 10 == 0:  # 每处理10个样本输出一次详细信息
                    logging.info(f"样本处理时间: {sample_process_time:.2f}秒")
                    
            except Exception as e:
                # 处理格式错误
                num_wrong_format += 1
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                logging.error(f"步骤: {j} 格式错误 - {str(e)}")

        # 计算任务处理时间
        task_process_time = time.time() - task_start_time
        
        # 输出整体准确率和统计信息
        logging.info("="*50)
        logging.info(f"任务: {task} - 噪声类型: {noise_type} 完成")
        logging.info("="*50)
        logging.info(f"总样本数: {num_action}")
        logging.info(f"正确预测数: {corr_action}")
        logging.info(f"格式错误数: {num_wrong_format}")
        logging.info(f"总体准确率: {corr_action/num_action:.4f}")
        
        text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
        icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
        
        logging.info(f"文本元素数量: {len(text_correct)}")
        logging.info(f"图标元素数量: {len(icon_correct)}")
        logging.info(f"文本元素准确率: {text_acc:.4f}")
        logging.info(f"图标元素准确率: {icon_acc:.4f}")
        logging.info(f"任务处理时间: {task_process_time:.2f}秒")
        logging.info(f"平均每个样本处理时间: {task_process_time/num_action:.2f}秒")
        logging.info("="*50)

        # 计算文本和图标准确率
        tasks_result.append([text_acc, icon_acc, task])

    # 计算当前噪声类型的处理时间
    noise_process_time = time.time() - start_time

    # 输出当前噪声类型的所有任务结果
    logging.info("="*50)
    logging.info(f"噪声类型 {noise_type} 的所有任务结果:")
    logging.info("="*50)
    for text_acc, icon_acc, task_name in tasks_result:
        samples_count = all_samples_count.get(task_name, 0)
        logging.info(f"任务 {task_name} - 样本数: {samples_count} - 文本准确率: {text_acc:.4f}, 图标准确率: {icon_acc:.4f}")
        
    if len(tasks_result) > 1:
        avg_text_acc = sum([res[0] for res in tasks_result]) / len(tasks_result)
        avg_icon_acc = sum([res[1] for res in tasks_result]) / len(tasks_result)
        logging.info(f"平均文本准确率: {avg_text_acc:.4f}")
        logging.info(f"平均图标准确率: {avg_icon_acc:.4f}")

    logging.info(f"噪声类型 {noise_type} 总处理时间: {noise_process_time:.2f}秒")
    logging.info("="*50)

    # 保存当前噪声类型的结果到文件
    result_file = os.path.join(result_dir, f"Uground_infer_{noise_type}_result.txt")
    logging.info(f"结果已保存到: {result_file}")
    with open(result_file, 'a+') as f:
        f.write(f"\n\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {args.model_path}\n")
        f.write(f"任务: {args.task}\n")
        f.write(f"噪声类型: {noise_type}\n")
        f.write(str(tasks_result)+'\n')
        f.write(f"处理时间: {noise_process_time:.2f}秒\n")
        
    # 保存当前噪声类型的详细结果到JSON文件
    detail_result_file = os.path.join(result_dir, f"Uground_infer_{noise_type}_detail_result.json")
    logging.info(f"详细结果已保存到: {detail_result_file}")
    with open(detail_result_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model_path": args.model_path,
            "tasks": tasks,
            "noise_type": noise_type,
            "results": tasks_result,
            "process_time": noise_process_time,
            "samples": result
        }, f, indent=4)

# 计算总体处理时间
total_process_time = time.time() - start_time
logging.info(f"所有噪声类型总处理时间: {total_process_time:.2f}秒")

# 保存总结果日志
summary_log_file = os.path.join(result_dir, "noise_evaluation_summary.txt")
logging.info(f"总结果日志已保存到: {summary_log_file}")
with open(summary_log_file, 'a+') as f:
    f.write(f"\n\n==== 噪声评估总结 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")
    f.write(f"模型: {args.model_path}\n")
    f.write(f"任务: {args.task}\n")
    f.write(f"噪声类型: {', '.join(noise_types)}\n")
    f.write(f"总处理时间: {total_process_time:.2f}秒\n")
    f.write("="*50 + "\n")
