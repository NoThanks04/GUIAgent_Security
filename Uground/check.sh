# python Uground/check_attack.py \
#     --model_path osunlp/UGround-V1-7B \
#     --screenspot_imgs output \
#     --screenspot_test datas \
#     --task all \
#     --max_pixels 1024

# CUDA_VISIBLE_DEVICES=0 python scripts/check_attack.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../outputs/ug/untarget \
#     --screenspot_test ../datas \
#     --task web \
#     --max_pixels 1024
# [[0.6495726495726496, 0.46798029556650245]] web untarget
# [[0.0, 0.0]]

# CUDA_VISIBLE_DEVICES=0 python scripts/check_attack.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../outputs/ug/untarget \
#     --screenspot_test ../datas \
#     --task desktop \
#     --max_pixels 1024
# [[0.5154639175257731, 0.2571428571428571]]
# [[0.0, 0.0]]

# CUDA_VISIBLE_DEVICES=3 python scripts/check_attack.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../outputs/ug/untarget \
#     --screenspot_test ../datas \
#     --task mobile \
#     --max_pixels 1024
# [[0.7448275862068966, 0.46919431279620855]]
# [[0.0, 0.0]]

# CUDA_VISIBLE_DEVICES=0 python scripts/check_attack.py \
#     --model_path /data/home/zhr/models/UGround-V1-7B \
#     --screenspot_imgs ../outputs/ug/target/256 \
#     --screenspot_test ../datas \
#     --task web \
#     --max_pixels 256
# [[0.2264957264957265, 0.1477832512315271]]
# [[0.20085470085470086, 0.30049261083743845]]


#对攻击图片进行评估
#对目标攻击进行评估
#对web端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/check_attack.py \
    --model_path "./model" \
    --screenspot_imgs ../outputs/ug/target \
    --screenspot_test ../datas/Initial \
    --task web \
    --max_pixels 256

#对移动端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/check_attack.py \
    --model_path "./model" \
    --screenspot_imgs ../outputs/ug/target \
    --screenspot_test ../datas/Initial \
    --task mobile \
    --max_pixels 256

#对桌面端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/check_attack.py \
    --model_path "./model" \
    --screenspot_imgs ../outputs/ug/target \
    --screenspot_test ../datas/Initial \
    --task desktop \
    --max_pixels 256

#对非目标攻击进行评估
#对web端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/check_attack.py \
    --model_path "./model" \
    --screenspot_imgs ../outputs/ug/untarget \
    --screenspot_test ../datas/Initial \
    --task web \
    --max_pixels 256

#对移动端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/check_attack.py \
    --model_path "./model" \
    --screenspot_imgs ../outputs/ug/untarget \
    --screenspot_test ../datas/Initial \
    --task mobile \
    --max_pixels 256

#对桌面端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/check_attack.py \
    --model_path "./model" \
    --screenspot_imgs ../outputs/ug/untarget \
    --screenspot_test ../datas/Initial \
    --task desktop \
    --max_pixels 256