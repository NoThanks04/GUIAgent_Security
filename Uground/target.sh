#进行目标攻击
#Linux
#对web端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/target.py \
    --model_path "./model" \
    --screenspot_imgs ../datas/Initial/screenspotv2_image \
    --screenspot_test ../datas/Initial/screenspotv2_web_ug_target.json \
    --output_path ../outputs/ug/target \
    --max_pixels 256

#对移动端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/target.py \
    --model_path "./model" \
    --screenspot_imgs ../datas/Initial/screenspotv2_image \
    --screenspot_test ../datas/Initial/screenspotv2_mobile_ug_target.json \
    --output_path ../outputs/ug/target \
    --max_pixels 256

#对桌面端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/target.py \
    --model_path "./model" \
    --screenspot_imgs ../datas/Initial/screenspotv2_image \
    --screenspot_test ../datas/Initial/screenspotv2_desktop_ug_target.json \
    --output_path ../outputs/ug/target \
    --max_pixels 256 

#进行非目标攻击

#对web端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/untarget.py \
    --model_path "./model" \
    --screenspot_imgs ../datas/Initial/screenspotv2_image \
    --screenspot_test ../datas/Initial/screenspotv2_web_ug_target.json \
    --output_path ../outputs/ug/untarget \
    --max_pixels 256 

#对移动端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/untarget.py \
    --model_path "./model" \
    --screenspot_imgs ../datas/Initial/screenspotv2_image \
    --screenspot_test ../datas/Initial/screenspotv2_mobile_ug_target.json \
    --output_path ../outputs/ug/target \
    --max_pixels 256

#对桌面端数据集进行攻击
CUDA_VISIBLE_DEVICES="0" \
python scripts/untarget.py \
    --model_path "./model" \
    --screenspot_imgs ../datas/Initial/screenspotv2_image \
    --screenspot_test ../datas/Initial/screenspotv2_desktop_ug_target.json \
    --output_path ../outputs/ug/untarget \
    --max_pixels 256 