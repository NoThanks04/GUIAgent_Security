#对原始数据集进行评测
#windows
$env:CUDA_VISIBLE_DEVICES="0"
python scripts/infer_screenspot.py `
    --model_path "./model" `
    --screenspot_imgs ../datas/Initial/screenspotv2_image `
    --screenspot_test ../datas/Initial `
    --task mobile `
    --max_pixels 256

#Linux
#移动端数据集评测
export CUDA_VISIBLE_DEVICES="0" && \
python scripts/infer_screenspot.py \
    --model_path "./model" \
    --screenspot_imgs ../datas/Initial/screenspotv2_image \
    --screenspot_test ../datas/Initial \
    --task mobile \
    --max_pixels 256

#web端数据集进行评测
export CUDA_VISIBLE_DEVICES="0" && \
python scripts/infer_screenspot.py \
    --model_path "./model" \
    --screenspot_imgs ../datas/Initial/screenspotv2_image \
    --screenspot_test ../datas/Initial \
    --task web \
    --max_pixels 256

#桌面端数据集进行评测
export CUDA_VISIBLE_DEVICES="0" && \
python scripts/infer_screenspot.py \
    --model_path "./model" \
    --screenspot_imgs ../datas/Initial/screenspotv2_image \
    --screenspot_test ../datas/Initial \
    --task desktop \
    --max_pixels 256

#全部数据集进行评测
export CUDA_VISIBLE_DEVICES="0" && \
python scripts/infer_screenspot.py \
    --model_path "./model" \
    --screenspot_imgs ../datas/Initial/screenspotv2_image \
    --screenspot_test ../datas/Initial \
    --task all \
    --max_pixels 256