#四种自然噪声类别
#高斯噪声（gaussian_noise）：在图像上添加随机噪点
#高斯模糊（gaussian_blur）：使图像变得模糊
#颜色抖动（color_jitter）：调整图像的亮度、对比度和饱和度
#对比度调整（contrast_adjusted）：降低图像对比度

# "gaussian_noise", "gaussian_blur", "color_jitter", "contrast_adjusted"
#Linux
#高斯噪声（gaussian_noise）
$env:CUDA_VISIBLE_DEVICES="0"
CUDA_VISIBLE_DEVICES="0" python scripts/infer_noisy.py \
    --model_path "./model" \
    --screenspot_imgs "../datas/Noisy" \
    --screenspot_test "../datas/Noisy" \
    --task all \
    --noise_type gaussian_noise \
    --max_pixels 256

#高斯模糊（gaussian_blur）
$env:CUDA_VISIBLE_DEVICES="0"
CUDA_VISIBLE_DEVICES="0" python scripts/infer_noisy.py \
    --model_path "./model" \
    --screenspot_imgs "../datas/Noisy" \
    --screenspot_test "../datas/Noisy" \
    --task all \
    --noise_type gaussian_blur \
    --max_pixels 256

#颜色抖动（color_jitter）
CUDA_VISIBLE_DEVICES="0" python scripts/infer_noisy.py \
    --model_path "./model" \
    --screenspot_imgs "../datas/Noisy" \
    --screenspot_test "../datas/Noisy" \
    --task all \
    --noise_type color_jitter \
    --max_pixels 256

#对比度调整（contrast_adjusted）
$env:CUDA_VISIBLE_DEVICES="0"
CUDA_VISIBLE_DEVICES="0" python scripts/infer_noisy.py \
    --model_path "./model" \
    --screenspot_imgs "../datas/Noisy" \
    --screenspot_test "../datas/Noisy" \
    --task all \
    --noise_type contrast_adjusted \
    --max_pixels 256