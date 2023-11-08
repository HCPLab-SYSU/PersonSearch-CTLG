import os
import json
from PIL import Image,ImageChops
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch
import csv

def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(w / iw, h / ih)  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    
    new_image = Image.new('RGB', target_size, (0, 0, 0))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式

    return new_image


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "./ldm-super-resolution-4x-openimages"

# load model and scheduler

pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
pipeline = pipeline.to(device)

root='/data2/liuzhibin/STM/dataset/ICFG-PEDES/imgs'
rows=[]
with open("/data2/liuzhibin/STM/dataset/ICFG-PEDES/ICFG-PEDES.json",'r',encoding='utf8')as fp:
    json_data = json.load(fp)
    pre_id = 0
    for i in json_data:
        id = i['id']
        if id > pre_id:
            pre_id = id
            print(id)
            source_path = os.path.join(root,i['file_path'])
            dir = i['file_path'].strip().split('/')[0]
            target_path = os.path.join("./processed_images_icfg_all",i['file_path'])
            caption = i['captions'][0]
            if os.path.exists(os.path.join("./processed_images_icfg_all",dir))==False:
                os.mkdir(os.path.join("./processed_images_icfg_all",dir))
            
            low_res_img = Image.open(source_path).convert("RGB")
            low_res_img = low_res_img.resize((60, 160))
            w,h = low_res_img.size
            tsize = max(low_res_img.size)
            low_res_img = pad_image(low_res_img,(tsize,tsize))

            # run pipeline in inference (sample random noise and denoise)
            upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
            # save image
            upscaled_image=upscaled_image.crop((2*(h-w),0,2*(h+w),4*h))
            upscaled_image.save(target_path)
            
            rows.append((caption,target_path))


with open('processed_images_icfg_all.csv','w',encoding='utf8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['prompt','path'])
    writer.writerows(rows)
