import torchvision.transforms as transforms
import torch
import yaml
from function import *
from test_config import parse_args
import time
from models import *
import os
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from text_reid_dataset import CUHKPEDES_BERT_Token, RSTPReid_BERT_Token, ICFGPEDES_BERT_Token

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        print(grads.shape)
        return np.mean(grads, axis=(2))
        
def reshape_transform(tensor, height=24, width=8):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

if __name__ == '__main__':
    args = parse_args()
    # load GPU
    str_ids = args.gpus.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)
    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    with open('%s/opts_test.yaml' % args.checkpoint_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)
    network = eval(args.model) # 从参数传入模型的名字
    model = network(args).cuda()
    model = torch.nn.DataParallel(model, device_ids=gpu_ids) 
    model.eval()

    dst_best = os.path.join(args.checkpoint_dir , "best.pth.tar")
    model_file = dst_best
    print(model_file)
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print('Load checkpoint at epoch %d.' % (start_epoch))


    test_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if args.dataset == 'CUHKPEDES':
        data_split = CUHKPEDES_BERT_Token(args, 'train', annotation_path='CUHK-PEDES/reid_raw.json',  transform=test_transform)
    loader = data.DataLoader(data_split, 4,  shuffle=False, num_workers=1)

    # inference(loader, network, args)
    class SimilarityToConceptTarget:
        def __init__(self, features):
            self.features = features
        
        def __call__(self, model_output):
            cos = torch.nn.CosineSimilarity(dim=0)
            return cos(model_output, self.features)
    
    
    with torch.no_grad():
        for images, captions, labels, mask, img_paths in tqdm(loader):
            print(images.shape, img_paths)
            images = images.cuda()
            captions = captions.cuda()
            mask = mask.cuda()
            image_embeddings, text_embeddings = model(images, captions, mask)
            image_embeddings /=  (image_embeddings.norm(dim=1, keepdim=True) + 1e-12)
            text_embeddings /= (text_embeddings.norm(dim=1, keepdim=True) + 1e-12)
            score = torch.mm(text_embeddings, image_embeddings.T)
            print("====>score: ",score)

            image_concept_features = image_embeddings[0, :]
            text_concept_features = text_embeddings[0, :] 

            
            print(text_concept_features.shape, text_concept_features.shape)
            break
        
    target_layers = [model.module.model_img.vit.blocks[-1].norm1]
    img_targets = [SimilarityToConceptTarget(image_concept_features)]
    txt_targets = [SimilarityToConceptTarget(text_concept_features)]
    

    with GradCAM(model=model.module.model_img,
                target_layers=target_layers,
                # reshape_transform=reshape_transform,
                use_cuda=False) as cam:

        grayscale_cam = cam(input_tensor=images,
                            targets=txt_targets,
                            )[0, :]

    print(grayscale_cam.shape)
    rgb_img = cv2.imread(img_paths[0], 1)[:, :, ::-1]
    # print(rgb_img.shape)
    rgb_img = cv2.resize( rgb_img, (128,384))
    rgb_img = np.float32(rgb_img) / 255
    print(rgb_img.shape)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite('cam_img.png', cam_image)