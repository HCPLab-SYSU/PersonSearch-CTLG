import torchvision.transforms as transforms
import torch
import yaml
from function import *
from test_config import parse_args
import time
# from models.model import Network
from models import *
import os
import shutil
import torch.backends.cudnn as cudnn
# from tensorboard_logger import configure, log_value

def test(data_loader, network, args):
    # val_loss = AverageMeter()
    # switch to evaluate mode
    network.eval()
    max_size = args.batch_size * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size, args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    index = 0
    with torch.no_grad():
        for images, captions, labels, mask, img_paths in data_loader:
            images = images.cuda()
            captions = captions.cuda()
            mask = mask.cuda()

            interval = images.shape[0]
            image_embeddings, text_embeddings = network(images, captions, mask)
            # img_f1, txts_f1  = network(images, captions, mask, forward_flag=1)
            # image_embeddings, text_embeddings = network(images, captions, mask, img_f1, txts_f1, forward_flag=2)

            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            labels_bank[index:index + interval] = labels

            index = index + interval

        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]
        # we input the two times of images, so we need to select half of them
        ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test_map(text_bank, labels_bank, images_bank[::2], labels_bank[::2])
        torch.cuda.empty_cache()
        return ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP


def main(model, args):
    
    test_loader = data_config(args, 'test')
                   
    ac_t2i_top1_best = 0.0
    ac_t2i_top5_best = 0.0
    ac_t2i_top10_best = 0.0
    mAP_best = 0.0
    best = 0
    dst_best = os.path.join(args.checkpoint_dir , "model_best.pth.tar")
    if os.path.exists(dst_best):
        model_file = dst_best
        print(model_file)
        start, network = load_checkpoint(model, model_file)
        ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test(test_loader, network, args)
        if ac_top1_t2i > ac_t2i_top1_best:
            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top5_best = ac_top5_t2i
            ac_t2i_top10_best = ac_top10_t2i
            mAP_best = mAP

    else:
        ckpt_names = os.listdir(args.checkpoint_dir)
        ckpt_names = sorted([x for x in ckpt_names if x.endswith('.pth.tar')])

        for ckpt in ckpt_names:
            model_file = os.path.join(args.checkpoint_dir, ckpt)
            print(model_file)
            # model_file = os.path.join(args.model_path, 'model_best.pth.tar')
            if os.path.isdir(model_file):
                continue
            start, network = load_checkpoint(model, model_file)
            ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test(test_loader, network, args)
            if ac_top1_t2i > ac_t2i_top1_best:
                ac_t2i_top1_best = ac_top1_t2i
                ac_t2i_top5_best = ac_top5_t2i
                ac_t2i_top10_best = ac_top10_t2i
                mAP_best = mAP
                # shutil.copyfile(model_file, dst_best)

    print('Epoch:{}:t2i_top1_best: {:.5f}, t2i_top5_best: {:.5f},t2i_top10_best: {:.5f},'
          'mAP_best: {:.5f}'.format(
            best, ac_t2i_top1_best, ac_t2i_top5_best, ac_t2i_top10_best, mAP_best))

if __name__ == '__main__':
    args = parse_args()
    sys.stdout = Logger(os.path.join(args.checkpoint_dir, "test_log.txt"))
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
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    main(model, args)