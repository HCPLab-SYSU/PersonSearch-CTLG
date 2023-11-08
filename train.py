import sys
import torch
import torch.backends.cudnn as cudnn
import os
import yaml
import time

from train_config import parse_args
from function import data_config, optimizer_function, load_checkpoint, lr_scheduler, AverageMeter, save_checkpoint, \
    gradual_warmup, fix_seed, test_map, Logger, SAM
from models import *
from loss import InfoNCELoss, CMPMLoss

from torch.cuda.amp import autocast, GradScaler

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


def train(epoch, train_loader, val_loader, network, optimizer, compute_loss, args):
    train_loss = AverageMeter()
    # switch to train mode
    network.train()
    
    # NOTE AMP For fast training
    # GradScaler对象用来自动做梯度缩放
    scaler = GradScaler()

    for step, (images, captions, labels, mask, img_paths) in enumerate(train_loader):
        images = images.cuda()
        captions = captions.cuda()
        labels = labels.cuda()
        mask = mask.cuda()
        optimizer.zero_grad()

        with autocast():
            if args.optimizer == 'SAM':
                # first forward-backward pass
                imgs, txts  = network(images, captions, mask)
                loss = compute_loss(imgs, txts, labels)  # use this loss for any training statistics
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                # second forward-backward pass
                imgs, txts  = network(images, captions, mask)
                loss = compute_loss(imgs, txts, labels)  # use this loss for any training statistics
                loss.backward()
                optimizer.second_step(zero_grad=True)
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
            else:
                # compute loss
                imgs, txts  = network(images, captions, mask)
                loss = compute_loss(imgs, txts, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
               
        # if args.optimizer == 'SAM':
        #     # first forward-backward pass
        #     imgs, txts  = network(images, captions, mask)
        #     loss = compute_loss(imgs, txts, labels)  # use this loss for any training statistics
        #     loss.backward()
        #     optimizer.first_step(zero_grad=True)
            
        #     # second forward-backward pass
        #     imgs, txts  = network(images, captions, mask)
        #     loss = compute_loss(imgs, txts, labels)  # use this loss for any training statistics
        #     loss.backward()
        #     optimizer.second_step(zero_grad=True)
            
        # else:
        #     # compute loss
        #     imgs, txts  = network(images, captions, mask)
        #     loss = compute_loss(imgs, txts, labels)
        #     # graduate
        #     loss.backward()
        #     optimizer.step()

        train_loss.update(loss, images.shape[0])
        if step % 100 == 0:
            print(
                "Train Epoch:[{}/{}] iteration:[{}/{}] loss:{:.4f}".format(epoch, args.num_epoches, step,
                                                                                len(train_loader), train_loss.avg,))

    state = {"epoch": epoch,
             "state_dict": network.state_dict(),
            #  "W": compute_loss.W
             }

    print("Evaluating in validation set")
    ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test(val_loader, network, args)
    result = [ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP]

    return state, epoch, result

def main(network, dataloader, compute_loss, optimizer, scheduler, start_epoch, args, checkpoint_dir):
    start = time.time()
    best_result = 0 
    for epoch in range(start_epoch, args.num_epoches):
        print("-"*70)
        if epoch < args.warm_epoch:
            print('learning rate warm_up')
            if args.optimizer == 'sgd':
                optimizer = gradual_warmup(epoch, args.sgd_lr, optimizer, epochs=args.warm_epoch)
            else:
                optimizer = gradual_warmup(epoch, args.adam_lr, optimizer, epochs=args.warm_epoch)

        # train and update
        model_state, epoch, result  = train(epoch, dataloader['train'], dataloader['val'], network, optimizer, compute_loss, args)
        sum_eval_result = sum(result)
        scheduler.step()

        Epoch_time = time.time() - start
        start = time.time()
        print('Epoch_training complete in {:.0f}m {:.0f}s'.format(
            Epoch_time // 60, Epoch_time % 60))

        if best_result < sum_eval_result and result[0] > 0.6:
            best_result = sum_eval_result
            print("*"*15, f"Best Eval model in {epoch}", "*"*15)
            save_checkpoint(model_state, 'best', checkpoint_dir)   # save best

    save_checkpoint(model_state, 'last', checkpoint_dir)   # save last

    ckpt_names = os.listdir(checkpoint_dir)
    ckpt_names = sorted([x for x in ckpt_names if x.endswith('.pth.tar')])
    test_loader = dataloader['test']
    for ckpt in ckpt_names:
        model_file = os.path.join(checkpoint_dir, ckpt)
        print(model_file)
        if os.path.isdir(model_file):
            continue
        start, network = load_checkpoint(model, model_file)
        test(test_loader, network, args)


if __name__=='__main__':
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
        cudnn.benchmark = True  # make the training speed faster
    fix_seed(args.seed)

    name = args.name
    # set some paths
    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, name)
    print('>>>>>>>>>>>>>>>>> checkpoint dir:', checkpoint_dir)

    sys.stdout = Logger(os.path.join(checkpoint_dir, "train_log.txt"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open('%s/opts_train.yaml' % checkpoint_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    # Network
    network = eval(args.model)  # 从参数传入模型的名字
    model = network(args).cuda()   
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        # compute the model size:
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # load checkpoint
    if args.resume:
        start_epoch, model = load_checkpoint(model, args.resume)
    else:
        print("Do not load checkpoint,Epoch start from 0")
        start_epoch = 0

    dataloaders = {x: data_config(args, x)
                   for x in ['train', 'val', 'test']}


    # loss function
    # compute_loss = CMPMLoss(args).cuda()
    compute_loss = InfoNCELoss(args.temperature).cuda()

    # compute_loss = ModifiedInfoNCELoss(args).cuda()


    # optimizer
    optimizer = optimizer_function(args, model)
    exp_lr_scheduler = lr_scheduler(optimizer, args)
    main(model, dataloaders, compute_loss, optimizer, exp_lr_scheduler, start_epoch, args, checkpoint_dir)

