from email.policy import strict
import errno
import sys
import os.path as osp
import torch.utils.data as data
import os
import torch
import numpy as np
import random
import torchvision.transforms as transforms
from text_reid_dataset import CUHKPEDES_BERT_Token, PETA_BERT_Token, RSTPReid_BERT_Token, ICFGPEDES_BERT_Token
import math

   
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class UniqueBatchSampler(data.BatchSampler):
    # NOTE 对原来的 BatchSampler的一点小修正，确保batch中的id不会重复。
    # idx_interval = 10

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            flag = 0
            for i in batch:
                if abs(i-idx) < 10:
                    flag = 1
            if flag:
                continue

            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

# def unique_batch_sampler():
#     print(list(UniqueBatchSampler(data.RandomSampler(range(1000)), batch_size=10, drop_last=False)))

# unique_batch_sampler()


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


def data_config(args, split):
    transform_train_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((args.height, args.width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.RandomErasing(scale=(0.02, 0.4))
    ]

    transform_val_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # define dictionary: data_transforms
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
        'test': transforms.Compose(transform_val_list),
    }

    if split == 'train':
        if args.dataset == 'CUHKPEDES':
            data_split = CUHKPEDES_BERT_Token(args, split, annotation_path=args.anno_path,  transform=data_transforms[split])
        elif args.dataset == 'RSTPReid':
            data_split = RSTPReid_BERT_Token(args, split, annotation_path='RSTPReid/data_captions.json', transform=data_transforms[split])
        elif args.dataset == 'ICFGPEDES':
            # data_split = ICFGPEDES_BERT_Token(args, split, annotation_path='ICFG-PEDES/ICFG-PEDES.json', transform=data_transforms[split])
            data_split = ICFGPEDES_BERT_Token(args, split, annotation_path=args.anno_path, transform=data_transforms[split])

        elif args.dataset == 'concat':
            data_split1 = CUHKPEDES_BERT_Token(args, split, annotation_path='CUHK-PEDES/reid_raw.json',  transform=data_transforms[split])
            # data_split2 = RSTPReid_BERT_Token(args, split, annotation_path='RSTPReid/data_captions.json', transform=data_transforms[split])
            # data_split3 = ICFGPEDES_BERT_Token(args, split, annotation_path='ICFG-PEDES/ICFG-PEDES.json', transform=data_transforms[split])
            # data_split = data.ConcatDataset([data_split1, data_split2, data_split3])

            # NOTE add PETA additional classification dataset
            data_split2 = PETA_BERT_Token(args, split,  transform=data_transforms[split])
            data_split = data.ConcatDataset([data_split1, data_split2])
        shuffle = True

        # NOTE origin sampler setting
        loader = data.DataLoader(data_split, args.batch_size, shuffle=shuffle, num_workers=4)

        ## NOTE weighted sampling
        # shuffle = False
        # data_split = CUHKPEDES_BERT_Token(dir, split, max_length, annotation_path='CUHK-PEDES/reid_raw_freq.json',  transform=transform, cap_augment=True, embedding_type = embedding_type)
        # sampler = data.WeightedRandomSampler(data_split.sample_weight(), len(data_split))
        # loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, sampler=sampler, num_workers=8)

        # # # NOTE unique batch sampler
        # ubs = UniqueBatchSampler(data.RandomSampler(range(len(data_split))), batch_size=args.batch_size, drop_last=False)
        # loader = data.DataLoader(data_split, batch_sampler=ubs, num_workers=8)

    else:
        if args.dataset == 'CUHKPEDES':
            data_split = CUHKPEDES_BERT_Token(args, split, annotation_path='CUHK-PEDES/reid_raw.json',  transform=data_transforms[split])
        elif args.dataset == 'RSTPReid':
            data_split = RSTPReid_BERT_Token(args, split, annotation_path='RSTPReid/data_captions.json', transform=data_transforms[split])
        elif args.dataset == 'ICFGPEDES':
            data_split = ICFGPEDES_BERT_Token(args, split, annotation_path='ICFG-PEDES/ICFG-PEDES.json', transform=data_transforms[split])
        elif args.dataset == 'concat':
            # concat的数据集暂时用这个作为test吧...
            data_split = CUHKPEDES_BERT_Token(args, split, annotation_path='CUHK-PEDES/reid_raw.json',  transform=data_transforms[split])
            
        shuffle = False
        loader = data.DataLoader(data_split, args.batch_size, shuffle=shuffle, num_workers=4)

    return loader


def optimizer_function(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.adam_lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)
        print("optimizer is：Adam")

    if args.optimizer == 'adamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.adam_lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)
        print("optimizer is：AdamW")

    if args.optimizer == 'SAM':
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model.parameters(), base_optimizer, lr=args.adam_lr)
    return optimizer


def lr_scheduler(optimizer, args):

    if args.lr_decay_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min', factor=args.lr_decay_ratio,
                                                           patience=5, min_lr=args.end_lr)
        print("lr_scheduler is ReduceLROnPlateau")
    elif args.lr_decay_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.num_epoches, eta_min=args.adam_lr/10)
        print("lr_scheduler is CosineAnnealingLR")
    else:
        if '_' in args.epoches_decay:
            epoches_list = args.epoches_decay.split('_')
            epoches_list = [int(e) for e in epoches_list]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list, gamma=args.lr_decay_ratio)
            print("lr_scheduler is MultiStepLR")
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay), gamma=args.lr_decay_ratio)
            print("lr_scheduler is StepLR")
    return scheduler


def load_checkpoint(model,resume):
    start_epoch=0
    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        # checkpoint= torch.load(resume, map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('Load checkpoint at epoch %d.' % (start_epoch))
    return start_epoch,model


class AverageMeter(object):
    """
    Computes and stores the averate and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, epoch, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    filename = os.path.join(dst, str(epoch)) + '.pth.tar'
    torch.save(state, filename)


def gradual_warmup(epoch,init_lr,optimizer,epochs):
    lr = init_lr
    if epoch < epochs:
        warmup_percent_done = (epoch+1) / epochs
        warmup_learning_rate = init_lr * warmup_percent_done
        lr = warmup_learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def compute_topk(query, gallery, target_query, target_gallery, k=[1,10], reverse=False):
    result = []
    query = query / (query.norm(dim=1,keepdim=True)+1e-12)
    gallery = gallery / (gallery.norm(dim=1,keepdim=True)+1e-12)
    sim_cosine = torch.matmul(query, gallery.t())
    result.extend(topk(sim_cosine, target_gallery, target_query, k))
    if reverse:
        result.extend(topk(sim_cosine, target_query, target_gallery, k, dim=0))
    return result


def topk(sim, target_gallery, target_query, k=[1,10], dim=1):
    result = []
    maxk = max(k)
    size_total = len(target_gallery)
    _, pred_index = sim.topk(maxk, dim, True, True)
    pred_labels = target_gallery[pred_index]
    if dim == 1:
        pred_labels = pred_labels.t()
    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

    for topk in k:
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
    return result


def check_exists(root):
    if os.path.exists(root):
        return True
    return False


def load_embedding(path):
    word_embedding=torch.from_numpy(np.load(path))
    (vocab_size,embedding_size)=word_embedding.shape
    print('Load word embedding,the shape of word embedding is [{},{}]'.format(vocab_size,embedding_size))
    return word_embedding


def load_part_model(model,path):
    model_dict = model.state_dict()
    checkpoint = torch.load(path)
    pretrained_dict = checkpoint["state_dict"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def visulize(i, qf, ql, gf, gl):
    query = qf.view(-1, 1)  # 2048*1
    score = torch.mm(gf, query) # 3078*2048 x 2048*1 -->  3078*1 (3078 --> num of gallery images)
    score = score.squeeze(1).cpu()  # 3074
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1] # order of the scores. maybe visualize the result here???
    print("label index: ",i,"top 10 retrieval images index: ",index[:10])


def test_map(query_feature,query_label,gallery_feature, gallery_label):
    # all_text * dim, all_images* dim(half of all),  
    # print(query_feature.shape, query_label.shape, gallery_feature.shape, gallery_label.shape)
    query_feature = query_feature / (query_feature.norm(dim=1, keepdim=True) + 1e-12)
    gallery_feature = gallery_feature / (gallery_feature.norm(dim=1, keepdim=True) + 1e-12)

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i],  gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp # rank1就是CMC[0]出现的次数。后面以此类推。
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_label)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    return CMC[0], CMC[4], CMC[9], ap / len(query_label)

def test_map_pooling(query_feature,query_label,gallery_feature, gallery_label):
    # all_text * dim, all_images* dim(half of all),  
    # print(query_feature.shape, query_label.shape, gallery_feature.shape, gallery_label.shape)
    
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i],  gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp # rank1就是CMC[0]出现的次数。后面以此类推。
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_label)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    return CMC[0], CMC[4], CMC[9], ap / len(query_label)


def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)  # 2048*1
    score = torch.mm(gf, query) # 3078*2048 x 2048*1 -->  3078*1 (3078 --> num of gallery images)
    score = score.squeeze(1).cpu()  # 3074
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1] # order of the scores. maybe visualize the result here???
    gl=gl.cuda().data.cpu().numpy() # 3074
    ql=ql.cuda().data.cpu().numpy() # 1
    query_index = np.argwhere(gl == ql) # 一个描述对应了多图？都是同一个id的人，匹配到其中1个就算作是OK的？
    CMC_tmp = compute_mAP(index, query_index)
    return CMC_tmp


def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)   # find the match index and set it True
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten() # 得到正确id的排序位置。0是最佳的匹配-。-，往后算就是topk了？

    cmc[rows_good[0]:] = 1  # rows_good[0]以后的全部+1？ -- find the image in topk.
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
