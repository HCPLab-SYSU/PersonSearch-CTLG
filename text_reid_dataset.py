import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from imageio import imread
import torch
import json
import transformers
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import random



#判断路径是否存在
def check_exists(root):
    if os.path.exists(root):
        return True
    return False

class CUHKPEDES_BERT_Token(data.Dataset):
    '''
    Args:
        root (string): Base root directory of dataset where [split].pkl and [split].h5 exists
        split (string): 'train', 'val' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed vector. E.g, ''transform.RandomCrop'
        target_transform (callable, optional): A funciton/transform that tkes in the
            targt and transfomrs it.
    '''

    def __init__(self, args, split,  annotation_path='CUHK-PEDES/reid_raw_freq.json', transform=None):
        self.root = args.dir
        self.max_length = args.max_length
        self.transform = transform
        self.cap_augment = args.cap_aug
        self.split = split.lower() 
        self.reid_raw_path = os.path.join(self.root, annotation_path)
        self.clip = True if args.model=='CLIP' else False
        if self.clip:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32")
            self.max_length = 77

        print("load annotations from", self.reid_raw_path)
        with open(self.reid_raw_path, 'r') as f:
            data = json.load(f)
            self.txt_data = self.data_parse(data)

        if args.embedding_type == 'BERT':
            tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')
        elif args.embedding_type == 'Roberta':
            tokenizer_class, pretrained_weights = (transformers.RobertaTokenizer, 'roberta-base')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)


        if not check_exists(self.root):
            print(self.root)
            raise RuntimeError('Dataset not found or corrupted.' +
                               'Please follow the directions to generate datasets')

        if self.split == 'train':
            data = self.txt_data["train"]
            self.train_labels = [int(i)-1 for i in data['labels']]
            self.train_captions = data['captions']
            self.train_images = data['image_paths']
            # self.train_attention_mask = data['attention_mask']
        elif self.split == 'val':
            data = self.txt_data["val"]
            self.val_labels = [int(i) - 11004 for i in data['labels']]
            self.val_captions = data['captions']
            self.val_images = data['image_paths']
            # self.val_attention_mask = data['attention_mask']

        elif self.split == 'test':
            data = self.txt_data["test"]
            self.test_labels = [int(i) -12004 for i in data['labels']]
            self.test_captions = data['captions']
            self.test_images = data['image_paths']
            # self.test_attention_mask = data['attention_mask']

        else:
            raise RuntimeError('Wrong split which should be one of "train","val" or "test"')


        if self.split == 'train':
            # NOTE 改为随机强度的增强
            aug_p_rand = random.uniform(0.2, 0.3)
            self.cap_aug_act = []
            if args.cap_wordnet > 0:
                self.cap_aug_act.append(naw.SynonymAug(aug_src='wordnet', aug_p=args.cap_wordnet))
                # self.cap_aug_act.append(naw.SynonymAug(aug_src='wordnet', aug_p=aug_p_rand))
            if args.cap_crop > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='crop', aug_p= args.cap_crop))
                # self.cap_aug_act.append(naw.RandomWordAug(action='crop', aug_p= aug_p_rand))
            if args.cap_delete > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='delete', aug_p= args.cap_delete))
                # self.cap_aug_act.append(naw.RandomWordAug(action='delete', aug_p= aug_p_rand))
            if args.cap_swap > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='swap', aug_p= args.cap_swap))
                # self.cap_aug_act.append(naw.RandomWordAug(action='swap', aug_p= aug_p_rand))
            # mixup 
            self.mixup = args.mixup

    def cap_transform(self, caption):
        if random.random() < 0.5:
            # 随机选一种来增强
            aug = random.choice(self.cap_aug_act)
            caption = aug.augment(caption)
        return caption

    def __len__(self):
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.val_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """

        if self.split == 'train':
            img_path, caption, label = 'CUHK-PEDES/imgs/'+self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, label = 'CUHK-PEDES/imgs/'+self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            img_path, caption, label = 'CUHK-PEDES/imgs/'+self.test_images[index], self.test_captions[index], self.test_labels[index]
        img_path = os.path.join(self.root, img_path)
        img = imread(img_path)

        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        img = Image.fromarray(img)
        label=torch.tensor(label)


        if self.clip:
            img = self.preprocess(img)
        else:
            if self.transform is not None:
                img = self.transform(img)

        if self.split == 'train' and self.mixup:
            if random.random() < 0.5:
                rand_index = random.randint(0, len(self.train_captions)-1)
                rand_img_path, rand_caption = 'CUHK-PEDES/imgs/'+self.train_images[rand_index], self.train_captions[rand_index]
                rand_img = imread(os.path.join(self.root, rand_img_path))
                if len(rand_img.shape) == 2:
                    img = np.dstack((rand_img, rand_img, rand_img))
                rand_img = Image.fromarray(rand_img)
                rand_img = self.transform(rand_img)
                
                img = 0.5*(img+rand_img)
                caption += rand_caption
        
        # print(caption, img.shape)

        if self.split =='train' and self.cap_augment:
            caption = self.cap_transform(caption)

        # caption = caption[1:-1]
        # print(caption, img.shape)

        caption_id, attention_mask = self.cap_tokenize(caption, self.tokenizer)
        caption_id = torch.tensor(caption_id).long()
        attention_mask = torch.tensor(attention_mask).long()
        # print(img, caption, label, attention_mask, img_path)
        # if args.clip:
        # return img, caption, label, attention_mask, img_path
        # else:
        return img, caption_id, label, attention_mask, img_path

    def sample_weight(self):
        # print(self.txt_data.keys())

        sample_weight = 1/ (self.txt_data['train']['word_freq'] + 0.2)
        return sample_weight

    def data_parse(self, data):
        phase = self.split
        # phases = ["train","val","test"]
        ret_dict = {}

        print('phase:', phase)
        processing_data = [x for x in data if x['split']==phase]
        captions = []
        images_path = []
        labels = []
        word_freqs = []
        for i in tqdm(processing_data):
            for j, cap in enumerate(i['captions']):
                captions.append(cap)
                images_path.append(i['file_path'])
                labels.append(i['id'])


        images_path = np.array(images_path)
        labels = np.array(labels)
        word_freqs = np.array(word_freqs)

        dict={'captions': captions, 'image_paths': images_path, 'labels': labels, 'word_freq': word_freqs}
        ret_dict[phase] = dict
        return ret_dict

    def cap_tokenize(self, cap, tokenizer):
        tokenized = tokenizer.encode(cap, add_special_tokens=True)
        max_len = self.max_length
        
        if len(tokenized) < max_len:
            tokenized += [0] * (max_len-len(tokenized))
        else:
            tokenized = tokenized[:max_len]

        tokenized = np.array(tokenized)   
        attention_mask = np.where(tokenized != 0, 1, 0)

        return tokenized, attention_mask


class RSTPReid_BERT_Token(data.Dataset):
    '''
    Args:
        root (string): Base root directory of dataset where [split].pkl and [split].h5 exists
        split (string): 'train', 'val' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed vector. E.g, ''transform.RandomCrop'
        target_transform (callable, optional): A funciton/transform that tkes in the
            targt and transfomrs it.
    '''
    def __init__(self, args, split,  annotation_path='RSTPReid/data_captions.json', transform=None):
        self.root = args.dir
        self.max_length = args.max_length
        self.transform = transform
        self.cap_augment = args.cap_aug
        self.split = split.lower() 
        self.reid_raw_path = os.path.join(self.root, annotation_path)
        print("load annotations from", self.reid_raw_path)
        with open(self.reid_raw_path, 'r') as f:
            data = json.load(f)
            self.txt_data = self.data_parse(data)

        if args.embedding_type == 'BERT':
            tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')
        elif args.embedding_type == 'Roberta':
            tokenizer_class, pretrained_weights = (transformers.RobertaTokenizer, 'roberta-base')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        if not check_exists(self.root):
            print(self.root)
            raise RuntimeError('Dataset not found or corrupted.' +
                               'Please follow the directions to generate datasets')
        if self.split == 'train':
            data = self.txt_data["train"]
            self.train_labels = [int(i) for i in data['labels']]
            self.train_captions = data['captions']
            self.train_images = data['image_paths']
            # self.train_attention_mask = data['attention_mask']
        elif self.split == 'val':
            data = self.txt_data["val"]
            self.val_labels = [int(i) - 3701 for i in data['labels']]
            self.val_captions = data['captions']
            self.val_images = data['image_paths']
            # self.val_attention_mask = data['attention_mask']

        elif self.split == 'test':
            data = self.txt_data["test"]
            self.test_labels = [int(i) -3901 for i in data['labels']]
            self.test_captions = data['captions']
            self.test_images = data['image_paths']
            # self.test_attention_mask = data['attention_mask']

        else:
            raise RuntimeError('Wrong split which should be one of "train","val" or "test"')

        if self.split == 'train':

            self.cap_aug_act = []
            if args.cap_wordnet > 0:
                self.cap_aug_act.append(naw.SynonymAug(aug_src='wordnet', aug_p=args.cap_wordnet))
            if args.cap_crop > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='crop', aug_p= args.cap_crop))
            if args.cap_delete > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='delete', aug_p= args.cap_delete))
            if args.cap_swap > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='swap', aug_p= args.cap_swap))

            # mixup 
            self.mixup = args.mixup

    def cap_transform(self, caption):
        if random.random() < 0.5:
            # 随机选一种来增强
            aug = random.choice(self.cap_aug_act)
            caption = aug.augment(caption)
        return caption

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """

        if self.split == 'train':
            img_path, caption, label = 'RSTPReid/imgs/'+self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, label = 'RSTPReid/imgs/'+self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            img_path, caption, label = 'RSTPReid/imgs/'+self.test_images[index], self.test_captions[index], self.test_labels[index]
        img_path = os.path.join(self.root, img_path)
        img = imread(img_path)

        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label=torch.tensor(label)
        if self.split == 'train' and self.cap_augment:
            caption = self.cap_transform(caption)

        caption_id, attention_mask = self.cap_tokenize(caption, self.tokenizer)
        caption_id = torch.tensor(caption_id).long()
        attention_mask = torch.tensor(attention_mask).long()
        return img, caption_id, label, attention_mask, img_path


    def __len__(self):
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.val_labels)
        else:
            return len(self.test_labels)
    

    def data_parse(self, data):
        phase = self.split
        ret_dict = {}

        print('phase:', phase)
        
        if phase == 'train':
            processing_data = [x for x in data.values() if x['id']<3701]
        elif phase == 'val':
            processing_data = [x for x in data.values() if x['id']<3901 and x['id']>=3701]
        elif phase == 'test':
            processing_data = [x for x in data.values() if  x['id']>=3901]

        captions = []
        images_path = []
        labels = []
        word_freqs = []
        for i in tqdm(processing_data):
            for j, cap in enumerate(i['captions']):
                captions.append(cap)
                images_path.append(i['img_path'])
                labels.append(i['id'])
                # word_freqs.append( i['word_freq'][j] )   # 大致的反比抽样， 加个0.2避免太过极端...（也即极限也是1：6抽样）；（+0.3 --> 1:

        images_path = np.array(images_path)
        labels = np.array(labels)
        word_freqs = np.array(word_freqs)

        dict={'captions': captions, 'image_paths': images_path, 'labels': labels, 'word_freq': word_freqs}
        ret_dict[phase] = dict
        return ret_dict


    def cap_tokenize(self, cap, tokenizer):
        tokenized = tokenizer.encode(cap, add_special_tokens=True, truncation=True)     # some sentence excess 512...
        max_len = self.max_length
        
        if len(tokenized) < max_len:
            tokenized += [0] * (max_len-len(tokenized))
        else:
            tokenized = tokenized[:max_len]

        tokenized = np.array(tokenized)   
        attention_mask = np.where(tokenized != 0, 1, 0)

        return tokenized, attention_mask



class ICFGPEDES_BERT_Token(data.Dataset):
    '''
    Args:
        root (string): Base root directory of dataset where [split].pkl and [split].h5 exists
        split (string): 'train', 'val' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed vector. E.g, ''transform.RandomCrop'
        target_transform (callable, optional): A funciton/transform that tkes in the
            targt and transfomrs it.
    '''

    def __init__(self, args, split,  annotation_path='ICFG-PEDES/ICFG-PEDES.json', transform=None):
        self.root = args.dir
        self.max_length = args.max_length
        self.transform = transform
        self.cap_augment = args.cap_aug
        self.split = split.lower() 
        self.reid_raw_path = os.path.join(self.root, annotation_path)
        print("load annotations from", self.reid_raw_path)
        with open(self.reid_raw_path, 'r') as f:
            data = json.load(f)
            self.txt_data = self.data_parse(data)

        if args.embedding_type == 'BERT':
            tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')
        elif args.embedding_type == 'Roberta':
            tokenizer_class, pretrained_weights = (transformers.RobertaTokenizer, 'roberta-base')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)


        if not check_exists(self.root):
            print(self.root)
            raise RuntimeError('Dataset not found or corrupted.' +
                               'Please follow the directions to generate datasets')

        if self.split == 'train':
            data = self.txt_data["train"]
            self.train_labels = [int(i) for i in data['labels']]
            self.train_captions = data['captions']
            self.train_images = data['image_paths']
        elif self.split == 'val':
            data = self.txt_data["val"]
            self.val_labels = [int(i) - 3102 for i in data['labels']]
            self.val_captions = data['captions']
            self.val_images = data['image_paths']
        elif self.split == 'test':
            data = self.txt_data["test"]
            self.test_labels = [int(i) -3102 for i in data['labels']]
            self.test_captions = data['captions']
            self.test_images = data['image_paths']
        else:
            raise RuntimeError('Wrong split which should be one of "train","val" or "test"')

        if self.split == 'train':

            self.cap_aug_act = []
            if args.cap_wordnet > 0:
                self.cap_aug_act.append(naw.SynonymAug(aug_src='wordnet', aug_p=args.cap_wordnet))
            if args.cap_crop > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='crop', aug_p= args.cap_crop))
            if args.cap_delete > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='delete', aug_p= args.cap_delete))
            if args.cap_swap > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='swap', aug_p= args.cap_swap))

            # mixup 
            self.mixup = args.mixup

    def cap_transform(self, caption):
        if random.random() < 0.5:
            # 随机选一种来增强
            aug = random.choice(self.cap_aug_act)
            caption = aug.augment(caption)
        return caption

    def __len__(self):
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.val_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """

        if self.split == 'train':
            img_path, caption, label = 'ICFG-PEDES/imgs/'+self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, label = 'ICFG-PEDES/imgs/'+self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            img_path, caption, label = 'ICFG-PEDES/imgs/'+self.test_images[index], self.test_captions[index], self.test_labels[index]
        img_path = os.path.join(self.root, img_path)
        img = imread(img_path)

        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.split == 'train' and self.mixup:
            if random.random() < 0.5:
                rand_index = random.randint(0, len(self.train_captions)-1)
                rand_img_path, rand_caption = 'ICFG-PEDES/imgs/'+self.train_images[rand_index], self.train_captions[rand_index]
                rand_img = imread(os.path.join(self.root, rand_img_path))
                if len(rand_img.shape) == 2:
                    img = np.dstack((rand_img, rand_img, rand_img))
                rand_img = Image.fromarray(rand_img)
                rand_img = self.transform(rand_img)
                
                img = 0.5*(img+rand_img)
                caption += rand_caption
        label=torch.tensor(label)
        # print(caption, img.shape)

        if self.split =='train' and self.cap_augment:
            caption = self.cap_transform(caption)
        caption_id, attention_mask = self.cap_tokenize(caption, self.tokenizer)
        caption_id = torch.tensor(caption_id).long()
        attention_mask = torch.tensor(attention_mask).long()
        return img, caption_id, label, attention_mask, img_path

    def data_parse(self, data):
        phase = self.split
        # phases = ["train","val","test"]
        ret_dict = {}

        print('phase:', phase)
        if phase != "val":
            processing_data = [x for x in data if x['split']==phase]
        else:
            processing_data = [x for x in data if x['split']=='test']

        captions = []
        images_path = []
        labels = []
        for i in tqdm(processing_data):
            for j, cap in enumerate(i['captions']):
                captions.append(cap)
                images_path.append(i['file_path'])
                labels.append(i['id'])

        images_path = np.array(images_path)
        labels = np.array(labels)

        dict={'captions': captions, 'image_paths': images_path, 'labels': labels}
        ret_dict[phase] = dict
        return ret_dict

    def cap_tokenize(self, cap, tokenizer):
        tokenized = tokenizer.encode(cap, add_special_tokens=True)
        max_len = self.max_length
        
        if len(tokenized) < max_len:
            tokenized += [0] * (max_len-len(tokenized))
        else:
            tokenized = tokenized[:max_len]

        tokenized = np.array(tokenized)   
        attention_mask = np.where(tokenized != 0, 1, 0)

        return tokenized, attention_mask



class PETA_BERT_Token(data.Dataset):
    '''
    Args:
        root (string): Base root directory of dataset where [split].pkl and [split].h5 exists
        split (string): 'train', 'val' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed vector. E.g, ''transform.RandomCrop'
        target_transform (callable, optional): A funciton/transform that tkes in the
            targt and transfomrs it.
    '''

    def __init__(self, args, split,  annotation_path='PETA/PETA_captions.json', transform=None):
        self.root = args.dir
        self.max_length = args.max_length
        self.transform = transform
        self.cap_augment = args.cap_aug
        self.split = split.lower() 
        self.reid_raw_path = os.path.join(self.root, annotation_path)
        print("load annotations from", self.reid_raw_path)
        with open(self.reid_raw_path, 'r') as f:
            data = json.load(f)
            self.txt_data = self.data_parse(data)

        if args.embedding_type == 'BERT':
            tokenizer_class, pretrained_weights = (transformers.BertTokenizer, 'bert-base-uncased')
        elif args.embedding_type == 'Roberta':
            tokenizer_class, pretrained_weights = (transformers.RobertaTokenizer, 'roberta-base')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)


        if not check_exists(self.root):
            print(self.root)
            raise RuntimeError('Dataset not found or corrupted.' +
                               'Please follow the directions to generate datasets')

        if self.split == 'train':
            data = self.txt_data["train"]
            self.train_labels = [int(i) for i in data['labels']]
            self.train_captions = data['captions']
            self.train_images = data['image_paths']
        elif self.split == 'val':
            data = self.txt_data["val"]
            self.val_labels = [int(i) - 3102 for i in data['labels']]
            self.val_captions = data['captions']
            self.val_images = data['image_paths']
        elif self.split == 'test':
            data = self.txt_data["test"]
            self.test_labels = [int(i) -3102 for i in data['labels']]
            self.test_captions = data['captions']
            self.test_images = data['image_paths']
        else:
            raise RuntimeError('Wrong split which should be one of "train","val" or "test"')

        if self.split == 'train':

            self.cap_aug_act = []
            if args.cap_wordnet > 0:
                self.cap_aug_act.append(naw.SynonymAug(aug_src='wordnet', aug_p=args.cap_wordnet))
            if args.cap_crop > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='crop', aug_p= args.cap_crop))
            if args.cap_delete > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='delete', aug_p= args.cap_delete))
            if args.cap_swap > 0:
                self.cap_aug_act.append(naw.RandomWordAug(action='swap', aug_p= args.cap_swap))

            # mixup 
            self.mixup = args.mixup

    def cap_transform(self, caption):
        if random.random() < 0.5:
            # 随机选一种来增强
            aug = random.choice(self.cap_aug_act)
            caption = aug.augment(caption)
        return caption

    def __len__(self):
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.val_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """

        if self.split == 'train':
            img_path, caption, label = 'PETA/'+self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, label = 'PETA/'+self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            img_path, caption, label = 'PETA/'+self.test_images[index], self.test_captions[index], self.test_labels[index]
        img_path = os.path.join(self.root, img_path)
        img = imread(img_path)

        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.split == 'train' and self.mixup:
            if random.random() < 0.5:
                rand_index = random.randint(0, len(self.train_captions)-1)
                rand_img_path, rand_caption = 'PETA/'+self.train_images[rand_index], self.train_captions[rand_index]
                rand_img = imread(os.path.join(self.root, rand_img_path))
                if len(rand_img.shape) == 2:
                    img = np.dstack((rand_img, rand_img, rand_img))
                rand_img = Image.fromarray(rand_img)
                rand_img = self.transform(rand_img)
                
                img = 0.5*(img+rand_img)
                caption += rand_caption
        label=torch.tensor(label)

        if self.split =='train' and self.cap_augment:
            caption = self.cap_transform(caption)
        caption_id, attention_mask = self.cap_tokenize(caption, self.tokenizer)
        caption_id = torch.tensor(caption_id).long()
        attention_mask = torch.tensor(attention_mask).long()
        return img, caption_id, label, attention_mask, img_path

    def data_parse(self, data):
        phase = self.split
        ret_dict = {}

        print('phase:', phase)
        if phase != "val":
            processing_data = [x for x in data if x['split']==phase]
        else:
            processing_data = [x for x in data if x['split']=='test']

        captions = []
        images_path = []
        labels = []
        for i in tqdm(processing_data):
            for j, cap in enumerate(i['captions']):
                captions.append(cap)
                images_path.append(i['file_path'])
                labels.append(i['id'])

        images_path = np.array(images_path)
        labels = np.array(labels)
        dict={'captions': captions, 'image_paths': images_path, 'labels': labels}
        ret_dict[phase] = dict
        return ret_dict

    def cap_tokenize(self, cap, tokenizer):
        tokenized = tokenizer.encode(cap, add_special_tokens=True)
        max_len = self.max_length
        
        if len(tokenized) < max_len:
            tokenized += [0] * (max_len-len(tokenized))
        else:
            tokenized = tokenized[:max_len]

        tokenized = np.array(tokenized)   
        attention_mask = np.where(tokenized != 0, 1, 0)

        return tokenized, attention_mask



if __name__ == '__main__':
    # import torchvision.transforms as transforms
    from train_config import parse_args
    args=parse_args()
    # args.embedding_type='BERT'
    # args.max_length = 64
    # args.batch_size=64
    # transform_val_list = [
    #     transforms.Resize((384, 128)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ]
    # split = 'train'
    # transform=transforms.Compose(transform_val_list)
    # data_split = CUHKPEDES_BERT_Token(args, split, annotation_path='CUHK-PEDES/reid_raw.json',  transform=data_transforms[split])
    # # data_split = CUHKPEDES_BERT_Token(args.dir, split, args.max_length, transform=transform, embedding_type=args.embedding_type)
    # # data_split = RSTPReid_BERT_Token(args.dir, split, args.max_length,transform=transform, embedding_type=args.embedding_type)

    # loader = data.DataLoader(data_split, args.batch_size, shuffle=False, num_workers=8)
    # sample=next(iter(loader))
    # img, caption, label, mask, img_path=sample
    # print(img.shape, caption, label, mask, img_path)
    # print(img.shape)
    # # print(caption.shape)

    # for step, (images, captions, labels, mask, img_paths) in enumerate(loader):
    #     continue
    # # print(caption)