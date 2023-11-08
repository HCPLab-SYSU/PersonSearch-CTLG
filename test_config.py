import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')
    parser.add_argument('--name', default='Experiment01', type=str, help='output model name')

    #model_name
    parser.add_argument('--model', type=str,
                    default="ViT_Bert",
                    help='model definition')

    #dataset_Directory
    parser.add_argument('--dir', type=str,
                        default=r'./dataset',
                        help='directory to store dataset')
    parser.add_argument('--dataset', type=str,
                        default="CUHKPEDES")

    ##save_Directory
    parser.add_argument('--checkpoint_dir', type=str,
                        default="./log",
                        help='directory to store checkpoint')
    parser.add_argument('--log_dir', type=str,
                        default="./log",
                        help='directory to store log')

    #word_embedding
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--embedding_type', type=str,
                        default='BERT')

    #image setting
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=384)

    #ViT setting
    parser.add_argument('--num_classes', type=int, default=11003)
    parser.add_argument('--feature_size', type=int, default=768)
    parser.add_argument('--stride_size', type=int, default=16)

    parser.add_argument('--pretrained', action='store_false',
                       help='whether or not to restore the pretrained visual model')
    parser.add_argument('--droprate', default=0, type=float, help='drop rate')

    #experiment setting
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoches', type=int, default=70)
    parser.add_argument('--resume', type=str,
                        default="",
                        help='restore the pretrained whole model')


    #Optimization setting
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--wd', type=float, default=0.00004)

    #adam_setting
    parser.add_argument('--adam_lr', type=float, default=0.0001, help='the learning rate of adam')
    parser.add_argument('--adam_alpha', type=float, default=0.9)
    parser.add_argument('--adam_beta', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)

    parser.add_argument('--lr_decay_type', type=str, default='MultiStepLR',
                        help='One of "MultiStepLR" or "StepLR" or "ReduceLROnPlateau"')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--epoches_decay', type=str, default='20', help='#epoches when learning rate decays')
    parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')

    # Default setting
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='0')

    # transfer learning
    parser.add_argument('--clip', action='store_true',
                    help='use clip pretrain')

    # loss hyperparameters
    parser.add_argument('--temperature',  type=float, default=0.02)

    # caption augment
    parser.add_argument('--cap_aug', action='store_true', help='whether to use caption augment')
    parser.add_argument('--cap_wordnet', type=float, default=0.2)
    parser.add_argument('--cap_crop', type=float, default=0.2)
    parser.add_argument('--cap_delete', type=float, default=0.2)
    parser.add_argument('--cap_swap', type=float, default=0)

    # mixup
    parser.add_argument('--mixup', action='store_true', help='whether to use cross-modal mixup augment')

    # feature mixup
    parser.add_argument('--feature_mixup', action='store_true', help='whether to use cross-modal feature mixup augment')

    args = parser.parse_args()
    return args
