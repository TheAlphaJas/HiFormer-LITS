import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.HiFormer import HiFormer
import configs.HiFormer_configs as configs 
from trainer import trainer


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='./data/Synapse/test_vol_h5', help='root dir for data')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=401, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=10, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--num_workers', type=int,  default=2,
                    help='number of workers')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--output_dir', type=str,
                    default='./results', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='hiformer-b', help='[hiformer-s, hiformer-b, hiformer-l]')
parser.add_argument('--eval_interval', type=int,
                    default=20, help='evaluation epoch')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')

args = parser.parse_args()

args.output_dir = args.output_dir + f'/{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    CONFIGS = {
        'hiformer-s': configs.get_hiformer_s_configs(),
        'hiformer-b': configs.get_hiformer_b_configs(),
        'hiformer-l': configs.get_hiformer_l_configs(),
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24


    model = HiFormer(config=CONFIGS[args.model_name], img_size=args.img_size, n_classes=args.num_classes).cuda()
    
    def random_split_array(data, ratios=(0.8, 0.1, 0.1)):
      np.random.shuffle(data)  # Shuffle for randomness
      split_indices = np.cumsum(ratios[:-1]) * len(data)
      return np.split(data, split_indices.astype(int))


    original = []
    for i in range(131) :
      original.append(i)

    train, test, val = random_split_array(original,(0.8,0.1,0.1))
    print(train)
    print(test)
    print(val)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_val = []
    Y_val = []

    scan_list = os.listdir("//kaggle/working/Task03_Liver")
    scan_list.sort()
    for i in scan_list:
      num = int(i.split("_")[-1])
      path = "/kaggle/working/Task03_Liver/" + i
      imgpath = path + "/images"
      maskpath = path + "/masks"
      piclist = os.listdir(imgpath)
      if num in train:
        for j in piclist:
            X_train.append(imgpath + "/" + j)
            Y_train.append(maskpath + "/liver/" +j)
      elif num in test:
        for j in piclist:
            X_test.append(imgpath + "/" + j)
            Y_test.append(maskpath + "/liver/" +j)
      else:
        for j in piclist:
            X_val.append(imgpath + "/" + j)
            Y_val.append(maskpath + "/liver/" +j)
    print(len(X_train))     
    print("Train length : ", len(X_train))
    print("Test length : ", len(X_test))
    print("Val length : ", len(X_val))
    trainer(args, model, args.output_dir, X_train, Y_train, X_val, Y_val)
