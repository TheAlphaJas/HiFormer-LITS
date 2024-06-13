import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_lits import RandomRotationTransform, LITSDataset, LITSTestDataset

from utils import test_single_volume,random_split_array

from models.HiFormer import HiFormer
import configs.HiFormer_configs as configs 

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/test_vol_h5', help='root dir for data')
parser.add_argument('--model_weight', type=str,
                    default=None, help='path for model weights')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--model_name', type=str,
                    default='hiformer-b', help='[hiformer-s, hiformer-b, hiformer-l]')
parser.add_argument('--is_liver', action='store_true',
                    default=0, help='add for liver, remove for tumor')

args = parser.parse_args()


def inference(args, testloader, model):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label = sampled_batch["image"], sampled_batch["label"]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size])
        metric_list += np.array(metric_i)
        logging.info(' idx %d mean_dice %f mean_hd95 %f' % (i_batch, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    metric_list = metric_list / len(db_test)

    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]

    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    return "Testing Finished!"



if __name__ == "__main__":
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


    args.is_pretrain = True
    if (args.is_liver):
      organ = "liver"
    else:
      organ = "cancer"
    
    model = HiFormer(config=CONFIGS[args.model_name], img_size=args.img_size, n_classes=args.num_classes).cuda()
    msg = model.load_state_dict(torch.load(args.model_weight))
    print("HiFormer Model: ", msg)

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)

    logging.basicConfig(filename=log_folder + '/' + args.model_name + ".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    test_save_path = None

    #Splitting Dataset    
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

    scan_list = os.listdir(args.root_path)
    scan_list.sort()
    for i in scan_list:
      num = int(i.split("_")[-1])
      path = os.path.join(args.root_path,i)
      imgpath = os.path.join(path,"images")
      maskpath = os.path.join(path,"masks")
      piclist = os.listdir(imgpath)
      if num in train:
        for j in piclist:
            X_train.append(os.path.join(imgpath,j))
            Y_train.append(os.path.join(os.path.join(maskpath,organ),j))
      elif num in test:
        for j in piclist:
            X_test.append(os.path.join(imgpath,j))
            Y_test.append(os.path.join(os.path.join(maskpath,organ),j))
      else:
        for j in piclist:
            X_val.append(os.path.join(imgpath,j))
            Y_val.append(os.path.join(os.path.join(maskpath,organ),j))

    db_test = LITSTestDataset(X_test,Y_test,transform=None)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    inference(args, testloader, model)