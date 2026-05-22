from config import get_parser
from date_loader import Data_loader
from main import os,Main
from utils import load_arg,save_arg
import torch

def train(args,parser):
    source_domain = args.source_domain
    target_domain = args.target_domain
    #-----create a storage folder-----
    model_dirName = f'{source_domain}2{target_domain}'
    args.model_filepath = os.path.join(args.checkpoint_dir, model_dirName)
    if not os.path.exists(args.model_filepath):
        os.makedirs(args.model_filepath)
        print(f"folder '{model_dirName}' has been created in '{args.checkpoint_dir}'")
    #-----load/create the configuration file----
    args.phase = 'train'
    args.config = args.model_filepath+'/config_'+args.phase+'.yaml'
    if not load_arg(args,parser):
        save_arg(args)
    args = load_arg(args,parser)
    Dataloader = Data_loader(args, phase="train")
    main = Main(args, Dataloader)
    train_sequence = [0, 1]
    for item in train_sequence:
        main.playAllTrain(item)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.using_cuda:
        torch.cuda.set_device(args.gpu)
    checkpoint_dir = "./checkpoints/"
    args.checkpoint_dir = checkpoint_dir
    if os.path.exists(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)
    train(args,parser)
