from config import get_parser
from date_loader import Data_loader
from main import os,Main
from utils import load_arg,save_arg
import torch

def test(args,parser):
    domains = ['eth', 'hotel', 'zara01', 'zara02', 'students001', 'students003', 'uni_examples', 'zara03']
    # Select the source domain and target domain based on domains:
    # For example:  source_domain = 0   ->  source domain: eth
    #               target_domain = 1   -> target domain: hotel
    source_domain = 0
    target_domain = 1
    train_set = [source_domain, target_domain]
    #-----check the model parameter file-----
    model_dirName = f'{domains[train_set[0]]}2{domains[train_set[1]]}'
    args.model_filepath = os.path.join(args.checkpoint_dir, model_dirName)
    if not os.path.exists(args.model_filepath):
        print("The model parameter file does not exist！")
    #-----load/create the configuration fil----
    args.config = args.model_filepath+'/config_'+args.phase+'.yaml'
    if not load_arg(args,parser):
        save_arg(args)
    args = load_arg(args,parser)
    Dataloader = Data_loader(args, train_set, phase="test")
    main = Main(args, Dataloader)
    main.playtest()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.using_cuda:
        torch.cuda.set_device(args.gpu)
    checkpoint_dir = "./checkpoints/"
    args.checkpoint_dir = checkpoint_dir
    if os.path.exists(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)
    test(args,parser)