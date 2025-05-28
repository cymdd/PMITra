import torch
import copy
import numpy as np
import os
import yaml

def L2forTest(outputs,targets,obs_length):
    seq_length = outputs.shape[0]
    error = torch.norm(outputs-targets,p=2,dim=2)
    error_pred_length = error[obs_length-1:]
    error = torch.sum(error_pred_length)
    error_cnt = error_pred_length.numel()
    if error == 0:
        return 0,0,0,0,0,0
    final_error = torch.sum(error_pred_length[-1])
    final_error_cnt = error_pred_length[-1].numel()
    first_erro = torch.sum(error_pred_length[0])
    first_erro_cnt = error_pred_length[0].numel()
    return error.item(),error_cnt,final_error.item(),final_error_cnt,first_erro.item(),first_erro_cnt

def displacement_error(pred, pred_gt, mode='raw'):
    assert pred_gt.shape[1:] == pred.shape[1:]
    loss = pred_gt - pred
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred, pred_gt, mode='raw'):
    assert pred_gt.shape[1:] == pred.shape[1:]
    loss = pred_gt - pred
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)

def display_performance(perf_dict):
    print("==> Current Performances (ADE & FDE):")
    for a, b in perf_dict.items():
        c = copy.deepcopy(b)
        if isinstance(c, list):
            c[0] = np.round(c[0], 4)
            c[1] = np.round(c[1], 4)
        print("   ", a, c)

def load_arg(p, parser):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            # default_arg = yaml.load(f,Loader=yaml.FullLoader)
            default_arg = yaml.safe_load(f)

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s=1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False

def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_filepath):
        os.makedirs(args.model_filepath)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def balance_mapping(large_number,small_number):
    # large_number = len(large_list)
    # small_number = len(small_list)
    quotient = large_number // small_number
    remainder = large_number % small_number

    # Initialize the list of mapping results
    mapping = []
    currentIndex=0
    # small_number
    for right in range(small_number):
        # Calculate the number of large number terms that the current decimal term should match
        current_match_count = quotient + (1 if remainder>0 else 0)
        current_match_count += currentIndex
        # Add each matching large number item
        for left in range(currentIndex,current_match_count):
            mapping.append([left,right])
        currentIndex = current_match_count
        remainder-=1
    return mapping

def update_probabilities(probabilities):
    probabilities = torch.tensor(probabilities, dtype=torch.float32)
    probabilities = probabilities.clone().detach()
    # Retain the values with a probability greater than the threshold and set the other values to 0
    max_val = max(probabilities)
    min_val = min(probabilities)
    # Remove the maximum and minimum values
    trimmed_arr = [x for x in probabilities if x != max_val and x != min_val]
    # Calculate the average value of the remaining values
    if len(trimmed_arr) == 0:
        return probabilities
    Threshold = sum(trimmed_arr) / len(trimmed_arr)
    # Threshold = torch.max(probabilities)
    updated_probs = torch.where(probabilities < Threshold, torch.zeros_like(probabilities), probabilities)
    # Calculate the sum of the remaining probabilities
    remaining_sum = torch.sum(updated_probs)
    if torch.isclose(remaining_sum, torch.tensor(0.0)):
        return probabilities
    # Readjust the sum of the remaining probability values to 1
    updated_probs /= remaining_sum

    return updated_probs.to('cuda')

def modifyArgsfile(file_path, key, value):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cluster_num_line = None
    for i, line in enumerate(lines):
        if line.startswith(key):
            cluster_num_line = i
            break

    if cluster_num_line is not None:
        del lines[cluster_num_line]

    new_line = f'{key}: {value}\n'
    lines.insert(cluster_num_line, new_line)

    with open(file_path, 'w') as file:
        file.writelines(lines)

    print(f'{key} has been updated to {value}')