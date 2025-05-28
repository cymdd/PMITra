from utils import *
from basemodel import *
from clas_model import *
import torch.nn as nn
import torch.nn.functional as F
import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class GaussianNLLLoss(nn.Module):
    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = 0.5*(torch.log(scale**2) + torch.abs(target - loc)**2 / scale**2)
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class PMIra(nn.Module):
    def __init__(self, args, Temperal_Encoder_state_dict):
        super(PMIra, self).__init__()
        self.args = args
        self.batch_norm_gt = [{},{}]
        self.pre_obs = [{},{}]
        self.Temperal_Encoder_net = EncoderTrainer(self.args)
        if Temperal_Encoder_state_dict != None:
            self.Temperal_Encoder_net.load_state_dict(Temperal_Encoder_state_dict)

        self.Laplacian_Decoder = Laplacian_Decoder(self.args)
        self.AttentionModule = AttentionModule(self.args.hidden_size)
        self.Global_interactionlist = [nn.ModuleList() for _ in range(self.args.cluster_num)]
        self.Global_interaction = Global_interaction(self.args)
        for globnum in range(len(self.Global_interactionlist)):
            self.Global_interactionlist[globnum].extend(
                self.Global_interaction for _ in range(self.args.message_layers))
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

    def forward(self, inputs_0, inputs_1, all_labels, all_abs, outbatch,iftest):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if iftest:
            train_x_0, train_y_0, nei_list_batch_0, batch_split_0, batch_abs_gt_0, nei_num_batch_0 = self.getTrainX(
                inputs_0, device, 0)
            _, full_hidden_0, self.x_encoded_dense_0, self.hidden_state_unsplited_0, cn_0 =  self.Temperal_Encoder_net.forward(
                train_x_0, True)  # [N, D], [N, D]
            cluster_prob_0 = get_scene_data(batch_abs_gt_0, all_abs, all_labels, self.args)

            self.hidden_state_global_0 = hidden_state_global_0 = torch.zeros_like(self.hidden_state_unsplited_0, device=device)
            cn_global_0 = torch.zeros_like(cn_0, device=device)

            for index, batch in enumerate(batch_split_0):
                left_0, right_0 = batch[0], batch[1]
                element_states_0 = self.hidden_state_unsplited_0[left_0: right_0]  # [N, D]
                cn_state_0 = cn_0[left_0: right_0]  # [N, D]
                if element_states_0.shape[0] != 1:
                    self.hidden_state_global_0, cn_global_0, _= self.Graph_conv_interaction(batch_abs_gt_0, left_0,right_0,
                                                                                          element_states_0,cn_state_0,nei_list_batch_0,
                                                                                          index, device,cluster_prob_0,
                                                                                          hidden_state_global_0,cn_global_0)
                else:
                    self.hidden_state_global_0[left_0: right_0] = element_states_0
                    cn_global_0[left_0: right_0] = cn_state_0
            mdn_out = self.Laplacian_Decoder.forward(self.x_encoded_dense_0, self.hidden_state_global_0, cn_global_0)
            GATraj_loss, full_pre_tra = self.mdn_loss(train_y_0.permute(2, 0, 1), mdn_out)  # [K, H, N, 2]
            return GATraj_loss, full_pre_tra
        else:
            train_x_0, train_y_0, nei_list_batch_0, batch_split_0, batch_abs_gt_0, nei_num_batch_0 = self.getTrainX(
                inputs_0, device, 0)
            train_x_1, train_y_1, nei_list_batch_1, batch_split_1, batch_abs_gt_1, nei_num_batch_1 = self.getTrainX(
                inputs_1, device, 1)

            _,full_hidden_0, self.x_encoded_dense_0, self.hidden_state_unsplited_0, cn_0 =  self.Temperal_Encoder_net.forward(train_x_0, True)  # [N, D], [N, D]
            _,full_hidden_1, self.x_encoded_dense_1, self.hidden_state_unsplited_1, cn_1 =  self.Temperal_Encoder_net.forward(train_x_1, True)  # [N, D], [N, D]

            cluster_prob_0 = get_scene_data(batch_abs_gt_0, all_abs, all_labels, self.args)
            cluster_prob_1 = get_scene_data(batch_abs_gt_1, all_abs, all_labels, self.args)

            split_refle = balance_mapping(len(batch_split_0), len(batch_split_1))
            self.hidden_state_global_0 = hidden_state_global_0 = torch.zeros_like(self.hidden_state_unsplited_0, device=device)
            self.hidden_state_global_1 = hidden_state_global_1 = torch.zeros_like(self.hidden_state_unsplited_1, device=device)
            cn_global_0 = torch.zeros_like(cn_0, device=device)
            cn_global_1= torch.zeros_like(cn_1, device=device)
            # 分布对齐损失初始化
            pattern_num = 0
            pattern_loss=0

            for index, batch in enumerate(split_refle):
                [b0,b1] = batch
                left_0, right_0 = batch_split_0[b0][0], batch_split_0[b0][1]
                left_1, right_1 = batch_split_1[b1][0], batch_split_1[b1][1]
                element_states_0 = self.hidden_state_unsplited_0[left_0: right_0]  # [N, D]
                element_states_1 = self.hidden_state_unsplited_1[left_1: right_1]  # [N, D]
                cn_state_0 = cn_0[left_0: right_0]  # [N, D]
                cn_state_1 = cn_1[left_1: right_1]  # [N, D]
                if element_states_0.shape[0] != 1:
                    self.hidden_state_global_0, cn_global_0, single_GCNelem_0= self.Graph_conv_interaction(batch_abs_gt_0, left_0,
                                                                                                        right_0, element_states_0,cn_state_0,nei_list_batch_0,
                                                                                                        b0, device, cluster_prob_0, hidden_state_global_0, cn_global_0)
                else:
                    single_GCNelem_0 = []
                    self.hidden_state_global_0[left_0: right_0] = element_states_0
                    cn_global_0[left_0: right_0] = cn_state_0
                if element_states_1.shape[0] != 1:
                    self.hidden_state_global_1, cn_global_1, single_GCNelem_1 = self.Graph_conv_interaction(batch_abs_gt_1, left_1, right_1, element_states_1,
                                                                                              cn_state_1,nei_list_batch_1, b1, device, cluster_prob_1,
                                                                                              hidden_state_global_1, cn_global_1)
                else:
                    single_GCNelem_1 = []
                    self.hidden_state_global_1[left_1: right_1] = element_states_1
                    cn_global_1[left_1: right_1] = cn_state_1

                cur_pattern_loss = self.pattern_Loss(single_GCNelem_0,single_GCNelem_1,device)
                pattern_loss += cur_pattern_loss
                if pattern_loss != 0:
                    pattern_num += 1

            if pattern_num != 0:
                     pattern_loss /= pattern_num
            mdn_out = self.Laplacian_Decoder.forward(self.x_encoded_dense_0, self.hidden_state_global_0, cn_global_0)
            GATraj_loss, full_pre_tra = self.mdn_loss(train_y_0.permute(2, 0, 1), mdn_out)  # [K, H, N, 2]
            return GATraj_loss, pattern_loss, full_pre_tra

    def Graph_conv_interaction(self, batch_abs_gt, left, right, element_states, cn_state, nei_list_batch, b, device, cluster_prob, hidden_state_global, cn_global):
        # --------set1------------#
        corr = batch_abs_gt[self.args.obs_length - 1, left: right, :2].repeat(
            element_states.shape[0], 1, 1)  # [N, N, D]
        corr_index = corr.transpose(0, 1) - corr  # [N, N, D]
        nei_index = torch.tensor(nei_list_batch[b][self.args.obs_length - 1], device = device)  # [N, N]
        origin_element_states = element_states.clone()
        origin_cn_state = cn_state.clone()
        single_GCNelem = []
        zero_elem = torch.zeros_like(element_states[0], device=device)
        for traj_index in range(left, right):
            if left != 0:
                cur_rela_index = traj_index % left
            else:
                cur_rela_index = traj_index
            ori_prob = cluster_prob[traj_index]
            upda_prob = update_probabilities(ori_prob)
            element_states = origin_element_states
            cn_state = origin_cn_state
            single_GCNelem.append([])
            for globnum in range(len(self.Global_interactionlist)):
                # 概率为0的图卷积网络不参与
                if (upda_prob[globnum] == 0):
                    single_GCNelem[-1].append(zero_elem.tolist())
                    continue
                for i in range(len(self.Global_interactionlist[globnum])):
                    element_states, cn_state = self.Global_interactionlist[globnum][i](corr_index, nei_index,element_states,cn_state)
                cur_elem = element_states[cur_rela_index] * upda_prob[globnum]
                hidden_state_global[traj_index] += cur_elem
                cn_global[traj_index] += cn_state[cur_rela_index] * upda_prob[globnum]
                single_GCNelem[-1].append(cur_elem.tolist())
        return hidden_state_global, cn_global,single_GCNelem

    def getTrainX(self,inputs,device,setNum):
        batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split = inputs  # #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H], [B, 2]
        self.batch_norm_gt[setNum] = batch_norm_gt
        #Obtain the motion offset between time steps, taking the last time step of the observed trajectory as the origin
        train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length - 1, :,:]  # [H, N, 2]

        train_x = train_x.permute(1, 2, 0)  # [N, 2, H]
        train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0)  # [N, 2, H]
        self.pre_obs[setNum] = batch_norm_gt[1:self.args.obs_length]
        return train_x,train_y,nei_list_batch,batch_split,batch_abs_gt,nei_num_batch

    def mdn_loss(self, y, y_prime):
        batch_size=y.shape[1]
        y = y.permute(1, 0, 2)  #[N, H, 2]
        # [F, N, H, 2], [F, N, H, 2], [N, F]
        out_mu, out_sigma, out_pi = y_prime 
        y_hat = torch.cat((out_mu, out_sigma), dim=-1)
        reg_loss, cls_loss = 0, 0
        full_pre_tra = []
        l2_norm = (torch.norm(out_mu - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(batch_size)]
        reg_loss = reg_loss + self.reg_loss(y_hat_best, y)
        soft_target = F.softmax(-l2_norm / self.args.pred_length, dim=0).t().detach() # [N, F]
        cls_loss = cls_loss + self.cls_loss(out_pi, soft_target)
        loss = reg_loss + cls_loss
        #best ADE
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs[0],sample_k), axis=0))
        # best FDE
        l2_norm_FDE = (torch.norm(out_mu[:,:,-1,:] - y[:,-1,:], p=2, dim=-1) )  # [F, N]
        best_mode = l2_norm_FDE.argmin(dim=0)
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs[0],sample_k), axis=0))
        return loss, full_pre_tra

    def pattern_Loss(self,single_GCNelem_0, single_GCNelem_1,device):
        if single_GCNelem_0 == [] or single_GCNelem_1 == []:
            return 0
        else:
            partternloss = 0
            final_array_0 = torch.tensor(single_GCNelem_0,device=device)
            final_array_1 = torch.tensor(single_GCNelem_1,device=device)
            for gcn_num in range(self.args.cluster_num):
                sour_curgcn_elem = final_array_0[:,gcn_num]
                tar_curgcn_elem = final_array_1[:,gcn_num]
                sour_curgcn_sum = self.AttentionModule.forward(sour_curgcn_elem)
                tar_curgcn_sum = self.AttentionModule.forward(tar_curgcn_elem)
                euclidean_distance_squared = (sour_curgcn_sum - tar_curgcn_sum).norm()
                partternloss = partternloss + euclidean_distance_squared
            return partternloss


    
