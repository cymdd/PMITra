from utils import *
from basemodel import EncoderTrainer
from clas_model import *
import torch
import time
import torch.nn as nn
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Main():
    def __init__(self, args, Dataloader):
        self.args = args
        self.lr=self.args.learning_rate
        self.dataloader_gt = Dataloader
        self.epoch = 0

    def save_model(self,epoch):
        model_path= self.args.model_filepath + '/' + self.args.train_model + '_' +\
                                   str(epoch) + '.pth'
        checkpoint = {
            'epoch': epoch,
            'net': self.net,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler
        }
        torch.save(checkpoint, model_path,_use_new_zipfile_serialization=False)
        self.args.load_model=epoch
        modifyArgsfile(self.args.config, 'load_model', epoch)


    def load_model(self):
        if self.args.load_model >= 0:
            self.args.model_save_path = self.args.model_filepath + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.pth'
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                #cuda:1 The gpu number stored in the model parameter file
                checkpoint = torch.load(self.args.model_save_path,map_location={'cuda:1': 'cuda:'+str(self.args.gpu)})
                model_epoch = checkpoint['epoch']
                self.epoch = int(model_epoch)+1
                self.net = checkpoint['net']
                self.optimizer = checkpoint['optimizer']
                self.scheduler = checkpoint['scheduler']
                print('Loaded checkpoint at epoch', model_epoch)


    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss(reduce=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,\
        T_max = self.args.num_epochs, eta_min=self.args.eta_min)

    def playtest(self):
        print('Testing begin')
        model_filepath = os.path.join(self.args.model_filepath,"best_model.pth")
        if os.path.exists(model_filepath):
            model_state_dict = torch.load(model_filepath,map_location={'cuda:1': 'cuda:'+str(self.args.gpu)})
            self.net = model_state_dict
            self.net.args = self.args
            test_error, test_final_error, first_erro_test = self.test_epoch()
            print('Set: {}, test_error: {:.5f} test_final_error: {:.5f}'.format(self.args.test_set,\
                                                                 test_error,test_final_error))
        else:
            print("No model weight file!")
            return

    def playEntireTrain(self, Temperal_Encoder_state_dict, perf_dict, dict_key, all_labels, all_abs):
        torch.cuda.empty_cache()
        print(self.args.load_model)
        if self.args.load_model >= 0:
            model = import_class(self.args.model)
            self.net = model(self.args, None)
            self.set_optimizer()
            self.load_model()
        else:
            model = import_class(self.args.model)
            self.net = model(self.args, Temperal_Encoder_state_dict)
            self.set_optimizer()
            self.epoch=0
            if self.args.using_cuda:
                self.net = self.net.cuda()
        print('Training begin')
        epochs_tqdm = tqdm(range(self.epoch,self.args.num_epochs))
        start = time.time()
        for epoch in epochs_tqdm:
            with torch.no_grad():
                print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))
            train_loss = self.train_epoch(epoch, all_labels, all_abs)
            val_loss, val_error, val_final_error, val_erro_first,valtime = self.val_epoch(all_labels, all_abs)
            val_res = [val_error,val_final_error]
            with torch.no_grad():
                print(f'epoch={epoch} | val_loss={val_loss} | time={valtime}')
                val_loss_logfilepath = os.path.join(self.args.model_filepath, 'val_loss_log.txt')
                cur_valloss_content = f'epoch={epoch+1} | valid_error={val_error} | valid_final={val_final_error} | time={valtime}'
                with open(val_loss_logfilepath, 'a') as file:
                    file.write(cur_valloss_content+ '\n')
            self.scheduler.step()
            self.save_model(epoch)
            with torch.no_grad():
                print('----epoch {} \n train_loss={:.5f}, valid_error={:.3f}, valid_final={:.3f}, valid_first={:.3f}'\
                    .format(epoch, train_loss,val_error, val_final_error,val_erro_first))
                if val_res[1] < perf_dict[dict_key][1]:
                    perf_dict[dict_key][0], perf_dict[dict_key][1] = val_res[0], val_res[1]
                    torch.save(self.net, os.path.join(self.args.model_filepath, "best_model.pth"))
                    with open(os.path.join(self.args.model_filepath, "Performances.pkl"), "wb") as f:
                        pickle.dump(perf_dict, f, 4)
                    print("==>best_model Saved")
            if epoch%10 == 0:
                print(self.net.Global_interactionlist)
            epochs_tqdm.update(1)
        end = time.time()
        traintime = end-start
        epochs_tqdm.close()
        torch.save(self.net, os.path.join(self.args.model_filepath,f"epoch{self.args.num_epochs}_bs{self.args.batch_size}_wholeModel.pth"))
        train_timeContent = f'train_time = {traintime/3600} H'
        print(train_timeContent)
        with open(val_loss_logfilepath, 'a') as file:
            file.write(train_timeContent + '\n')
            file.write(f'graphconv = {self.args.cluster_num}' + '\n')

    def get_inputsfw(self,batch,epoch,setNum, isval):
        # inputs_gt: the coordinates of all trajectory sequences at each time step
        # batch_split: indexes of trajectory sequences in different scenarios
        # nei_lists: whether a pedestrian who steps down at each time is still a neighbor
        if isval:
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_val_batch(batch, epoch)
        else:
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_train_batch(batch,epoch,setNum)  # batch_split:[batch_size, 2]
        inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
        if self.args.using_cuda:
            inputs_gt = tuple([i.cuda() for i in inputs_gt])
        batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
        inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split  # [H, N, 2], [H, N, 2], [B, H, N, N], [N, H]
        return inputs_fw

    def train_epoch(self, epoch, all_labels, all_abs):
        self.net.train()
        loss_epoch=0
        ballance_array = balance_mapping(self.dataloader_gt.trainbatchnums,self.dataloader_gt.valbatchnums)
        for index, batch in enumerate(ballance_array):
            start = time.time()

            inputs_fw_0 = self.get_inputsfw(batch[0], epoch, 0, False)
            inputs_fw_1 = self.get_inputsfw(batch[1], epoch, None,True)
            self.net.zero_grad()
            predict_loss,pattern_loss, full_pre_tra = self.net.forward(inputs_fw_0, inputs_fw_1, all_labels, all_abs, batch,iftest=False)
            if predict_loss == 0 or pattern_loss == 0:
                continue
            ballance_param = 0.4
            totalLoss = ballance_param  * predict_loss + (1-ballance_param) * pattern_loss
            print(f'predict_loss = {predict_loss} | pattern_loss = {pattern_loss}')
            loss_epoch = loss_epoch + totalLoss.item()
            totalLoss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            end= time.time()
            with torch.no_grad():
                print('train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f}'.\
                format(index+1,len(ballance_array), epoch, totalLoss, end - start))
        train_loss_epoch = loss_epoch / len(ballance_array)
        return train_loss_epoch

    def playtrain_TemperEncoder(self, net, perf_dict, dict_key, loss_fn):
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.args.num_epochs, eta_min=self.args.eta_min)
        save_dir_base = self.args.model_filepath

        all_indices = list(range(self.dataloader_gt.trainbatchnums))
        train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

        start_time = time.time()
        print("#################### TemperEncoder train begin ####################")
        for epoch in tqdm(range(self.args.TemperEncoder_epochs), desc="Epoch"):
            train_loss, val_loss = 0, 0
            val_res = np.zeros(2).astype(float)
            net.train()
            for index, batch in enumerate(train_indices):
                optimizer.zero_grad()
                inputs_fw_0 = self.get_inputsfw(batch, epoch, 0, False)
                train_x_0, train_y_0 = self.getTrainXandY(inputs_fw_0)
                output, _, _, _, _= net.forward(train_x_0,False)
                train_y_0 = train_y_0.transpose(1, 2)

                l2_norm = torch.norm(output[0]-train_y_0, p=2,dim=-1).sum(dim=-1)
                loss = l2_norm.sum()/len(l2_norm)/self.args.pred_length
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            train_loss /= (self.dataloader_gt.trainbatchnums-1)

            net.eval()
            with torch.no_grad():
                valdatalen = 0
                for index, batch in enumerate(val_indices):
                    inputs_fw_0 = self.get_inputsfw(batch, None, 0, False)
                    _, batch_norm_gt, _, _, _ = inputs_fw_0
                    val_x_0, val_y_0 = self.getTrainXandY(inputs_fw_0)
                    val_output, _, _, _, _ = net.forward(val_x_0, False)
                    val_y_0 = val_y_0.transpose(1, 2)
                    l2_norm = torch.norm(val_output[0] - val_y_0, p=2, dim=-1).sum(dim=-1)
                    val_loss = l2_norm.sum() / len(l2_norm) / self.args.pred_length
                    val_loss += loss.item()
                    ADE = displacement_error(val_output[0], val_y_0, mode="sum").item() / self.args.pred_length
                    FDE = final_displacement_error(val_output[0][:, -1], val_y_0[:, -1], mode="sum").item()
                    val_res += [ADE, FDE]
                    valdatalen += val_output[0].shape[0]
            val_res /= valdatalen

            if epoch % 1 == 0:
                end_time = time.time()
                with torch.no_grad():
                    print("==> Epoch: %d | Train Loss: %.4f | Val Loss: %.4f | Val ADE: %.4f | Val FDE: %.4f | Time: %.4f"
                          % (epoch, train_loss, val_loss, val_res[0], val_res[1], end_time - start_time))
                start_time = end_time

            if val_res[1] < perf_dict[dict_key][1]:
                perf_dict[dict_key][0], perf_dict[dict_key][1] = val_res[0], val_res[1]
                torch.save(net.state_dict(), os.path.join(save_dir_base, dict_key + ".pth"))
                with open(os.path.join(save_dir_base, "Performances.pkl"), "wb") as f:
                    pickle.dump(perf_dict, f, 4)
                print("==> Saved")
        print("#################### TemperEncoder train finished ####################")

    def getClasDataAndClusterNum(self,TemperNet):
        labels_filepath = os.path.join(self.args.model_filepath, 'cluster_labels.pkl')
        allabs_filepath = os.path.join(self.args.model_filepath, 'allabs.pkl')
        if os.path.exists(labels_filepath) and os.path.exists(allabs_filepath):
            with open(labels_filepath, "rb") as f1:
                labels = pickle.load(f1)
            with open(allabs_filepath, "rb") as f2:
                all_abs = pickle.load(f2)
        else:
            all_hidden,alltra_ab = [], []
            # data load
            for batch in range(self.dataloader_gt.trainbatchnums):
                inputs_fw_0 = self.get_inputsfw(batch, None, 0, False)
                alltra_ab.append(inputs_fw_0[0].transpose(1,0))
                train_x_0, _ = self.getTrainXandY(inputs_fw_0)
                _, full_hidden, _, _, _ = TemperNet.forward(train_x_0, True)
                all_hidden.append(full_hidden)
            for batch in range(self.dataloader_gt.valbatchnums):
                inputs_fw_1 = self.get_inputsfw(batch, None, None, True)
                alltra_ab.append(inputs_fw_1[0].transpose(1,0))
                train_x_1, _= self.getTrainXandY(inputs_fw_1)
                _, full_hidden, _, _, _ = TemperNet.forward(train_x_1, True)
                all_hidden.append(full_hidden)
            X = torch.cat(all_hidden, dim=0)
            all_abs = torch.cat(alltra_ab, dim=0)
            # Cluster analysis is used to determine the optimal number of clusters
            max_clusters = self.args.cluster_num
            best_n_clusters = determine_optimal_clusters(X, max_clusters)
            print(f"Best cluster number: {best_n_clusters}")
            self.args.cluster_num = best_n_clusters
            modifyArgsfile(self.args.config, 'cluster_num', best_n_clusters)
            print(f"self.args.cluster_num = {self.args.cluster_num}")

            X = X.cpu().detach().numpy()
            # Cluster data and labels
            kmeans = KMeans(n_clusters=self.args.cluster_num, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            with open(labels_filepath, "wb") as f1:
                pickle.dump(labels, f1, 4)
            with open(allabs_filepath, "wb") as f2:
                pickle.dump(all_abs, f2, 4)

        return labels,all_abs

    def playAllTrain(self,sequence_code):
        save_dir = self.args.model_filepath
        if os.path.exists(os.path.join(save_dir, "Performances.pkl")):
            with open(os.path.join(save_dir, "Performances.pkl"), "rb") as f:
                perf_dict = pickle.load(f)
            display_performance(perf_dict)
        else:
            perf_dict = {
                "Obs_Encoder[ADE,FDE]": [1e3, 1e3],
                "whole_model[ADE,FDE]": [1e3, 1e3]
            }
        save_dir_base = self.args.model_filepath
        Temperal_Encoder_filepath = os.path.join(save_dir_base, "Obs_Encoder[ADE,FDE].pth")

        if sequence_code==0:
            if os.path.exists(Temperal_Encoder_filepath):
                return
            else:
                Temperal_Encoder_net = EncoderTrainer(obs_len=self.args.obs_length, pre_len=self.args.pred_length,
                                                      hidden_size=self.args.hidden_size, num_layer=1,
                                                      args=self.args).cuda()
                self.playtrain_TemperEncoder(Temperal_Encoder_net, perf_dict, "Obs_Encoder[ADE,FDE]", nn.MSELoss())
        else:
            if os.path.exists(Temperal_Encoder_filepath):
                Temperal_Encoder_net = EncoderTrainer(obs_len=self.args.obs_length, pre_len=self.args.pred_length,
                                                      hidden_size=self.args.hidden_size, num_layer=1,args=self.args).cuda()
                Temperal_Encoder_state_dict = torch.load(Temperal_Encoder_filepath)
                Temperal_Encoder_net.load_state_dict(Temperal_Encoder_state_dict)
                all_labels, all_abs = self.getClasDataAndClusterNum(Temperal_Encoder_net)
                self.playEntireTrain(Temperal_Encoder_state_dict, perf_dict, "whole_model[ADE,FDE]", all_labels, all_abs)
            else:
                print("TemperEncoder has'nt been trained!")

    def val_epoch(self, all_labels, all_abs):
        self.net.eval()
        error_epoch,final_error_epoch, first_erro_epoch = 0,0,0
        error_epoch_list, final_error_epoch_list, first_erro_epoch_list= [], [], []
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = 1e-5,1e-5,1e-5
        start = time.time()
        for batch in range(self.dataloader_gt.testbatchnums):
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_test_batch(batch, None)  # batch_split:[batch_size, 2]
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split
            GATraj_loss, full_pre_tra = self.net.forward(inputs_fw , [], all_labels, all_abs, batch,iftest=True)

            for pre_tra in full_pre_tra:
                error, error_cnt, final_error, final_error_cnt, first_erro,first_erro_cnt = \
                L2forTest(pre_tra, batch_norm_gt[1:, :, :2],self.args.obs_length)
                error_epoch_list.append(error)
                final_error_epoch_list.append(final_error)
                first_erro_epoch_list.append(first_erro)

            first_erro_epoch = first_erro_epoch + min(first_erro_epoch_list)
            final_error_epoch = final_error_epoch + min(final_error_epoch_list)
            error_epoch = error_epoch + min(error_epoch_list)
            error_cnt_epoch = error_cnt_epoch + error_cnt
            final_error_cnt_epoch = final_error_cnt_epoch + final_error_cnt
            first_erro_cnt_epoch = first_erro_cnt_epoch + first_erro_cnt
            error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        end = time.time()
        return GATraj_loss, error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch,first_erro_epoch/ first_erro_cnt_epoch,end-start

    def test_epoch(self):
        labels_filepath = os.path.join(self.args.model_filepath, 'cluster_labels.pkl')
        allabs_filepath = os.path.join(self.args.model_filepath, 'allabs.pkl')
        if os.path.exists(labels_filepath) and os.path.exists(allabs_filepath):
            with open(labels_filepath, "rb") as f1:
                all_labels = pickle.load(f1)
            with open(allabs_filepath, "rb") as f2:
                all_abs = pickle.load(f2)

        self.net.eval()
        error_epoch, final_error_epoch, first_erro_epoch = 0, 0, 0
        error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = 1e-5, 1e-5, 1e-5
        start = time.time()
        for batch in range(self.dataloader_gt.testbatchnums):
            inputs_gt, batch_split, nei_lists = self.dataloader_gt.get_test_batch(batch,
                                                                                  0)  # batch_split:[batch_size, 2]
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, seq_list_gt, nei_num = inputs_gt
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split
            _, full_pre_tra = self.net.forward(inputs_fw, [], all_labels, all_abs, batch, iftest=True)

            for pre_tra in full_pre_tra:
                error, error_cnt, final_error, final_error_cnt, first_erro, first_erro_cnt = \
                    L2forTest(pre_tra, batch_norm_gt[1:, :, :2], self.args.obs_length)
                error_epoch_list.append(error)
                final_error_epoch_list.append(final_error)
                first_erro_epoch_list.append(first_erro)

            first_erro_epoch = first_erro_epoch + min(first_erro_epoch_list)
            final_error_epoch = final_error_epoch + min(final_error_epoch_list)
            error_epoch = error_epoch + min(error_epoch_list)
            error_cnt_epoch = error_cnt_epoch + error_cnt
            final_error_cnt_epoch = final_error_cnt_epoch + final_error_cnt
            first_erro_cnt_epoch = first_erro_cnt_epoch + first_erro_cnt
            error_epoch_list, final_error_epoch_list, first_erro_epoch_list = [], [], []
        end = time.time()
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch,first_erro_epoch/ first_erro_cnt_epoch

    def getTrainXandY(self, inputs):
        _, batch_norm_gt, _, _, _ = inputs
        train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length - 1, :,
                                                                :]  # [H, N, 2]
        train_x = train_x.permute(1, 2, 0)  # [N, 2, H]
        train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0)  # [N, 2, H]
        return train_x, train_y