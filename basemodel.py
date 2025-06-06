import torch
import torch.nn as nn
import torch.nn.functional as F
from laplace_decoder import MLPDecoder,GRUDecoder

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            # print("LSTM------",m.named_parameters())
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)  # initializing the lstm bias with zeros
        else:
            print(m, "************")


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        device = x.device
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight.to(device) * x + self.bias.to(device)



class MLP_gate(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP_gate, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.W = nn.Linear(feature_dim, 1)

    def forward(self, x):
        scores = self.W(x)
        alpha = F.softmax(scores, dim=0)
        return (x * alpha).sum(dim=0)


#Micro-model
class EncoderTrainer(nn.Module):
    def __init__(self, args,obs_len=8, pre_len=12, hidden_size=64, num_layer=3):
        super(EncoderTrainer, self).__init__()
        self.encoder = Temperal_Encoder(args)
        self.decoder = TemperalGRUDecoder(args)

    def forward(self, train_x_0, isTemperTest):
        output = None
        full_hidden = None
        trj_hidden_0, hidden, cn= self.encoder(train_x_0)
       #计算源域的初步预测轨迹
        outs = trj_hidden_0
        if isTemperTest:
            _,traj_hidden_future =  self.decoder(outs, isTemperTest)
            full_hidden = torch.cat((trj_hidden_0,traj_hidden_future), dim=-1)
        else:
            output, _= self.decoder(outs, isTemperTest)
        return output, full_hidden, trj_hidden_0, hidden, cn

class TemperalGRUDecoder(nn.Module):
    def __init__(self, args, num_modes=1):
        super(TemperalGRUDecoder, self).__init__()
        min_scale: float = 1e-3
        self.args = args
        self.hidden_size = self.args.hidden_size
        self.future_steps = self.args.pred_length
        self.min_scale = min_scale
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                           hidden_size=self.hidden_size,
                           num_layers=1,
                           batch_first=True)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.mlp = MLP(self.hidden_size)

    def forward(self, hidden_state,isTemperTest):
        hidden_state = hidden_state.expand(self.future_steps, *hidden_state.shape)
        hidden_state = hidden_state.transpose(0, 1)
        out, (hn, cn) = self.lstm(hidden_state)  # [H, N, D]
        traj_hidden_future=None
        if isTemperTest:
            state, cn = hn.squeeze(0),cn.squeeze(0)
            xstateMlp = torch.clone(state)
            traj_hidden_future = self.mlp(xstateMlp) + state
        loc = self.loc(out) # [N, H, 2]
        scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [N, H, 2]
        return (loc, scale), traj_hidden_future

class Temperal_Encoder(nn.Module):
    """Construct the sequence model"""

    def __init__(self,args):
        super(Temperal_Encoder, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size

        self.conv1d=nn.Conv1d(2, self.hidden_size, kernel_size=3, stride=1, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead= self.args.x_encoder_head,\
                         dim_feedforward=self.hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.args.x_encoder_layers)
        self.mlp1 = MLP(self.hidden_size)
        self.mlp = MLP(self.hidden_size)
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=False)
        initialize_weights(self.conv1d.modules())

    def forward(self, x):
        self.x_dense=self.conv1d(x).permute(0,2,1) #[N, H, dim]
        self.x_dense=self.mlp1(self.x_dense) + self.x_dense #[N, H, dim]
        self.x_dense_in = self.transformer_encoder(self.x_dense) + self.x_dense  #[N, H, D]
        output, (hn, cn) = self.lstm(self.x_dense_in)
        self.x_state, cn = hn.squeeze(0), cn.squeeze(0)  # [N, D]
        xstateMlp = torch.clone(self.x_state)
        self.x_encoded = self.mlp(xstateMlp) + self.x_state#[N, D]
        return self.x_encoded, self.x_state, cn


class Global_interaction(nn.Module):
    def __init__(self,args):
        super(Global_interaction, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        # Motion gate
        self.ngate = MLP_gate(self.hidden_size*3, self.hidden_size)  #sigmoid
        # Relative spatial embedding layer
        self.relativeLayer = MLP(2, self.hidden_size)
        # Attention
        self.WAr = MLP(self.hidden_size*3, 1)
        self.weight = MLP(self.hidden_size)

    def forward(self, corr_index, nei_index, hidden_state, cn):
        '''
        States Refinement process
        Params:
            corr_index: relative coords of each pedestrian pair [N, N, D]
            nei_index: neighbor exsists flag [N, N]
            nei_num: neighbor number [N]
            hidden_state: output states of GRU [N, D]
        Return:
            Refined states
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self_h = hidden_state
        self.N = corr_index.shape[0]
        self.D = self.hidden_size
        nei_inputs = self_h.repeat(self.N, 1) #[N, N, D]
        nei_index_t = nei_index.view(self.N*self.N) #[N*N]
        corr_t=corr_index.contiguous().view((self.N * self.N, -1)) #[N*N, D]
        if corr_t[nei_index_t > 0].shape[0] == 0:
            # Ignore when no neighbor in this batch
            return hidden_state, cn
        r_t = self.relativeLayer(corr_t[nei_index_t > 0]) #[N*N, D]
        inputs_part = nei_inputs[nei_index_t > 0].float()
        hi_t = nei_inputs.view((self.N, self.N, self.hidden_size)).permute(1, 0, 2).contiguous().view(-1, self.hidden_size) #[N*N, D]
        tmp = torch.cat((r_t, hi_t[nei_index_t > 0],nei_inputs[nei_index_t > 0]), 1) #[N*N, 3*D]
        # Motion Gate
        nGate = self.ngate(tmp).float() #[N*N, D]
        # Attention
        Pos_t = torch.full((self.N * self.N,1), 0, device=device).view(-1).float()
        tt = self.WAr(torch.cat((r_t, hi_t[nei_index_t > 0], nei_inputs[nei_index_t > 0]), 1)).view(-1).float() #[N*N, 1]
        #have bug if there's any zero value in tt
        Pos_t[nei_index_t > 0] = tt
        Pos = Pos_t.view((self.N, self.N))
        Pos[Pos == 0] = -10000
        Pos = torch.softmax(Pos, dim=1)
        Pos_t = Pos.view(-1)
        # Message Passing
        H = torch.full((self.N * self.N, self.D), 0, device=device).float()
        H[nei_index_t > 0] = inputs_part * nGate
        H[nei_index_t > 0] = H[nei_index_t > 0] * Pos_t[nei_index_t > 0].repeat(self.D, 1).transpose(0, 1)
        H = H.view(self.N, self.N, -1) #[N, N, D]
        H_sum = self.weight(torch.sum(H, 1))  # [N, D]
        # Update hidden states
        C = H_sum + cn  # [N, D]
        H = hidden_state + F.tanh(C)  # [N, D]
        return H, C



class Laplacian_Decoder(nn.Module):

    def __init__(self,args):
        super(Laplacian_Decoder, self).__init__()
        self.args = args
        if args.mlp_decoder:
            self._decoder = MLPDecoder(args)
        else:
            self._decoder = GRUDecoder(args)

    def forward(self,x_encode, hidden_state, cn):
        mdn_out = self._decoder(x_encode, hidden_state, cn)
        loc, scale, pi = mdn_out  # [F, N, H, 2], [F, N, H, 2], [N, F]
        return (loc, scale, pi)