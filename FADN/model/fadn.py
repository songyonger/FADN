from model import common
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def make_model(args, parent=False):
    return FADN(args)
    
class MaskPredictor(nn.Module):
    def __init__(self,in_channels):
        super(MaskPredictor,self).__init__()
        self.spatial_mask=nn.Conv2d(in_channels=in_channels,out_channels=3,kernel_size=1,bias=False)
    def forward(self,x):
        spa_mask=self.spatial_mask(x)
        spa_mask=F.gumbel_softmax(spa_mask,tau=1,hard=True,dim=1)
        return spa_mask

class DyResBlock(nn.Module):
    def __init__(self,kernel_size,in_chn,out_chn):
        super(DyResBlock,self).__init__()
        self.in_chn=in_chn
        self.shape_l_0=(in_chn,1,kernel_size,kernel_size)
        self.shape_mh_0=(out_chn,in_chn,kernel_size,kernel_size)
        self.shape_h_1=(out_chn,in_chn,kernel_size,kernel_size)
        self.MaskPredictor=MaskPredictor(self.in_chn)
        self.kernel_size=kernel_size
        self.unfold=nn.Unfold(kernel_size=kernel_size,dilation=1,padding=(kernel_size-1) // 2,stride=1)
        self.low_weights_0=nn.Parameter(torch.rand(self.shape_l_0) * 0.001, requires_grad=True)
        self.mid_weights_0=nn.Parameter(torch.rand(self.shape_mh_0) * 0.001,requires_grad=True)
        self.hig_weights_0=nn.Parameter(torch.rand(self.shape_mh_0)*0.001,requires_grad=True)
        self.hig_weights_1=nn.Parameter(torch.rand(self.shape_h_1)*0.001,requires_grad=True)
        self.relu=nn.ReLU(inplace=True)
        self.conv_1_low=nn.Conv2d(in_channels=in_chn,out_channels=in_chn,kernel_size=1,stride=1,bias=False,padding=0)
        self.conv_1_mid=nn.Conv2d(in_channels=in_chn,out_channels=in_chn,kernel_size=1,stride=1,bias=False,padding=0)

    def forward(self,x):
        n,c,h,w=x.size()
        low_weight_0=self.low_weights_0.view(c,self.kernel_size,self.kernel_size)
        mid_weight_0=self.mid_weights_0.view(c,-1)
        hig_weight_0=self.hig_weights_0.view(c,-1)
        hig_weight_1=self.hig_weights_1.view(c,-1)
        MaskPredictor=self.MaskPredictor(x)
        unfold=self.unfold(x).view(n,c*self.kernel_size*self.kernel_size,h,w)

        low_fre_num=[]
        mid_fre_num=[]
        hig_fre_num=[]
        sparsity=[]
        
        for i in range(n):
            low_fre_num.append(len(torch.nonzero(MaskPredictor[i,0,...])))
            mid_fre_num.append(len(torch.nonzero(MaskPredictor[i,1,...])))
            hig_fre_num.append(len(torch.nonzero(MaskPredictor[i,2,...])))
            sparsity.append((0.0633*low_fre_num[i]+0.5555*mid_fre_num[i]+hig_fre_num[i]) / (h*w))
            
        low_fre_mask=(MaskPredictor[:,0,...]).unsqueeze(1)
        mid_fre_mask=(MaskPredictor[:,1,...]).unsqueeze(1)
        hig_fre_mask=(MaskPredictor[:,2,...]).unsqueeze(1)
        
        unfold_low=unfold * (low_fre_mask.expand_as(unfold))
        unfold_low = unfold_low.view(n,c,self.kernel_size*self.kernel_size,h*w)
        unfold_low = unfold_low.permute(0,1,3,2).view(n,c,h,w,self.kernel_size,self.kernel_size)
        low=torch.einsum('nchwkj,ckj->nchw',unfold_low,low_weight_0)
        low=self.conv_1_low(self.relu(low))

        unfold_mid=unfold * (mid_fre_mask.expand_as(unfold))
        unfold_mid=unfold_mid.view(n,c*self.kernel_size*self.kernel_size,h*w)
        mid_0=unfold_mid.transpose(1,2).matmul(mid_weight_0.t()).transpose(1,2)
        mid_0=F.fold(mid_0,(h,w),(1,1))
        mid=self.conv_1_mid(self.relu(mid_0))
        
        unfold_hig=unfold * (hig_fre_mask.expand_as(unfold))
        unfold_hig=unfold_hig.view(n,c*self.kernel_size*self.kernel_size,h*w)
        hig_0=unfold_hig.transpose(1,2).matmul(hig_weight_0.t()).transpose(1,2)
        hig_0=F.fold(hig_0,(h,w),(1,1))
        hig_0=self.relu(hig_0)
        unfold=self.unfold(hig_0).view(n,c*self.kernel_size*self.kernel_size,h,w)
        unfold_hig=unfold * (hig_fre_mask.expand_as(unfold))
        unfold_hig=unfold_hig.view(n,c*self.kernel_size*self.kernel_size,h*w)
        hig=unfold_hig.transpose(1,2).matmul(hig_weight_1.t()).transpose(1,2)
        hig=F.fold(hig,(h,w),(1,1))
        
        return (x+low+mid+hig), sparsity

class FADN(nn.Module):
    def __init__(self,args,conv=common.default_conv):
        super(FADN,self).__init__()
        self.n_resblocks=args.n_resblocks
        n_feats=args.n_feats
        kernel_size=3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [DyResBlock(kernel_size,n_feats,n_feats) for _ in range(self.n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        n,c,h,w=x.size()
        sparsity_sum=[0]*n
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for i in range(self.n_resblocks):
            res,sparsity=self.body[i](res)
            sparsity_sum = [sparsity_sum[i]+sparsity[i] for i in range(min(len(sparsity),len(sparsity_sum)))]
        sparsity_avg = [(k / (self.n_resblocks)) for k in sparsity_sum]
        res=self.body[self.n_resblocks](res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        
        return x, sparsity_avg

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
