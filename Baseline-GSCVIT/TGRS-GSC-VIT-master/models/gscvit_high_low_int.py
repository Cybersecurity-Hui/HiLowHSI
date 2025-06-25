import random
from functools import partial
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from thop import profile
from timm.models.vision_transformer import _cfg
from torchsummaryX import summary
from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat, get_default_qat_qconfig, HistogramObserver, PerChannelMinMaxObserver
import torch.quantization as tq
from torch.quantization import QConfig, default_observer, default_per_channel_weight_observer
import torch.nn.quantized as nnq

def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# 通道校准策略1
class ChannelAdjustmentLayer1(nn.Module):
    def __init__(self, target_channels=256):
        super(ChannelAdjustmentLayer1, self).__init__()
        self.target_channels = target_channels

    def forward(self, x):
        B, C, H, W = x.size()

        if C == self.target_channels:
            return x

        if C < self.target_channels:
            # 逐个通道复制，放到被复制通道的后面
            num_channels_to_copy = self.target_channels - C
            for i in range(num_channels_to_copy):
                channel_to_copy = torch.randint(0, C, (1,))
                x = torch.cat([x, x[:, channel_to_copy, :, :]], dim=1)

        else:
            # 逐个通道删除
            num_channels_to_remove = C - self.target_channels
            for i in range(num_channels_to_remove):
                rand=torch.randint(0,x.shape[1],(1,))
                x = torch.cat([x[:, :rand, :, :], x[:, rand+1:, :, :]], dim=1)
        return x


# 通道校准策略2
class ChannelAdjustmentLayer2(nn.Module):
    def __init__(self, target_channels=256):
        super(ChannelAdjustmentLayer2, self).__init__()
        self.target_channels = target_channels

    def forward(self, x):
        B, C, H, W = x.size()

        if C == self.target_channels:
            return x

        if C < self.target_channels:
            # 计算需要扩展的通道数
            num_channels_to_expand = self.target_channels - C
            # 计算每一端需要扩展的通道数
            channels_to_expand_per_side = num_channels_to_expand // 2

            # 两端均匀镜像扩展
            x = torch.cat([x[:, :channels_to_expand_per_side+1, :, :].flip(dims=(1,)),
                           x,
                           x[:, -channels_to_expand_per_side:, :, :].flip(dims=(1,))], dim=1)

        else:
            # 计算需要删除的通道数
            num_channels_to_remove = C - self.target_channels
            # 计算每一端需要删除的通道数
            channels_to_remove_per_side = num_channels_to_remove // 2

            # 两端均匀镜像删除
            x = x[:, channels_to_remove_per_side:-channels_to_remove_per_side, :, :]

        return x[:, :self.target_channels, :, :]


# 通道正则化
class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = ChanLayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


# 通道校准模块
class SpectralCalibration(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GSC(nn.Module):
    def __init__(self, dim_in, dim_out, padding=1, num_groups=8):
        super().__init__()
        self.gpwc = nn.Conv2d(dim_in, dim_out, groups=num_groups, kernel_size=1)
        self.gc = nn.Conv2d(dim_out, dim_out, kernel_size=3, groups=num_groups, padding=padding, stride=1)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.gc(self.gpwc(x))))


class GSSA(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.,
            group_spatial_size=3
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.group_spatial_size = group_spatial_size
        inner_dim = dim_head * heads

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)

        self.group_tokens = nn.Parameter(torch.randn(dim))

        self.group_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(inner_dim, inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h=heads),
        )

        self.group_attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        batch, height, width, heads, gss = x.shape[0], *x.shape[-2:], self.heads, self.group_spatial_size
        assert (height % gss) == 0 and (
                width % gss) == 0, f'height {height} and width {width} must be divisible by group spatial size {gss}'
        num_groups = (height // gss) * (width // gss)

        x = rearrange(x, 'b c (h g1) (w g2) -> (b h w) c (g1 g2)', g1=gss, g2=gss)

        w = repeat(self.group_tokens, 'c -> b c 1', b=x.shape[0])

        x = torch.cat((w, x), dim=-1)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))

        q = q * self.scale

        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        group_tokens, grouped_fmaps = out[:, :, 0], out[:, :, 1:]

        if num_groups == 1:
            fmap = rearrange(grouped_fmaps, '(b x y) h (g1 g2) d -> b (h d) (x g1) (y g2)', x=height // gss,
                             y=width // gss, g=gss, g2=gss)
            return self.to_out(fmap)

        group_tokens = rearrange(group_tokens, '(b x y) h d -> b h (x y) d', x=height // gss, y=width // gss)

        grouped_fmaps = rearrange(grouped_fmaps, '(b x y) h n d -> b h (x y) n d', x=height // gss, y=width // gss)

        w_q, w_k = self.group_tokens_to_qk(group_tokens).chunk(2, dim=-1)

        w_q = w_q * self.scale

        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)

        w_attn = self.group_attend(w_dots)

        aggregated_grouped_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, grouped_fmaps)

        fmap = rearrange(aggregated_grouped_fmap, 'b h (x y) (g1 g2) d -> b (h d) (x g1) (y g2)', x=height // gss,
                         y=width // gss, g1=gss, g2=gss)
        return self.to_out(fmap)


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            dim_head=16,
            heads=8,
            dropout=0.,
            norm_output=True,
            groupsize=4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            self.layers.append(
                PreNorm(dim, GSSA(dim, group_spatial_size=groupsize, heads=heads, dim_head=dim_head, dropout=dropout))
            )

        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)

        return self.norm(x)
        
def create_better_qconfig():
    #创建更优的量化配置"""
    activation_observer = torch.ao.quantization.MovingAverageMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,  # 使用对称量化
        reduce_range=False,
        averaging_constant=0.011  # 添加移动平均
    )
    
    weight_observer = torch.ao.quantization.MinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,  # 使用对称量化
        reduce_range=False
    )
    
    return torch.ao.quantization.QConfig(
        activation=activation_observer,
        weight=weight_observer
    )

class Low_Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, index=True):
        super().__init__()
        self.index = index
        self.quant = QuantStub()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        #x_or = x
        x = self.quant(x)
        x = self.conv(x)
        #x = self.conv(x)
        x = self.dequant(x)
        #return x + x_or
        return x
    
    def set_qconfig(self):
        # 获取高精度配置
        high_precision_qconfig = create_better_qconfig()
        
        # 为不同层设置不同的量化配置
        self.conv.qconfig = high_precision_qconfig
        
        # 为量化和反量化层设置配置
        self.quant.qconfig = high_precision_qconfig
        self.dequant.qconfig = high_precision_qconfig
        
class SpectralDecomp(nn.Module):
    #支持多通道高光谱输入的频域分解"""
    def __init__(self, spatial_radius=7, texture_contrast=0.8):
        super().__init__()
        self.spatial_radius = spatial_radius
        self.texture_contrast = texture_contrast
        
        # 多通道高斯模糊
        self.gaussian_blur = nn.Sequential(
            nn.ReflectionPad2d(spatial_radius),
            nn.Conv2d(1, 1, kernel_size=2*spatial_radius+1, bias=False)
        )
        self._init_gaussian_weights()
        
    def _init_gaussian_weights(self):
        #初始化适用于任意通道数的高斯核"""
        sigma = self.spatial_radius/3
        x = torch.arange(-self.spatial_radius, self.spatial_radius+1)
        kernel = torch.exp(-x.pow(2)/(2*sigma**2))
        kernel = kernel.view(1,1,-1) * kernel.view(1,1,-1,1)
        self.gaussian_blur[1].weight.data = kernel / kernel.sum()
        
    def forward(self, x):
        #
        #输入: (B, C, H, W) 张量
        ##输出: (low_freq, high_freq) 元组
        #"""
        B, C, H, W = x.shape
        
        # 多通道处理
        x_flat = x.reshape(B*C, 1, H, W)  # 展平为(B*C, 1, H, W)
        
        # 低频分量
        low = self.gaussian_blur(x_flat)
        low = low.view(B, C, H, W)  # 恢复原始形状
        
        # 高频分量
        high = (x - low) * self.texture_contrast
        
        return low, high
        
class GSCViT(nn.Module):
    def __init__(
            self,
            *,
            num_classes,
            depth,
            heads, 
            group_spatial_size,
            channels=200,
            dropout=0.1,
            padding,
            dims=(256, 128, 64, 32),
            num_groups=[16,16,16]
    ):
        super().__init__()
        num_stages = len(depth)-1
        
        # 初始化参数检查
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        hyperparams_per_stage = [heads]
        hyperparams_per_stage = list(map(partial(cast_tuple, length=num_stages), hyperparams_per_stage))
        
        
        # 初始特征提取
        self.sc = SpectralCalibration(channels, 256)
        self.bn_1 = nn.BatchNorm2d(256)
        self.relu_1 = nn.ReLU(inplace=True)
        
        # 频率分解模块
        self.decomp = SpectralDecomp(spatial_radius=7)
        self.low_channel = 128
        self.high_channel = 16
        self.totol_channel = self.low_channel + self.high_channel
        #self.totol_channel = self.low_channel
        self.quant_first = QuantStub()
        self.low_branch_first_gsc = nn.Conv2d(256,128,kernel_size=1,padding=0)
        self.dequant_first = DeQuantStub()
        self.quant_second = QuantStub()
        self.low_branch_first_conv = nn.Conv2d(128,128,kernel_size=1,padding=0)
        self.dequant_second = DeQuantStub()
        self.low_branch_first_bn = nn.BatchNorm2d(128)
        self.low_branch_first_relu = nn.ReLU(inplace=True)

        self.low_branch_second_gsc = nn.Conv2d(144,64,kernel_size=1,padding=0)
    
        self.low_branch_second_transformer = Transformer(dim=64,depth=1,heads=1,groupsize=4,dropout=dropout,norm_output=True)
    
        self.quant_third = QuantStub()
        self.low_branch_second_conv = nn.Conv2d(64,64,kernel_size=1,padding=0)
        self.dequant_third = DeQuantStub()
    
        self.low_branch_second_bn = nn.BatchNorm2d(64)
        self.low_branch_second_relu = nn.ReLU(inplace=True)    
            
        # 高频分支
        self.high_branch_first = nn.Sequential(
            GSC(256, 16, 1, 4),
            Transformer(
                dim=16,
                depth=1,
                heads=1,
                groupsize=4,
                dropout=dropout,
                norm_output=True
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.high_branch_second = nn.Sequential(
            GSC(144, 16, 1, 4),
            Transformer(
                dim=16,
                depth=1,
                heads=1,
                groupsize=4,
                dropout=dropout,
                norm_output=True
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
       
        # #融合层
        self.fusion_first = nn.Sequential(
            nn.Conv2d(144, 144, kernel_size=1,padding=0),
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True)
        )

        # self.fusion_second = nn.Sequential(
        #     nn.Conv2d(80, 80, kernel_size=1,padding=0),
        #     nn.BatchNorm2d(80),
        #     nn.ReLU(inplace=True)
        # )
        
        # 分类头
        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(80),
            nn.Linear(80, num_classes)
        )
    
    def forward(self, x):
        # 初始特征提取
        x = x.squeeze(dim=1)
        x = self.sc(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
            
        # 频率分解
        low, high = self.decomp(x)
            
        # 分支处理
        low = self.quant_first(low)  # 进入量化域
        low = self.low_branch_first_gsc(low)  # 量化卷积
        low = self.dequant_first(low)  # 退出量化域
        
        temp = low#FP32
        low = self.quant_second(low)
        low = self.low_branch_first_conv(low)
        low = self.low_branch_first_conv(low)
        low = self.dequant_second(low) + temp
        low = self.low_branch_first_bn(low)
        low = self.low_branch_first_relu(low)
        
        high = self.high_branch_first(high)#C=16
        # 特征融合
        x = torch.cat((low, high), dim=1)#C=144
        x = self.fusion_first(x)
        # 频率分解
        low, high = self.decomp(x)
            
        # 分支处理
        low = self.low_branch_second_gsc(low)#C=64
        temp = low
        low = self.low_branch_second_transformer(low)
        low = self.low_branch_second_conv(low) + temp
        #low = self.dequant_third(self.low_branch_second_conv(self.quant_third(low)))+temp
        low = self.low_branch_second_bn(low)
        low = self.low_branch_second_relu(low)
        
        high = self.high_branch_second(high)#C=16
 
        # 特征融合
        x = torch.cat((low, high), dim=1)#C=80
        # x = self.fusion_second(x)
        # 分类
        return self.mlp_head(x)
        
    def set_qconfig(self):
        # 获取高精度配置
        high_precision_qconfig = create_better_qconfig()
        
        # 为不同层设置不同的量化配置
        self.quant_first.qconfig = high_precision_qconfig
        self.low_branch_first_gsc.qconfig = high_precision_qconfig
        self.dequant_first.qconfig = high_precision_qconfig

        self.quant_second.qconfig = high_precision_qconfig
        self.low_branch_first_conv.qconfig = high_precision_qconfig
        self.dequant_second.qconfig = high_precision_qconfig

        # self.quant_third.qconfig = high_precision_qconfig
        # self.low_branch_second_conv.qconfig = high_precision_qconfig
        # self.dequant_third.qconfig = high_precision_qconfig

        #self.low_branch_second_gsc.set_qconfig()

def gscvit_HL_INT(dataset):
    model = None
    if dataset == 'sa':
        model = GSCViT(
            num_classes=16,
            channels=204,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'UP':
        model = GSCViT(
            num_classes=9,
            channels=103,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 32, 16),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'HongHu':
        model = GSCViT(
            num_classes=22,
            channels=270,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 32, 16),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'HanChuan':
        model = GSCViT(
            num_classes=16,
            channels=274,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 32, 16),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'pu':
        model = GSCViT(
            num_classes=9,
            channels=103,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'whulk':
        model = GSCViT(
            num_classes=9,
            channels=270,
            heads=(4, 4, 4),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'hrl':
        model = GSCViT(
            num_classes=14,
            channels=176,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'flt':
        model = GSCViT(
            num_classes=10,
            channels=80,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'ksc':
        model = GSCViT(
            num_classes=13,
            channels=176,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'ip':
        model = GSCViT(
            num_classes=16,
            channels=200,
            heads=(16, 16, 16),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4,4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[8, 8, 8]
        )
    elif dataset == 'Houston':
        model = GSCViT(
            num_classes=15,
            channels=144,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 32, 16),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'MUUFL':
        model = GSCViT(
            num_classes=11,
            channels=64,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'Trento':
        model = GSCViT(
            num_classes=6,
            channels=63,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[16, 16, 16]
        )
    elif dataset == 'botswana':
        model = GSCViT(
            num_classes=14,
            channels=145,
            heads=(1, 1, 1),
            depth=(1, 1, 1),
            group_spatial_size=[4, 4, 4],
            dropout=0.1,
            padding=[1, 1, 1],
            dims = (256, 128, 64),
            num_groups=[8, 8, 8]
        )
    return model



if __name__ == '__main__':
    img = torch.randn(1, 270, 8, 8)
    print("input shape:", img.shape)
    net = gscvit(dataset='whulk')
    net.default_cfg = _cfg()
    print("output shape:", net(img).shape)
    summary(net, torch.zeros((1, 1, 270, 8, 8)))
    flops, params = profile(net, inputs=(img,))
    print('params', params)
    print('flops', flops)  ## 打印计算量

