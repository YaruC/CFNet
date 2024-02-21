# coding:utf-8

import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import math
import torch.utils.model_zoo as model_zoo
from thop import profile
import pandas
import numpy as np
import os
import matplotlib.pyplot as plt
from timm.models.layers import trunc_normal_

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)
        

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        return x1, x2
class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction)
        self.channel_proj2 = nn.Linear(dim, dim // reduction)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj = nn.Linear(dim, dim)
        self.norm = norm_layer(dim)

    def forward(self, x1, x2):
        u1 = self.act1(self.channel_proj1(x1))
        u2 = self.act2(self.channel_proj2(x2))
        v1, v2 = self.cross_attn(u1, u2)
        y1 = x1 + v1
        y2 = x2 + v2
        out_x1 = self.norm(self.end_proj(y1))
        out_x2 = self.norm(self.end_proj(y2))
        return out_x1, out_x2

class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            
            nn.ReLU(inplace=True),
            
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
       
        residual = self.residual(x)
        
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

class FusinModel(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)

        return merge

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class Fusion_layer(nn.Module):
    def __init__(self, dim):
        super(Fusion_layer, self).__init__()
        self.dim = dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()
        self.BN = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, x3):
        x1_0 = x1 * x2
        x1_0 = torch.cat((x1_0, x1), dim=1)
        x1_0 = self.conv1(x1_0)
        x1_0 = self.BN(x1_0)
        x1_0 = self.relu(x1_0)
        x1_0 = self.sigmoid(x1_0)

        x2_0 = x1 * x3
        x2_0 = torch.cat((x2_0, x1), dim=1)
        x2_0 = self.conv2(x2_0)
        x2_0 = self.BN(x2_0)
        x2_0 = self.relu(x2_0)
        x2_0 = self.sigmoid(x2_0)

        x_2 = x1_0 * x2
        x_3 = x2_0 * x3

        x = x_2 + x_3 + x1
        x = self.conv4(x)

        return x
        
class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, int(in_channel // 4), kernel_size=1, stride=1, padding=0)
        self.context1 = nn.Sequential(
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
        )
        self.context2 = nn.Sequential(
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
        )
        self.context3 = nn.Sequential(
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
        )
        self.context4 = nn.Sequential(
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(int(in_channel // 4), depth, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(int(in_channel // 4), int(in_channel // 4))
        self.fc2 = nn.Linear(int(in_channel // 4), int(in_channel // 4))
        self.fc3 = nn.Linear(int(in_channel // 4), int(in_channel // 4))
        self.fc4 = nn.Linear(int(in_channel // 4), int(in_channel // 4))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

        self.ln = nn.LayerNorm(int(in_channel // 4))
        self.relu = nn.ReLU()

    def forward(self, x):
        img = self.conv1(x)
        img_1 = self.context1(img)
        img_2 = self.context2(img)
        img_3 = self.context3(img)
        img_4 = self.context4(img)
        img = img_1 + img_2 + img_3 + img_4
        global_pool_out = self.global_pool(img)
        fc1_out = self.fc1(global_pool_out.squeeze(dim=2).squeeze(dim=2))
        fc2_out = self.fc2(global_pool_out.squeeze(dim=2).squeeze(dim=2))
        fc3_out = self.fc3(global_pool_out.squeeze(dim=2).squeeze(dim=2))
        fc4_out = self.fc4(global_pool_out.squeeze(dim=2).squeeze(dim=2))
        fc_concat = torch.cat((fc1_out, fc2_out, fc3_out, fc4_out), dim=1)
        softmax_out = self.softmax(fc_concat)
        final_out = img_1 * softmax_out[:, 0:256].unsqueeze(2).unsqueeze(3) + \
                    img_2 * softmax_out[:, 256:512].unsqueeze(2).unsqueeze(3) + \
                    img_3 * softmax_out[:, 512:768].unsqueeze(2).unsqueeze(3) + \
                    img_4 * softmax_out[:, 768:1024].unsqueeze(2).unsqueeze(3)
        relu_out = self.relu(final_out)

        part1 = self.conv2(relu_out)

        out = self.dropout(x + part1)

        return out

        
class MultiModalProcess(nn.Module):
    def __init__(self, shallow_channels, deep_channels, out_channels):
        super(MultiModalProcess, self).__init__()

        self.shallow_fusion_conv = nn.Sequential(
            nn.Conv2d(shallow_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.deep_fusion_conv = nn.Sequential(
            nn.Conv2d(deep_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.concat_relu = nn.ReLU()

        self.attention_conv_bn_relu = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.final_conv_bn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.sigmoid = nn.Sigmoid()
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, shallow_fusion, deep_fusion):
        shallow_fusion = self.shallow_fusion_conv(shallow_fusion)
        deep_fusion = self.deep_fusion_conv(deep_fusion)

        deep_fusion = nn.functional.interpolate(deep_fusion, size=shallow_fusion.size()[2:], mode='bilinear',
                                                align_corners=False)

        concatenated = torch.cat((shallow_fusion, deep_fusion), dim=1)
        concatenated = self.concat_relu(concatenated)

        attention_output = self.attention_conv_bn_relu(concatenated)
        attention_weights = self.sigmoid(attention_output)

        weighted_shallow_fusion = shallow_fusion * attention_weights

        weighted_shallow_fusion = self.fusion_conv(weighted_shallow_fusion)

        deep_fusion = deep_fusion + weighted_shallow_fusion

        return deep_fusion

class CFNet(nn.Module):
    def __init__(self, n_class):
        super(CFNet, self).__init__()
        resnet_raw_model1 = models.resnet50(pretrained=True)
        resnet_raw_model2 = models.resnet50(pretrained=True)

        filters = [64, 256, 512, 1024]
        num_heads = [1, 2, 4, 8]
        norm_fuse = nn.BatchNorm2d

        self.encoder_two_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_two_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)
        self.encoder_two_bn1 = resnet_raw_model1.bn1
        self.encoder_two_relu = resnet_raw_model1.relu

        self.encoder_two_maxpool = resnet_raw_model1.maxpool
        self.encoder_two_layer1 = resnet_raw_model1.layer1
        self.encoder_two_layer2 = resnet_raw_model1.layer2
        self.encoder_two_layer3 = resnet_raw_model1.layer3
        self.encoder_two_layer4 = resnet_raw_model1.layer4


        self.encoder_one_conv1 = resnet_raw_model2.conv1
        self.encoder_one_bn1 = resnet_raw_model2.bn1
        self.encoder_one_relu = resnet_raw_model2.relu

        self.encoder_one_maxpool = resnet_raw_model2.maxpool
        self.encoder_one_layer1 = resnet_raw_model2.layer1
        self.encoder_one_layer2 = resnet_raw_model2.layer2
        self.encoder_one_layer3 = resnet_raw_model2.layer3
        self.encoder_one_layer4 = resnet_raw_model2.layer4

        self.conv0 = nn.Conv2d(16, 64, 1, 1)
        self.conv1 = nn.Conv2d(64, 256, 1, 1)
        self.conv2 = nn.Conv2d(128, 512, 1, 1)
        self.conv3 = nn.Conv2d(256, 1024, 1, 1)
        self.conv4 = nn.Conv2d(512, 2048, 1, 1)

        self.encoder_relu = nn.ReLU(inplace=True)
        self.encoder_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder_downsampling1 = nn.Conv2d(64, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_downsampling2 = nn.Conv2d(256, 512, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_downsampling3 = nn.Conv2d(512, 1024, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_downsampling4 = nn.Conv2d(1024, 2048, kernel_size=7, stride=2, padding=3, bias=False)

        channels = [64, 256, 512, 1024, 2048]
        self.local_att0 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[0], int(channels[0] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[0] // 4)),
            nn.ReLU(inplace=True),
        )
        self.global_att0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[0], int(channels[0] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[0] // 4)),
            nn.ReLU(inplace=True),
        )

        self.local_att1 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[1], int(channels[1] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[1] // 4)),
            nn.ReLU(inplace=True),
        )

        self.global_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[1], int(channels[1] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[1] // 4)),
            nn.ReLU(inplace=True),
        )

        self.local_att2 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[2], int(channels[2] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[2] // 4)),
            nn.ReLU(inplace=True),
        )

        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[2], int(channels[2] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[2] // 4)),
            nn.ReLU(inplace=True),
        )

        self.local_att3 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[3], int(channels[3] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[3] // 4)),
            nn.ReLU(inplace=True),
        )

        self.global_att3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[3], int(channels[3] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[3] // 4)),
            nn.ReLU(inplace=True),
        )

        self.local_att4 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[4], int(channels[4] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[4] // 4)),
            nn.ReLU(inplace=True),
        )

        self.global_att4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[4], int(channels[4] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[4] // 4)),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.aspp2 = ASPP(filters[2], filters[2])
        self.aspp3 = ASPP(filters[3], filters[3])

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, n_class, 2, padding=1)
        self.conv_2_3 = nn.Conv2d(2, 3, kernel_size=1)

        self.fusion_layer0 = Fusion_layer(1024)
        self.fusion_layer1 = Fusion_layer(512)
        self.fusion_layer2 = Fusion_layer(256)
        self.fusion_layer3 = Fusion_layer(64)
        self.modules = nn.ModuleList([
            FusinModel(dim=channels[0], reduction=1, num_heads=num_heads[0], norm_layer=norm_fuse),
            FusinModel(dim=channels[1], reduction=1, num_heads=num_heads[1], norm_layer=norm_fuse),
            FusinModel(dim=channels[2], reduction=1, num_heads=num_heads[2], norm_layer=norm_fuse),
            FusinModel(dim=channels[3], reduction=1, num_heads=num_heads[3], norm_layer=norm_fuse)])
        self.multi_modal_process1 = MultiModalProcess(filters[2],filters[3] ,filters[2])
        self.multi_modal_process2 = MultiModalProcess(filters[1],filters[2] , filters[1])
        self.multi_modal_process3 = MultiModalProcess(filters[0],filters[1] , filters[0])

    def forward(self, input):

        imgT1 = input[0]
        imgT2 = input[1]
        imgT1ce = input[2]
        imgflair = input[3]
        one = torch.cat([imgT1, imgT2], dim=1)
        two = torch.cat([imgT1ce, imgflair], dim=1)

        one = self.conv_2_3(one)
        two = self.conv_2_3(two)

        one = self.encoder_one_conv1(one)
        one = self.encoder_one_bn1(one)
        one = self.encoder_one_relu(one)

        two = self.encoder_one_conv1(two)
        two = self.encoder_one_bn1(two)
        two = self.encoder_one_relu(two)
        e02 = self.modules[0](one, two)

        one = self.encoder_one_maxpool(one)
        two = self.encoder_two_maxpool(two)
        one = self.encoder_one_layer1(one)
        two = self.encoder_two_layer1(two)
        e12 = self.modules[1](one, two)

        one = self.encoder_one_layer2(one)
        two = self.encoder_two_layer2(two)
        e22 = self.modules[2](one, two)

        one = self.encoder_one_layer3(one)
        two = self.encoder_two_layer3(two)
        e32 = self.modules[3](one, two)
        e33 = self.aspp3(e32)

        # decoder
       
        x3 = self.multi_modal_process1(e22, e33)
        d3 = self.decoder3(e33) + x3
        
        x2 = self.multi_modal_process2(e12, d3)
        d2 = self.decoder2(d3) + x2
       
        x1 = self.multi_modal_process3( e02, d2)
        d1 = self.decoder1(d2) + x1
        fuse = self.finaldeconv1(d1)
        fuse = self.finalrelu1(fuse)
        fuse = self.finalconv2(fuse)
        fuse = self.finalrelu2(fuse)
        fuse = self.finalconv3(fuse)
        return fuse


