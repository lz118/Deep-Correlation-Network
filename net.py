import torch
from torch import nn
import torch.nn.functional as F
# vgg16
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    stage = 1
    for v in cfg:
        if v == 'M':
            stage += 1
            if stage == 6:
                layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
            else:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            if stage == 6:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.extract = [3, 8, 15, 22, 29] # [ 8, 15, 22, 29]
        # 64:1 -->128:1/2 -->256:1/4 -->512 :1/8 --> 512:1/16 -->M-> 512,1/16
        self.base = nn.ModuleList(vgg(self.cfg, 3))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        tmp_x = []
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                tmp_x.append(x)     #collect feature maps 1(64)  1/2(128)  1/4(256)  1/8(512)  1/16(512)
        return tmp_x

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

def kernel2d_conv(feat_in, kernel, ksize):
    """
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out

class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class MySTN(nn.Module):
    def __init__(self, in_ch, mode='Curve'):
        super(MySTN, self).__init__()
        self.mode = mode
        self.down_block_1 = nn.Sequential(
            convblock(in_ch, 128, 3, 2, 1),
            convblock(128, 128, 1, 1, 0)
        )
        self.down_block_2 = nn.Sequential(
            convblock(128, 128, 3, 2, 1),
            convblock(128, 128, 1, 1, 0)
        )
        if mode =='Curve':
            self.up_blcok_1 = convblock(128, 128, 1, 1, 0)
            self.up_blcok_2 = convblock(128, 64, 1, 1, 0)
            self.wrap_filed = nn.Conv2d(64,2,3,1,1)
            self.wrap_filed.weight.data.normal_(mean=0.0, std=5e-4)
            self.wrap_filed.bias.data.zero_()
            self.wrap_grid = None
        elif mode =='Affine':
            self.down_block_3 = nn.Sequential(
                convblock(128, 128, 3, 2, 1),
                convblock(128, 128, 1, 1, 0),
            )
            self.deta = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(128,6,1,1,0)
            )
            # Start with identity transformation
            self.deta[-1].weight.data.normal_(mean=0.0, std=5e-4)
            self.deta[-1].bias.data.zero_()
            self.affine_matrix = None
            self.wrap_grid = None

    def forward(self, in_):
        size = in_.shape[2:]
        n1 = self.down_block_1(in_)
        n2 = self.down_block_2(n1)

        if self.mode=="Curve":
            n2 = self.up_blcok_1(F.interpolate(n2,size=n1.shape[2:],mode='bilinear',align_corners=True))
            n2 = self.up_blcok_2(F.interpolate(n2,size=in_.shape[2:],mode='bilinear',align_corners=True))

            xx = torch.linspace(-1, 1, size[1]).view(1, -1).repeat(size[0], 1)
            yy = torch.linspace(-1, 1, size[0]).view(-1, 1).repeat(1, size[1])
            xx = xx.view(1, size[0], size[1])
            yy = yy.view(1, size[0], size[1])
            grid = torch.cat((xx, yy), 0).float().unsqueeze(0).repeat(in_.shape[0], 1, 1, 1)
            grid = grid.clone().detach().requires_grad_(False)
            if in_.is_cuda:
                grid = grid.cuda()

            filed_residal = self.wrap_filed(n2)
            self.wrap_grid = grid + filed_residal

        elif self.mode=="Affine":
            identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)
            if in_.is_cuda:
                identity_theta = identity_theta.cuda()

            n3 = self.down_block_3(n2)
            deta = self.deta(n3)
            bsize = deta.shape[0]
            self.affine_matrix = deta.view(bsize,-1) + identity_theta.unsqueeze(0).repeat(bsize, 1)
            self.wrap_grid = F.affine_grid(self.affine_matrix.view(-1, 2, 3), in_.size(),align_corners=True).permute(0, 3, 1, 2)

    def wrap(self, x):
        if not x.shape[-1] == self.wrap_grid.shape[-1]:
            sampled_grid = F.interpolate(self.wrap_grid, size=x.shape[2:], mode='bilinear', align_corners=True)
            wrap_x = F.grid_sample(x, sampled_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
        else:
            wrap_x = F.grid_sample(x, self.wrap_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
        return wrap_x

    def wrap_inverse(self, x):
        t1 ,t2 = self.affine_matrix.view(-1, 2, 3)[:,:,:2],self.affine_matrix.view(-1, 2, 3)[:,:,2].unsqueeze(2)
        matrix_inverse = torch.cat((t1.inverse(),-t2),dim=2)
        sampled_grid = F.affine_grid(matrix_inverse, x.size(),align_corners=True)
        wrap_x = F.grid_sample(x, sampled_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return wrap_x

class PPM(nn.Module):
    def __init__(self,in_ch):
        super(PPM, self).__init__()
        self.conv = convblock(in_ch, 128, 3, 1, 1)
        self.b0 = nn.Sequential(
            nn.AdaptiveMaxPool2d(9),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )

        self.b1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(5),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(3),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fus = convblock(640, 128, 1, 1, 0)
        self.score=nn.Conv2d(128,1,1,1,0)

    def forward(self, x):
        x_size = x.size()[2:]
        x = self.conv(x)
        b0 = F.interpolate(self.b0(x), x_size, mode='bilinear', align_corners=True)
        b1 = F.interpolate(self.b1(x), x_size, mode='bilinear', align_corners=True)
        b2 = F.interpolate(self.b2(x), x_size, mode='bilinear', align_corners=True)
        b3 = F.interpolate(self.b3(x), x_size, mode='bilinear', align_corners=True)
        out = self.fus(torch.cat((b0, b1, b2, b3, x), 1))
        return out

class MAM(nn.Module):
    def __init__(self):
        super(MAM, self).__init__()
        self.stn = MySTN(256, "Affine")

        self.fus1 = convblock(256, 64, 1, 1, 0)
        self.alpha = nn.Conv2d(128, 1, 1, 1, 0)
        self.bata = nn.Conv2d(128, 1, 1, 1, 0)
        self.fus2 = convblock(128, 64, 1, 1, 0)

        self.dynamic_filter = nn.Conv2d(128,3*3*128,3,1,1)
        self.fus3 = convblock(128, 64, 1, 1, 0)
        self.combine = convblock(192,128,3,1,1)

    def forward(self, gr, gt):

        self.stn(torch.cat([gr,gt],dim=1))
        in1 = self.fus1(torch.cat([gr, self.stn.wrap(gt)],dim=1))

        affine_gt = self.alpha(gr)*gt + self.bata(gr)
        in2 = self.fus2(gr+affine_gt)

        filter = self.dynamic_filter(gr)
        in3 =  self.fus3(kernel2d_conv(gt,filter,3)+gr)
        return self.combine(torch.cat([in1,in2,in3],dim=1))



class LSTMCell(nn.Module):

    def __init__(self):
        super(LSTMCell, self).__init__()
        self.fus = convblock(256,128,1,1,0)
        self.conv = nn.Conv2d(256,512,3,1,1,bias=True)

    def forward(self, rgb, t,cur_state):
        h_cur, c_cur = cur_state
        in_ = self.fus(torch.cat([rgb, t],dim=1))
        combined = torch.cat([in_, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 128, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class MCLSTMCell(nn.Module):

    def __init__(self):
        """
        Modality Correction ConvLSTMCell
        """
        super(MCLSTMCell, self).__init__()
        #global-spatial context enhancement
        self.gc_enhance = nn.Conv2d(128,128,3,1,1)

        #modalities alignment at spatial location and pixel-wise correlation
        self.stn = MySTN(256,"Affine")
        self.fus1 = convblock(256,64,1,1,0)
        self.alpha = nn.Conv2d(128,1,1,1,0)
        self.bata = nn.Conv2d(128,1,1,1,0)
        self.fus2 = convblock(128, 64, 1, 1, 0)

        self.conv = nn.Conv2d(256,512,3,1,1,bias=True)

    def forward(self, rgb, t, global_context, cur_state):
        h_cur, c_cur = cur_state

        self.stn(torch.cat([rgb,t],dim=1))
        in1 = self.fus1(torch.cat([rgb, self.stn.wrap(t)],dim=1))

        affine_t = self.alpha(rgb)*t + self.bata(rgb)
        in2 = self.fus2(rgb+affine_t)

        combined = torch.cat([in1,in2,h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, 128, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        gc_enhance = torch.tanh(self.gc_enhance(global_context))

        c_next = f * c_cur + i * g + gc_enhance
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.backbone = vgg16()
        #transition
        trans_layers_mapping = [[128, 128], [256, 128], [512, 128], [512, 128]]
        self.trans = nn.ModuleList()
        for mapp in trans_layers_mapping:
            self.trans.append(convblock(mapp[0], mapp[1], 3, 2, 1))
        self.globalcontex = PPM(128)
        self.mam = MAM()
        self.topdown = MCLSTMCell()
        self.buttonup = MCLSTMCell()
        self.lstm_refine = LSTMCell()

        self.score = nn.Conv2d(128, 1, 1, 1, 0)

    def forward(self, rgb, t):
        size = rgb.shape[2:]
        Rh = self.backbone(rgb)[1:]
        Th = self.backbone(t)[1:]
        for i in range(len(Rh)):
            Rh[i] = self.trans[i](Rh[i])
            Th[i] =self.trans[i](Th[i])

        gr = self.globalcontex(Rh[-1])
        gt = self.globalcontex(Th[-1])

        global_context = F.interpolate(self.mam(gr,gt),size=Rh[0].shape[2:],mode='bilinear',align_corners=True)
        scores = [F.interpolate(self.score(global_context),size=size,mode='bilinear',align_corners=True)]

        featnums = len(Rh)
        #print(featnums)

        refine_hide_feats  = []

        Rh =[F.interpolate(feat,size=Rh[0].shape[2:],mode='bilinear',align_corners=True) for feat in Rh]
        Th = [F.interpolate(feat, size=Th[0].shape[2:], mode='bilinear', align_corners=True) for feat in Th]
        cur_state_topdown = [torch.zeros_like(Rh[0]).detach(), torch.zeros_like(Rh[0]).detach()]
        cur_state_buttonup = [torch.zeros_like(Rh[0]).detach(), torch.zeros_like(Rh[0]).detach()]
        cur_state_refine = [torch.zeros_like(Rh[0]).detach(), torch.zeros_like(Rh[0]).detach()]
        for i in range(featnums):
            cur_state_topdown = self.topdown(Rh[featnums-i-1], Th[featnums-i-1],global_context, cur_state_topdown)
            cur_state_buttonup = self.buttonup(Rh[i],Th[i],global_context, cur_state_buttonup)
            cur_state_refine = self.lstm_refine(cur_state_topdown[0],cur_state_buttonup[0], cur_state_refine)
            refine_hide_feats.append(cur_state_refine[0])

        for feat in refine_hide_feats:
            scores.append(F.interpolate(self.score(feat),size=size,mode='bilinear',align_corners=True))
        return scores

    def load_pretrained_model(self):
        st = torch.load("vgg16.pth")
        st2 = {}
        for key in st.keys():
            st2['base.' + key] = st[key]
        self.backbone.load_state_dict(st2)
        print('loading pretrained model success!')


if __name__ == "__main__":
    rgb = torch.rand(2, 3, 352, 352)
    t = rgb
    net = Mynet()
    map_list= net(rgb, t)
    torch.save(net.state_dict(),'test.pth')
    print(len(map_list),map_list[0].shape)
