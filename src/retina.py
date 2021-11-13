from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.models.unet import _get_sfs_idxs, model_sizes, hook_outputs

class Retina_Net_multi_head(nn.Module):
    def __init__(self,encoder,n_classes,img_size=256,final_bias=0.,chs=256,n_anchors=9,flatten=True):
        super().__init__()
        self.n_classes,self.flatten = n_classes,flatten
        sfs_szs = model_sizes(encoder,size=(img_size,img_size)) #grab each layer of outputs
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs))) #[6,5,4,2] = [C4,C3,C2,C1]
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs]) # hook of output [C4,C3,C2,C1]
        self.encoder = encoder #backbone
        
        self.c5top5 = conv2d(sfs_szs[-1][1],chs,ks=1,bias=True)
        self.c5top6 = conv2d(sfs_szs[-1][1],chs,ks=3,stride=2,bias=True)
        self.p6top7 = nn.Sequential(relu(),conv2d(chs,chs,ks=3,stride=2,bias=True))
        
        #TODO: P3P4,smoother
        self.latc4 = conv2d(sfs_szs[-2][1],chs,ks=1,bias=True)
        self.latc3 = conv2d(sfs_szs[-3][1],chs,ks=1,bias=True)
        self.smoother = conv2d(chs,chs,ks=3, stride=1,bias=True) # to remove upsample artifact
        
        self.classifier = self._head(n_classes,n_anchors,final_bias,chs=chs)
        self.box_regressor = self._head(4,n_anchors,final_bias,chs=chs)
        
    def _head(self,n_chs,n_anchors,final_bias=0.,chs=256):
        layers = [self.smoother] + [conv_layer(chs,chs,norm_type=None,bias=True) for _ in range(4)]
        layers = layers + [conv2d(chs,n_chs * n_anchors,bias=True)]
        #use kaiming init first, or adjust the last layer weights accordingly
        #layers[-1].bias.data.zero_().add_(final_bias)
        #layers[-1].weight.data.fill_(0)
        return nn.Sequential(*layers)
    
    def _flatten(self,x,n_classes):
        bs,ch,h,w = x.shape
        x = x.permute(0,2,3,1).contiguous()
        return x.view(bs,-1,n_classes)
        
        
    def forward(self,xb):
        c5 = self.encoder(xb)
        p_states = [self.c5top5(c5.clone()),self.c5top6(c5)]
        p_states.append(self.p6top7(p_states[1]))
        
        #TODO P3P4,P6P7
        c4 = self.sfs[0].stored
        #print(c4.shape[-2:])
        p4 = self.latc4(c4) + F.interpolate(p_states[0],c4.shape[-2:],mode='nearest')
        p_states = [p4] + p_states
        c3 = self.sfs[1].stored
        p3 = self.latc3(c3) + F.interpolate(p_states[0],c3.shape[-2:],mode='nearest')
        p_states = [p3] + p_states
        
        cls, box = [],[]
        for p in p_states:
            cls_p = self.classifier(p)
            cls.append(self._flatten(cls_p,self.n_classes))
            box_p = self.box_regressor(p)
            box.append(self._flatten(box_p,4))
        return [torch.cat(box,dim=1),torch.cat(cls,dim=1)]
        
    
    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()