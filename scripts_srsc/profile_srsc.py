import argparse
import torch
import torchprof
from thop import profile, clever_format

from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet
from models.srsc import SRSC


def parse_args():
    parser = argparse.ArgumentParser(description='Serial Restoration of Shuffled Clips')
    parser.add_argument('--model', type=str, default='r21d', help='c3d/r3d/r21d')
    parser.add_argument('--mask', default=False, action='store_true', help='whether to use mask in TaskNet')
    parser.add_argument('--tl', type=int, default=3, help='tuple length')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()


    ########### model ##############
    # ===== time 
    # if args.model == 'c3d':
    #     base = C3D(with_classifier=False)
    # elif args.model == 'r3d':
    #     base = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    # elif args.model == 'r21d':   
    #     base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    # srsc = SRSC(base_network=base, feature_size=512, tuple_len=args.tl,
    #             hidden_dim=512, input_dim=512, mask=args.mask, p=0.5).cuda()  

    # x = torch.rand([8,args.tl,3,16,112,112]).cuda()

    # with torchprof.Profile(srsc, use_cuda=True) as prof:
    # 	srsc(x)
    # print(prof.display(show_events=False))


    # ===== params
    x_for_base = torch.rand([8*args.    ,3,16,112,112]).cuda()
    x_for_srsc = torch.rand([8,args.tl,3,16,112,112]).cuda()
    base = [
        C3D(with_classifier=False),
        R3DNet(layer_sizes=(1,1,1,1), with_classifier=False),
        R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)            
    ]
    srsc = [
        SRSC(base_network=b, feature_size=512, tuple_len=args.tl,
                hidden_dim=512, input_dim=512, mask=args.mask, p=0.5).cuda()
        for b in base
    ]
    for i in range(3):
        macs, params = profile(base[i], inputs=(x_for_base,))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)
        macs, params = profile(srsc[i], inputs=(x_for_srsc,))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)