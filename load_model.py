import torch
from models.networks import define_G,define_D
from collections import OrderedDict
from torchvision.transforms import Compose, Normalize, ToTensor

def preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    preprocessing = Compose([
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img)

def dis(data):
    model_dict = torch.load("checkpoints/MDVA_all_active_withoutFeat_rotate35/15_net_D_B.pth")
    new_dict = OrderedDict()
    for k, v in model_dict.items():
        # load_state_dict expects keys with prefix 'module.'
        new_dict["module." + k] = v

    # make sure you pass the correct parameters to the define_G method
    generator_model = define_G(input_nc=1,output_nc=1,ngf=64,netG="resnet_9blocks",
                                norm="batch",use_dropout=True,init_gain=0.02,gpu_ids=[0])

    d_model=define_D(3, 64, 'basic',3,'instance','normal',0.02,[0])

    d_model.load_state_dict(new_dict)

    # normal_tensor = preprocess_image(data['A'])

    return d_model(data['A'])




