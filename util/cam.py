import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, ToTensor
device = torch.device('cuda')
import torch.nn as nn
import torch.nn.functional as F

class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al.
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''

    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers

        self.h = self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)

        self.activations = []
        self.grads = []

    def forward_hook(self, module, input, output):
        self.activations.append(output[0])
        # self.activations = output[0]

    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())
        # self.grads = grad_output[0].detach()
    def calculate_cam(self, model_input):
        if self.use_cuda:

            self.model.to(device)  # Module.to() is in-place method
            model_input = model_input.to(device)  # Tensor.to() is not a in-place method
        self.model.eval()

        cams = []
        for i in range(len(model_input)):
            # forward
            self.activations = []
            y_hat = self.model(model_input[i].unsqueeze(dim=0))
            # self.h.remove()
            max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)



            # backward
            self.model.zero_grad()
            y_c = y_hat[0, max_class]

            # neg_class = 1 - max_class  #观察二分类时另一类的CAM
            # y_c = y_hat[0, neg_class]  #观察二分类时另一类的CAM
            self.grads = []
            y_c.backward()

            # get activations and gradients
            activations = self.activations[0].cpu().data.numpy().squeeze()
            grads = self.grads[0].cpu().data.numpy().squeeze()
            self.activations = []
            self.grads = []
            # grads_new = np.abs(grads) * (grads > 1.3903e-10)


            # calculate weights
            weights = np.mean(grads.reshape(grads.shape[0], -1), axis=1)
            # weights = np.mean(np.abs(grads).reshape(grads.shape[0], -1), axis=1)#测试+—梯度
            weights = weights.reshape(-1, 1, 1)
            cam = (weights * activations).sum(axis=0)

#######################LayerCAM#####################
            # grads_new = grads * (grads < 10e-5)
            grads_new = np.abs(grads)
            cam = (grads_new * activations).sum(axis=0)
#################################################


            cam = np.maximum(cam, 0)  # ReLU
            if cam.max()!=0:
                cam = cam / cam.max()
            cams.append(cam)
        return cams,y_hat.to(device),max_class

    @staticmethod
    def show_cam_on_image(image, cam):
        # image: [H,W,C]
        h, w = image.shape[:2]

        cam = cv2.resize(cam, (h, w))
        cam = cam / cam.max()
        heatmap = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)  # [H,W,C]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        image = image / image.max()
        heatmap = heatmap / heatmap.max()

        result = 0.4 * heatmap + 0.6 * image
        result = result / result.max()

        cv2.imwrite('result.png', result*255)

        plt.figure()
        plt.imshow((result * 255).astype(np.uint8))
        plt.colorbar(shrink=0.8)
        plt.tight_layout()
        plt.show()
        cv2.waitKey(0)

    @staticmethod
    def preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        preprocessing = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        return preprocessing(img.copy()).unsqueeze(0)


if __name__ == '__main__':
    image_path = 'full_image/tbx11k/tb0037.png'
    # image_path = 'masked_image/1/CHNCXR_0327_1.png'
    # image_path = 'masked_image/0/CHNCXR_0097_0.png' # (224,224,3)
    image = cv2.imread(image_path)  # (224,224,3)
    # image = cv2.imread('/home/dzy-lab/projects/TBdata/train/1/dog.66.jpg')  # (224,224,3)
    # image = cv2.imread('both.png')  # (224,224,3)
    image = cv2.resize(image,[224,224])
    input_tensor = GradCAM.preprocess_image(image)






    # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_lastlayer/Seg_resnet50/49.pth')
    # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_lastlayer/Seg_resnet18/24.pth')
    # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_lastlayer/without_Seg_resnet18/24.pth')
    # model = torch.load('/home/dzy-lab/projects/TB_detection/trained_models/tbx11k_active_01_resnet50/9.pth')
    # grad_cam = GradCAM(model, model.resnet_layer[7], 224)#resnet18

    # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_lastlayer/Seg_mobilenetv3_small_100/7.pth')
    # # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_lastlayer/without_Seg_mobilenetv3_small_100/9.pth')
    # grad_cam = GradCAM(model, model.resnet_layer[2][5], 224)  #1   3   5

    # model = torch.load('/home/dzy-lab/projects/TB_detection/trained_models/TBX11Ktrainval_withoutSeg_densenet121/7.pth')
    # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_BigLastlayer/Seg_densenet121/15.pth')
    # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_lastlayer/without_Seg_densenet121/15.pth')
    # model = torch.load('/home/dzy-lab/projects/TB_detection/trained_models/tbx11k_active_01_densenet121/9.pth')
    # grad_cam = GradCAM(model, model.resnet_layer[0].denseblock4.denselayer16, 224)#dense [block3 layer24] [block2 layer12] [block 4 layer16]


    # # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_lastlayer/without_Seg_vgg16/7.pth')
    # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_bigLastlayer/Seg_SZ_vgg16/24.pth')
    model = torch.load('/home/dzy-lab/projects/TB_detection/trained_models_transform/tbx11k_active_01_vgg16/10.pth')
    grad_cam = GradCAM(model, model.resnet_layer[0][30], 224)  # vgg   16  23  30

    # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_lastlayer/Seg_inception_v3/10.pth')
    #
    # grad_cam = GradCAM(model, model.resnet_layer[17].branch_pool, 224)


    ####################################增大最后一层分辨率
    # model = torch.load('/home/dzy-lab/projects/TB_detection/trained_models_lastlayer/TBX11Ktrainval_withoutSeg_densenet121/7.pth')
    # grad_cam = GradCAM(model, model.resnet_layer[5].denselayer12, 224)

    # model = torch.load('/home/dzy-lab/projects/TB_detection/trained_models/TBX11Ktrainval_withoutSeg01_densenet121/6.pth')
    # grad_cam = GradCAM(model, model.resnet_layer[0].denseblock2.denselayer12, 224)

    # model = torch.load('/home/dzy-lab/projects/vechle/trained_models_bigLastlayer/Seg_SZ_inception_v3/17.pth')
    # grad_cam = GradCAM(model, model.resnet_layer[9], 224)

    # model = torch.load('/home/dzy-lab/projects/TB_detection/trained_models_lastlayer/TBX11Ktrainval_withoutSeg_vgg16/14.pth')
    # grad_cam = GradCAM(model, model.resnet_layer, 224)


    cam,outputs,max_class = grad_cam.calculate_cam(input_tensor)  ####maxclass为预测值
    # label = torch.tensor(int(image_path.split('/')[1])).to(device).view(1)  ############label为真是的值
    label = torch.tensor(1).to(device).view(1)
    loss_func = nn.NLLLoss()
    loss = loss_func(outputs, label)




    GradCAM.show_cam_on_image(image, cam[0])#######测试显示，可以注释掉
