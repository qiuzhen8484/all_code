import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

# from resNet_model import My_Resnet50
# from models import My_UNet512
from segmentation import DecideBoundaryLeftRight
from segmentation import TransformToBinaryImage
from utils import crop_boundry
from utils import calculate_Accuracy
from utils import compute_MSE_pixel
from utils import process_x_y
from utils import get_truth
from utils import MSE_pixel_loss
# from resNet_model import resnet_up

from skimage import segmentation
import scipy.io as sio
import argparse
import cv2
import numpy as np
import os
import cPickle as pkl
from pspnet import PSPNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description='PyTorch AS_OCT Demo')

parser.add_argument('--ori_data', type=str, default='./data/dataset/img',
                    help='dir of the all ori img')
parser.add_argument('--train_data', type=str, default='./data/dataset/train_data',
                    help='location of the data corpus')
parser.add_argument('--test_data', type=str, default='./data/dataset/test_data',
                    help='location of the data corpus')
parser.add_argument('--label_data', type=str, default='./data/dataset/train_label',
                    help='location of the data corpus')
parser.add_argument('--results', type=str,  default='./visual_results',
                    help='path to save the trained model')
parser.add_argument('--log', type=str,  default='./logs/log.txt',
                    help='note the best val loss')
parser.add_argument('--best_model', type=str,  default='./models/146_1e-10.pth',
                    help='the pretrain model')
parser.add_argument('--flag', type=str,  default='test',
                    help='the pretrain model')


parser.add_argument('--img_size', type=int, default=512,
                    help='the train img size')
parser.add_argument('--input_nc', type=int, default=3,
                    help='the channel of input img')
parser.add_argument('--n_class', type=int, default=4,
                    help='the channel of out img, decide the num of class')
parser.add_argument('--num_downs', type=int, default=7,
                    help='the num of the u-net block')
parser.add_argument('--gpu_num', type=int,  default=0,
                    help='the gpu id')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')


args = parser.parse_args()

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}
def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    #net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
    return net, epoch

# model = UnetGenerator(3,2,7)
snapshot = None
backend = 'densenet'
model, starting_epoch = build_network(snapshot, backend)

# model = resnet_up(pretrain=False)
print(args.gpu_num)
model.cuda(args.gpu_num)
lossfunc = nn.NLLLoss2d()
softmax_2d = nn.Softmax2d()

model.load_state_dict(torch.load(args.best_model))
print 'success the donwload the best model'

y_1 = []
y_2 = []

if args.flag=='train':
    args.test_data = args.train_data
img_list = os.listdir(args.test_data)
confusion = np.zeros([args.n_class, args.n_class])
for idx,img_path in enumerate(img_list):
    label_path = os.path.join(args.label_data,  img_path)
    label = cv2.imread(label_path)
    label = cv2.resize(label, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)[:, :, :1]

    ori_img_path = os.path.join(args.ori_data,img_path)
    img = cv2.imread(ori_img_path, 0)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 35, 110)
    # edges = cv2.Canny(img, 90, 150)
    # Sliding Window
    SlidingWindowSize = (20, 40)
    # SlidingWindowSize = (20,50)
    # Get Binary Map
    BinaryMap = TransformToBinaryImage(edges)
    imshape = edges.shape

    img2rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # topx, topy = DecideBoundaryTopDown(BinaryMap_, SlidingWindowSize, 580, 1462)
    ally = DecideBoundaryLeftRight(BinaryMap, SlidingWindowSize)
    # Draw The boundary
    BeginY = imshape[1] / 3
    newImage = img2rgb[BeginY:, ally[0]:ally[1]]

    resize_img = cv2.resize(newImage, (args.img_size, args.img_size))
    resize_img = np.transpose(resize_img, [2, 0, 1])
    resize_img = Variable(torch.from_numpy(resize_img)).float().cuda(args.gpu_num)
    resize_img = torch.unsqueeze(resize_img, 0)
    out,_ = model(resize_img)
    # pi = F.softmax(out)
    pi = torch.log(softmax_2d(out))
    ppi = np.argmax(pi.cpu().data.numpy(), 1).reshape((args.img_size, args.img_size))

    ## miou
    tmp_gt = label.reshape([-1])
    tmp_out = ppi.reshape([-1])
    for i in xrange(len(tmp_gt)):
        confusion[tmp_gt[i], tmp_out[i]] += 1


    # if idx%20:
    #     continue

    ## open and close
    new_Image = ppi.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new_Image = cv2.morphologyEx(new_Image, cv2.MORPH_CLOSE, kernel)
    # new_Image = morphology.convex_hull_object(new_Image)

    FinalShape = new_Image.copy()


    ori_img = cv2.imread(ori_img_path)

    ROI_img = ori_img[BeginY:, ally[0]:ally[1],:].copy()
    ROIShape = ROI_img.shape
    pred_img = cv2.resize(new_Image.astype(np.uint8), (ROI_img.shape[1], ROI_img.shape[0]))
    ROI_img = crop_boundry(ROI_img, pred_img)
    alpha_img = ori_img.copy()
    alpha_img[BeginY:, ally[0]:ally[1]] = ROI_img

    ROI_img = cv2.addWeighted(alpha_img, 0.4, ori_img, 0.6, 0)
    ## imshow the truth
    ROI_img = get_truth(ROI_img,img_path,ally,BeginY)
    save_name = os.path.join(args.results,img_path)
    cv2.imwrite(save_name, ROI_img)

    # predict the boundary
    FullImage = np.zeros(imshape)
    FinalShape = FinalShape * 1
    pred_imgFinal = cv2.resize(FinalShape.astype(np.uint8), (ROIShape[1], ROIShape[0]))
    FullImage[BeginY:, ally[0]:ally[1]] = pred_imgFinal
    cv2.imwrite('./tmp_data/%s.png'%img_path.split('.')[0], FullImage)
    boundarys = segmentation.find_boundaries(FullImage)
    boundarys = boundarys * 1
    boundarysxy = np.where(boundarys == 1)
    tmp_img = np.zeros(imshape)
    # with open('./logs/pspnet.pkl','w+') as f:
    #     pkl.dump([tmp_img,FullImage, img_path, ally,boundarys],f)
    MSE_loss = MSE_pixel_loss(tmp_img,FullImage, img_path, ally)
    print MSE_loss
    y_1.append(MSE_loss)
    # print loss, tmp_sum, other_sum
    print 'success %s'%(idx)
meanIU, pixelAccuracy, meanAccuracy, classAccuracy = calculate_Accuracy(confusion)
print('meanIU: {:.2f} | pixelAccuracy: {:.2f}  | meanAccuracy: {:.2f}| classAccuracy: {:.2f}'.format(meanIU, pixelAccuracy, meanAccuracy, classAccuracy))


y_1 = np.stack(y_1)
print np.mean(y_1,0)
# y_2 = np.stack(y_2)
# print np.mean(y_2)
