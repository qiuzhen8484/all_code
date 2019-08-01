# -*-coding:UTF-8-*-

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
from PIL import Image
import os.path
from torch.utils.data import Dataset

class ValDataset(Dataset):
    """`ASOCT-2 class
    """
    def __init__(self, eye_name, transform=None, target_transform=None):
        super(ValDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = './data'
        # now load the picked numpy arrays

        self.test_data = []
        # self.test_labels = []
        for root, dirs, files in os.walk(self.data):
            for file in files:
                if file.split('_')[0] + '_' + file.split('_')[-4] == eye_name:
                    self.test_data.append(file)

    def __getitem__(self, index):
        img_name = self.test_data[index]
        img = os.path.join(self.data, img_name)
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)
        return img_name, img

    def __len__(self):
        return len(self.test_data)


def predict_per_eye(gindex="0"):
    if gindex != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = gindex

    net = torch.load('./model/shufflenet_avgpool64_acc95.80880447258322.pkl')
    net.eval()
    transform_test = transforms.Compose([
        transforms.Resize((576, 192)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    eye_name = []
    for name in os.listdir('./data'):
        if name.split('_')[0] + '_' + name.split('_')[-4] not in eye_name:
            eye_name.append(name.split('_')[0] + '_' + name.split('_')[-4])

    print('the number of eye: ' + str(len(eye_name)))

    print("Best model has already loaded")
    print('================================================')
    if os.path.exists('./log/every_img_predict_class.txt'):
        os.remove('./log/every_img_predict_class.txt')
    txt = open('./log/every_img_predict_class.txt', 'x')

    for eye_id in eye_name:
        test_dataset = ValDataset(eye_name=eye_id, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        positive_num = 0
        total = 0

        for _, (im_name, inputs) in enumerate(test_loader):
            if gindex == "-1":
                net.cpu()
                inputs = Variable(inputs)
            else:
                net.cuda()
                inputs = Variable(inputs.cuda())
            outputs = net(inputs)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            txt.write(im_name[0] + ':  ' + str(predicted[0]) + '\n')
            if predicted[0] == 1:
                positive_num += 1

        if positive_num >= 0.5 * total:
            txt.write('the predicted label of ' + eye_id[-1] + ' eye' + ' of ' + eye_id[:-2] + ' is ' + str(1) + '\n' + '\n')
            print('the predicted label of ' + eye_id[-1] + ' eye' + ' of ' + eye_id[:-2] + ' is ' + str(1))
        else:
            txt.write('the predicted label of ' + eye_id[-1] + ' eye' + ' of ' + eye_id[:-2] + ' is ' + str(0) + '\n' + '\n')
            print('the predicted label of ' + eye_id[-1] + ' eye' + ' of ' + eye_id[:-2] + ' is ' + str(0))

    txt.close()


if __name__ == '__main__':
    predict_per_eye(gindex="7")
