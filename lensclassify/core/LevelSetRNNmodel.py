from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from visdom import Visdom
viz = Visdom()

class LevelSet_CNN_RNN_STN(object):

    def __init__(self):
        self.lambda_1 = 0.75
        self.lambda_2 = 0.005
        self.lambda_3 = 0.00
        self.lambda_shape = 0.001
        self.lambda_rnn = 0
        self.small_e = 0.00001
        self.e_ls = 1.0 / 128.0
        self.InnerAreaOption = 2
        self.UseLengthItemType = 1
        self.isShownVisdom = 1
        self.ShapePrior = 0
        self.RNNEvolution = 0
        self.CNNEvolution = 1
        self.inputSize = (512,512)
        self.ShapeTemplateName = ''
        self.gpu_num = 0
        self.GRU_hiddenSize = 0
        self.GRU_inputSize = 0
        self.GRU_TimeLength = 1
        self.GRU_Dimention = 2
        self.GRU_Number = 0
        self.HasGRU = 0
        self.UseHigh_Hfuntion = 0  # for length item
        self.PrintLoss = 1

        # This is calculate the gradient of the Image
        self.Sobelx = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        self.Sobely = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        WeightX = np.asarray([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        WeightY = np.asarray([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        WeightX = WeightX[np.newaxis, np.newaxis, :, :]
        WeightY = WeightY[np.newaxis, np.newaxis, :, :]
        WeightX = torch.FloatTensor(WeightX)
        WeightY = torch.FloatTensor(WeightY)
        self.Sobelx.weight.data = WeightX
        self.Sobely.weight.data = WeightY
        self.Sobelx.weight.requires_grad = False
        self.Sobely.weight.requires_grad = False

        self.Dif_xx = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        Dif_xx_weight = np.asarray([[0, 0, 0], [1, -2, 1], [0, 0, 0]])

        self.Dif_yy = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        Dif_yy_weight = np.asarray([[0, 1, 0], [0, -2, 0], [0, -1, 0]])

        self.Dif_xy = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        Dif_xy_weight = np.asarray([[0, -1, 1], [0, 1, -1], [0, 0, 0]])

        Dif_xx_weight = Dif_xx_weight[np.newaxis, np.newaxis, :, :]
        Dif_yy_weight = Dif_yy_weight[np.newaxis, np.newaxis, :, :]
        Dif_xy_weight = Dif_xy_weight[np.newaxis, np.newaxis, :, :]

        Dif_xx_weight = torch.FloatTensor(Dif_xx_weight)
        Dif_yy_weight = torch.FloatTensor(Dif_yy_weight)
        Dif_xy_weight = torch.FloatTensor(Dif_xy_weight)

        self.Dif_xx.weight.data = Dif_xx_weight
        self.Dif_yy.weight.data = Dif_yy_weight
        self.Dif_xy.weight.data = Dif_xy_weight

        self.Dif_xx.weight.requires_grad = False
        self.Dif_yy.weight.requires_grad = False
        self.Dif_xy.weight.requires_grad = False

        # The initial U_g and W_g
        self.Matrix_U_g = torch.nn.Linear(in_features=1, out_features=1, bias=False)
        self.Matrix_W_g = torch.nn.Linear(in_features=1, out_features=1, bias=False)

        self.softmax_2d = torch.nn.Softmax2d()
        self.sigmoid = torch.nn.Sigmoid()
        self.RNNLoss = torch.nn.BCEWithLogitsLoss()

        #The shape prior model
        self.SpatialTransformNet = STNNet()
        self.ShapePriorItem = ShapePriorBase()

    #Build a RNN module
    def BuildRNNModle1D(self):
        # The GRU level set evolution model
        self.GRU_hiddenSize = self.inputSize[0] * self.inputSize[1]
        self.GRU_inputSize  = self.inputSize[0] * self.inputSize[1]
        self.levelSetEvolutionModel = torch.nn.GRUCell(self.GRU_hiddenSize,self.GRU_inputSize)
        return self.levelSetEvolutionModel

    def BuildRNNModle2D(self):
        # The GRU level set evolution model
        self.GRU_hiddenSize = self.inputSize
        self.GRU_inputSize = self.inputSize
        self.levelSetEvolutionModel = []
        GRUOptions={
            'gpu_num':self.gpu_num
        }
        for i in range(self.GRU_Number):
            GRUInstance = GRU2D()
            GRUInstance.SetOptions(GRUOptions)
            self.levelSetEvolutionModel.append(GRUInstance)
        return 1

    def ForwardRNN2D(self, Init_LevelSetFunction, Image_):
        Levelset = Init_LevelSetFunction
        for i in range(self.GRU_Number):
            selectedModel = self.levelSetEvolutionModel[i]
            Levelset = selectedModel.forwardn(Levelset,Image_)
        return Levelset

    def ForwardRNN1D(self, Init_LevelSetFunction, Image_):
        HiddenLevelSet_1 = self.GenerateRLSInput(Image_=Image_, Phi_t0=Init_LevelSetFunction)
        ImageSize = Image_.size()
        LevelSetFunctionSize = Init_LevelSetFunction.size()
        Image_Input1 = torch.reshape(Image_,[ImageSize[0],-1])

        for i in range(self.GRU_TimeLength):
            HiddenLevelSet_Input1 = torch.reshape(HiddenLevelSet_1, [LevelSetFunctionSize[0], -1])
            HiddenLevelSet_2 = self.levelSetEvolutionModel(Image_Input1, HiddenLevelSet_Input1)
            ReshapeBackLevelSet = torch.reshape(HiddenLevelSet_2,ImageSize)
            HiddenLevelSet_1 = self.GenerateRLSInput(Image_=Image_Input1, Phi_t0=ReshapeBackLevelSet)
        return ReshapeBackLevelSet

    def BuildShapeModel(self, Options=2):
        LevelSet = self.ShapePriorItem.readOneLevelSet(self.ShapeTemplateName)
        if Options == 1:
            LevelSet = torch.from_numpy(LevelSet)
            LevelSet = torch.unsqueeze(input=LevelSet, dim=0)
            LevelSet = torch.unsqueeze(input=LevelSet, dim=0)

        if Options == 2:
            LevelSet_ = np.zeros_like(LevelSet)
            LevelSet_[LevelSet > 0] = 1
            LevelSet_[LevelSet < 0] = 0
            LevelSet = torch.from_numpy(LevelSet_)
            LevelSet = torch.unsqueeze(input=LevelSet, dim=0)
            LevelSet = torch.unsqueeze(input=LevelSet, dim=0)

        LevelSet = Variable(LevelSet).cuda(self.gpu_num).float()
        return LevelSet

    def SetPatameter(self,Lamda1=0.75, Lamda2=0.005, Lamda3=0.2, e_ls=1.0 / 128.0):
        self.lambda_1 = Lamda1
        self.lambda_2 = Lamda2
        self.lambda_3 = Lamda3
        self.e_ls = e_ls

    # Transform to [-1,1]
    # sigmoid is [0,1]
    # the output of the feature map is [0,1]
    # transform the output to [-0.5,0.5]
    # Generate the LevelSet Function
    def OutputLevelSet(self, FeatureMap):
        out = self.sigmoid(FeatureMap)
        out = out - 0.5
        return out

    # Generate the LevelSet Mask
    def LevelSetMask(self, FeatureMap):
        out = self.sigmoid(FeatureMap)
        out = out.data.cpu().numpy()
        out = out - 0.5
        out[out > 0] = 1
        out[out < 0] = 0
        return out

    # calculate the c1 and c2 of the level set
    # Notation: Two kinds of Item, one is U0(x,y)=H(phi(x,y)), second is U0(x,y)=phi(x,y)
    # Option = 1  U0xy = H(phi(x,y))
    # Option = 2  U0xy = Image
    def GetC1_C2(self, Phi_t0, Image, Option=1):
        SelectedTensor = self.HeavisideFunction(Phi_t0)
        if Option == 1:
            U0xy = self.HeavisideFunction(Phi_t0)
        if Option == 2:
            U0xy = Image
        c_1 = torch.sum(U0xy * SelectedTensor) / torch.sum(SelectedTensor)
        c_2 = torch.sum(U0xy * (1 - SelectedTensor)) / torch.sum(1 - SelectedTensor)
        return c_1, c_2

    # Calculate the curvature of the level set function
    def GetCurvature(self,Phi_t0):
        # Phi_t0 = HeavisideFunction(Phi_t0) #Get Level Set Map
        Item1 = self.Dif_xx(Phi_t0) * torch.pow(self.Sobely(Phi_t0), 2)
        Item2 = 2 * self.Sobelx(Phi_t0) * self.Sobely(Phi_t0) * self.Dif_xy(Phi_t0)
        Item3 = self.Dif_yy(Phi_t0) * torch.pow(self.Sobelx(Phi_t0), 2)
        Item4 = torch.pow(self.Sobelx(Phi_t0), 2) + torch.pow(self.Sobely(Phi_t0), 2)
        ItemAll = (Item1 + Item2 + Item3) / torch.pow(Item4, 3.0 / 2.0)
        return ItemAll

    # Put all the operator on the GPU
    def PutOnGpu(self,gpu_num):
        self.Sobelx.cuda(self.gpu_num)
        self.Sobely.cuda(self.gpu_num)
        self.Dif_xx.cuda(self.gpu_num)
        self.Dif_yy.cuda(self.gpu_num)
        self.Dif_xy.cuda(self.gpu_num)
        self.Matrix_U_g.cuda(self.gpu_num)
        self.Matrix_W_g.cuda(self.gpu_num)
        self.softmax_2d.cuda(self.gpu_num)
        self.sigmoid.cuda(self.gpu_num)
        self.SpatialTransformNet.cuda(self.gpu_num)

    # First step, generate RLS input:
    # x_t = g(I,\phi_t-1)
    def GenerateRLSInput(self, Image_, Phi_t0):
        Curvature_ = self.GetCurvature(Phi_t0)
        # U_g(I-c1)^2 + W_g(I-c2)^2
        # Notation: Two kinds of Item, one is U0(x,y)=H(phi(x,y)), second is U0(x,y)=phi(x,y)
        # We use the second term
        #C_1, C_2 = self.GetC1_C2(Phi_t0, Image_, Option=2)
        HeavisideLevelSet = self.HeavisideFunction(Phi_t0)
        U0xy = Image_
        C_1 = torch.sum(U0xy * HeavisideLevelSet) / torch.sum(HeavisideLevelSet)
        C_2 = torch.sum(U0xy * (1 - HeavisideLevelSet)) / torch.sum(1 - HeavisideLevelSet)

        Item1 = torch.pow(Image_ - C_1, 2)
        Item2 = torch.pow(Image_ - C_2, 2)
        FinalItem = Curvature_ - self.Matrix_U_g(Item1) + self.Matrix_W_g(Item2)

        return FinalItem

    def TestHeavisideFunction(self):
        x = np.linspace(-0.5, 0.5, 100)
        term1 = np.arctan(x / self.e_ls)
        term2 = 1.0 / 2.0 * (1.0 + (2.0 / np.pi) * term1)
        return term2

    def HeavisideFunction(self,FeatureMap):
        arctan_ = torch.atan(FeatureMap / self.e_ls)
        # c  = arctan_.data.cpu().numpy()
        H = 1.0 / 2.0 * (1.0 + (2.0 / np.pi) * arctan_)
        # d = H.data.cpu().numpy()
        return H

    def HighElsHeavisideFunction(self, FeatureMap):
        e_ls = 1 / 1024
        arctan_ = torch.atan(FeatureMap / e_ls)
        # c  = arctan_.data.cpu().numpy()
        H = 1.0 / 2.0 * (1.0 + (2.0 / np.pi) * arctan_)
        # d = H.data.cpu().numpy()
        return H

    def DiracDeltaFunction(self,FeatureMap):
        Output = (1.0 / np.pi) * self.e_ls / (self.e_ls * self.e_ls + torch.pow(FeatureMap, 2))
        return Output

    #If options changed, we rebuild the variable
    def ReBuildModel(self):
        #Build Shape Model

        #Build RNN Model

        return 1

    # InnerAreaOption = 1  U0xy = H(phi(x,y))
    # InnerAreaOption = 2  U0xy = Image
    # UseLengthItemType == 1 : |delta H(x)|
    # UseLengthItemType == 2 : dirace(phi(x,y))|delta phi(x,y)|
    # dic_options is a dictionary
    # Usage:
    # SetOptions({'InnerAreaOption':2,'UseLengthItemType':1,'isShownVisdom':1})
    def SetOptions(self,dic_options):
        self.InnerAreaOption = dic_options.get('InnerAreaOption',self.InnerAreaOption)
        self.UseLengthItemType = dic_options.get('UseLengthItemType',self.UseLengthItemType)
        self.isShownVisdom = dic_options.get('isShownVisdom',self.isShownVisdom)
        self.lambda_1 = dic_options.get('lambda_1',self.lambda_1)
        self.lambda_2 = dic_options.get('lambda_2',self.lambda_2)
        self.lambda_3 = dic_options.get('lambda_3',self.lambda_3)
        self.e_ls = dic_options.get('e_ls',self.e_ls)
        self.RNNEvolution = dic_options.get('RNNEvolution', self.RNNEvolution)
        self.ShapePrior = dic_options.get('ShapePrior', self.ShapePrior)
        self.inputSize = dic_options.get('inputSize', self.inputSize)
        self.ShapeTemplateName = dic_options.get('ShapeTemplateName', self.ShapeTemplateName)
        self.gpu_num = dic_options.get('gpu_num', self.gpu_num)
        self.CNNEvolution = dic_options.get('CNNEvolution', self.CNNEvolution)
        self.GRU_TimeLength =  dic_options.get('GRU_TimeLength', self.GRU_TimeLength)
        self.lambda_shape = dic_options .get('lambda_shape', self.lambda_shape)
        self.UseHigh_Hfuntion = dic_options.get('UseHigh_Hfuntion', self.UseHigh_Hfuntion)
        self.GRU_Dimention = dic_options.get('GRU_Dimention', self.GRU_Dimention)
        self.lambda_rnn = dic_options.get('Lamda_RNN', self.lambda_rnn)
        self.GRU_Number = dic_options.get('GRU_Number', self.GRU_Number)

        self.PutOnGpu(self.gpu_num)
        if self.RNNEvolution == 1:
            if self.GRU_Dimention == 1:
                self.BuildRNNModle1D()
            if self.GRU_Dimention == 2:
                self.BuildRNNModle2D()

    def LevelSetCombinedUNetLoss(self, Image_, OutPut_FeatureMap, LabelMap):

        return 1

    # FeatureMap is the size [255,255,N]
    # LabelMap is the size [255,255,N]
    # For binary classification, 0-background, 1-foreground
    # Option = 1  U0xy = H(phi(x,y))
    # Option = 2  U0xy = Image
    # UseLengthItem == 1 : |delta H(x)|
    # UseLengthItem == 2 : dirace(phi(x,y))|delta phi(x,y)|
    def LevelSetLoss(self,Image_,OutPut_FeatureMap,LabelMap):
        self.Image_ = Image_
        self.OutPut_FeatureMap = OutPut_FeatureMap
        self.LabelMap = LabelMap
        # Transform both to float tensor
        FeatureMap = self.OutPut_FeatureMap.float()
        LabelMap = self.LabelMap.float()

        # This is the level set.
        LevelSetFunction = self.OutputLevelSet(FeatureMap)

        preNum = LevelSetFunction.size()[2] * LevelSetFunction.size()[3]
        HeavisideLevelSet =  self.HeavisideFunction(LevelSetFunction)

        Loss_item1 = 0
        Loss_item2 = 0
        loss_item3 = 0
        if self.CNNEvolution == 1:
            # item 1 : |H(ls(x,y))-gt(x,y)|
            item1_2 = LabelMap  # 1*512*512
            item1_1 = torch.squeeze(HeavisideLevelSet, dim=1)  # 1*512*512

            # print(LabelMap)
            # a = classtensor.data.cpu().numpy()
            # print(classtensor)
            minums_ = item1_1 - item1_2
            # b = minums_.data.cpu().numpy()
            # print(minums_.size())
            item1_abs = torch.abs(minums_)
            item1_abs_pow = torch.pow(item1_abs, 2)
            Loss_item1 = torch.sum(item1_abs_pow) / preNum
            print('Loss Item=%f' % Loss_item1.data.cpu().numpy())

            # Item 2 the length of the zero-level set
            # UseLengthItem == 1 : |delta H(x)|
            # UseLengthItem == 2 : dirace(phi(x,y))|delta phi(x,y)|
            if self.UseHigh_Hfuntion == 0:
                HeavisideLevelSet_ = HeavisideLevelSet
            else:
                HeavisideLevelSet_ = self.HighElsHeavisideFunction(LevelSetFunction)

            if self.UseLengthItemType == 1:
                gradientX = self.Sobelx(HeavisideLevelSet)
                gradientY = self.Sobely(HeavisideLevelSet)
                gradientAll = gradientX + gradientY
                gradientAll = torch.abs(gradientAll)
                Loss_item2 = torch.sum(gradientAll) / preNum
                # print('Loss Item2=%f' % Loss_item2.data.cpu().numpy())
            else:
                gradientX = self.Sobelx(LevelSetFunction)
                gradientY = self.Sobely(LevelSetFunction)
                gradientAll = gradientX + gradientY
                gradientAll = torch.abs(gradientAll)
                item2_part1 = self.DiracDeltaFunction(LevelSetFunction)
                # Multiply element-wise
                Loss_item2 = torch.sum(item2_part1 * gradientAll) / preNum
                # print('Loss Item2=%f' % Loss_item2.data.cpu().numpy())
            print('Loss Item2=%f' % Loss_item2.data.cpu().numpy())

            # Item 3 the inner area and outer area
            # c_1 = torch.sum(SelectedTensor*SelectedTensor)/torch.sum(SelectedTensor)
            # c_2 = torch.sum(SelectedTensor*(1-SelectedTensor))/torch.sum(1-SelectedTensor)
            # Option = 1  U0xy = H(phi(x,y))
            # Option = 2  U0xy = Image
            # c_1, c_2 = GetC1_C2(Phi_t0=LevelSetFunction, Image_=InputImage, Option=option)
            if self.InnerAreaOption == 1:
                U0xy = HeavisideLevelSet
            if self.InnerAreaOption == 2:
                U0xy = Image_

            c_1 = torch.sum(U0xy * HeavisideLevelSet) / torch.sum(HeavisideLevelSet)
            c_2 = torch.sum(U0xy * (1 - HeavisideLevelSet)) / torch.sum(1 - HeavisideLevelSet)

            item3_part1_1 = U0xy - c_1
            item3_part1_2 = torch.abs(item3_part1_1)
            item3_part1_3 = torch.pow(item3_part1_2, 2)
            item3_part1_4 = item3_part1_3 * HeavisideLevelSet
            item3_part1_loss = torch.sum(item3_part1_4)

            item3_part2_1 = U0xy - c_2
            item3_part2_2 = torch.abs(item3_part2_1)
            item3_part2_3 = torch.pow(item3_part2_2, 2)
            item3_part2_4 = item3_part2_3 * (1 - HeavisideLevelSet)
            item3_part2_loss = torch.sum(item3_part2_4)

            loss_item3 = (item3_part1_loss + item3_part2_loss) / preNum
            print('Loss Item3=%f' % loss_item3.data.cpu().numpy())

        # item 4 shape prior
        # Add STN, to perform transformation and scale.
        Loss_Shape = 0
        ShapeOptions = 2
        show_transform_image = 1
        # Shape option = 1 : The shape template use the levelset presentation H(phi())
        # Shape option = 2 : The shape template use the 0-1 number
        if self.ShapePrior == 1:
            STNOutPut = self.SpatialTransformNet(LevelSetFunction)
            #Show the transformed Shape
            if show_transform_image == 1:
                transformedOut = STNOutPut.data.cpu().numpy()
                showImageVisdom(transformedOut)

            #Get the corresponding shape:
            ShapeTemplate = self.BuildShapeModel(Options=ShapeOptions)
            #Shape Prior
            ShapeItem1 = self.HeavisideFunction(STNOutPut)
            if ShapeOptions == 1:
                ShapeItem2 = self.HeavisideFunction(ShapeTemplate)
            if ShapeOptions == 2:
                ShapeItem2 = ShapeTemplate

            Loss_Shape_1 = ShapeItem1 - ShapeItem2
            Loss_Shape_2 = torch.abs(Loss_Shape_1)
            Loss_Shape_3 = torch.sum(Loss_Shape_2)/preNum
            Loss_Shape = Loss_Shape_3
            print('Loss Shape=%f' % Loss_Shape.data.cpu().numpy())

        # Item5 RNN Model
        LossRNN = 0
        RNNDimension = 2
        if self.RNNEvolution ==1:
            if RNNDimension == 1:
                RNN_Output = self.ForwardRNN1D(LevelSetFunction, Image_)
            if RNNDimension == 2:
                RNN_Output = self.ForwardRNN2D(LevelSetFunction, Image_)

            OutputSize = RNN_Output.size()
            RNN_Output_ = RNN_Output.view([OutputSize[0],-1])
            LabelMap_ = LabelMap.view([OutputSize[0],-1])
            LossRNN = self.RNNLoss(RNN_Output_, LabelMap_)

            #Loss for RNN model.
        AllLoss = self.lambda_1 * Loss_item1 + self.lambda_2 * Loss_item2 + self.lambda_3 * loss_item3 \
                  + self.lambda_shape * Loss_Shape + self.lambda_rnn * LossRNN
        print('All loss = %f' % AllLoss.data.cpu().numpy())
        return AllLoss

class ShapePriorBase(object):
    def __init__(self):
        self.FileName = ''

    def SetFileName(self,FileName_):
        self.FileName = FileName_

    def GetShapePrior(self):
        ShapeP = self.readOneLevelSet(self.FileName)
        return ShapeP

    def readOneLevelSet(self, FileName):
        with open(FileName, 'r+') as f:
            data = cPickle.load(f)
            return data

    def ShowLevelSet(self, graph):
        fig = plt.figure()
        ax = Axes3D(fig)
        # X, Y value
        X = np.arange(0, graph.shape[0], 1)
        Y = np.arange(0, graph.shape[1], 1)
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, graph, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        plt.show()

class STNNet(nn.Module):
    def __init__(self):
        super(STNNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(20, 30, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        TestImage = torch.randint(low=0, high=1, size=[1,1,512,512], dtype=torch.float32)
        #print(TestOutput)
        SizeOutput = self.localization(TestImage)
        self.LocSize = SizeOutput.size()

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.LocSize[1] * self.LocSize[2] * self.LocSize[3], 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.LocSize[1] * self.LocSize[2] * self.LocSize[3])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        return x

class GRU2D(nn.Module):
    def __init__(self):
        super(GRU2D, self).__init__()
        #For z
        self.z_Item_Uz = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.z_Item_Wz = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.r_Item_Ur = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.r_Item_Wr = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.h_Item_Uh = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.h_Item_Wh = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.Matrix_U_g = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.Matrix_W_g = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        self.sigmoid_  = nn.Sigmoid()
        self.tanh_     = nn.Tanh()
        self.forwardNum = 5
        self.gpu_num = 1
        self.e_ls = 1.0/32

        self.Sobelx = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        self.Sobely = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        WeightX = np.asarray([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        WeightY = np.asarray([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        WeightX = WeightX[np.newaxis, np.newaxis, :, :]
        WeightY = WeightY[np.newaxis, np.newaxis, :, :]
        WeightX = torch.FloatTensor(WeightX)
        WeightY = torch.FloatTensor(WeightY)
        self.Sobelx.weight.data = WeightX
        self.Sobely.weight.data = WeightY
        self.Sobelx.weight.requires_grad = False
        self.Sobely.weight.requires_grad = False

        self.Dif_xx = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        Dif_xx_weight = np.asarray([[0, 0, 0], [1, -2, 1], [0, 0, 0]])

        self.Dif_yy = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        Dif_yy_weight = np.asarray([[0, 1, 0], [0, -2, 0], [0, -1, 0]])

        self.Dif_xy = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False, padding=1)
        Dif_xy_weight = np.asarray([[0, -1, 1], [0, 1, -1], [0, 0, 0]])

        Dif_xx_weight = Dif_xx_weight[np.newaxis, np.newaxis, :, :]
        Dif_yy_weight = Dif_yy_weight[np.newaxis, np.newaxis, :, :]
        Dif_xy_weight = Dif_xy_weight[np.newaxis, np.newaxis, :, :]

        Dif_xx_weight = torch.FloatTensor(Dif_xx_weight)
        Dif_yy_weight = torch.FloatTensor(Dif_yy_weight)
        Dif_xy_weight = torch.FloatTensor(Dif_xy_weight)

        self.Dif_xx.weight.data = Dif_xx_weight
        self.Dif_yy.weight.data = Dif_yy_weight
        self.Dif_xy.weight.data = Dif_xy_weight

        self.Dif_xx.weight.requires_grad = False
        self.Dif_yy.weight.requires_grad = False
        self.Dif_xy.weight.requires_grad = False

    def SetOptions(self, dic_options):
        self.gpu_num = dic_options.get('gpu_num', self.gpu_num)
        self.putonGPU()
        return 1

    def putonGPU(self):
        self.z_Item_Uz.cuda(self.gpu_num)
        self.z_Item_Wz.cuda(self.gpu_num)
        self.r_Item_Ur.cuda(self.gpu_num)
        self.r_Item_Wr.cuda(self.gpu_num)
        self.h_Item_Uh.cuda(self.gpu_num)
        self.h_Item_Wh.cuda(self.gpu_num)
        self.Matrix_U_g.cuda(self.gpu_num)
        self.Matrix_W_g.cuda(self.gpu_num)

        self.Sobelx.cuda(self.gpu_num)
        self.Sobely.cuda(self.gpu_num)
        self.Dif_xx.cuda(self.gpu_num)
        self.Dif_yy.cuda(self.gpu_num)
        self.Dif_xy.cuda(self.gpu_num)

    #Calculate the curvature of the level set function
    def GetCurvature(self, Phi_t0):
        a = Phi_t0.data.cpu().numpy()
        # Phi_t0 = HeavisideFunction(Phi_t0) #Get Level Set Map
        Item1 = self.Dif_xx(Phi_t0) * torch.pow(self.Sobely(Phi_t0), 2)
        Item2 = 2 * self.Sobelx(Phi_t0) * self.Sobely(Phi_t0) * self.Dif_xy(Phi_t0)
        Item3 = self.Dif_yy(Phi_t0) * torch.pow(self.Sobelx(Phi_t0), 2)
        Item4 = torch.pow(self.Sobelx(Phi_t0), 2) + torch.pow(self.Sobely(Phi_t0), 2)

        Item4Values = Item4.data.cpu().numpy()
        Item4Index = np.where(Item4Values == 0)
        ItemMask = np.zeros_like(Item4Values)
        ItemMask[Item4Index] = 1
        ItemMaskTensor = torch.from_numpy(ItemMask)
        ItemMaskTensor = Variable(ItemMaskTensor).cuda(self.gpu_num).float()

        ItemDivide = torch.pow(Item4, 3.0 / 2.0)
        # Prevent Divide by zero
        ItemDivide = ItemDivide + ItemMaskTensor
        ItemAll = (Item1 + Item2 + Item3) / ItemDivide

        return ItemAll

    def GenerateRLSInput(self, Image_, Phi_t0):
        Curvature_ = self.GetCurvature(Phi_t0)
        # U_g(I-c1)^2 + W_g(I-c2)^2
        # Notation: Two kinds of Item, one is U0(x,y)=H(phi(x,y)), second is U0(x,y)=phi(x,y)
        # We use the second term
        #C_1, C_2 = self.GetC1_C2(Phi_t0, Image_, Option=2)
        HeavisideLevelSet = self.HeavisideFunction(Phi_t0)
        U0xy = Image_
        C_1 = torch.sum(U0xy * HeavisideLevelSet) / torch.sum(HeavisideLevelSet)
        C_2 = torch.sum(U0xy * (1 - HeavisideLevelSet)) / torch.sum(1 - HeavisideLevelSet)

        Item1 = torch.pow(Image_ - C_1, 2)
        Item2 = torch.pow(Image_ - C_2, 2)
        FinalItem = Curvature_ - self.Matrix_U_g(Item1) + self.Matrix_W_g(Item2)

        return FinalItem

    def HeavisideFunction(self,FeatureMap):
        arctan_ = torch.atan(FeatureMap / self.e_ls)
        # c  = arctan_.data.cpu().numpy()
        H = 1.0 / 2.0 * (1.0 + (2.0 / np.pi) * arctan_)
        # d = H.data.cpu().numpy()
        return H

    def DiracDeltaFunction(self,FeatureMap):
        Output = (1.0 / np.pi) * self.e_ls / (self.e_ls * self.e_ls + torch.pow(FeatureMap, 2))
        return Output

    def forward1(self, LevelSets, Images):
        Input = self.GenerateRLSInput(Image_=Images, Phi_t0=LevelSets)
        Z_ = self.sigmoid_(self.z_Item_Uz(Input) + self.z_Item_Wz(Input))
        R_ = self.sigmoid_(self.r_Item_Ur(Input) + self.r_Item_Wr(Input))
        ht_ = self.tanh_(self.h_Item_Uh(Input) + self.h_Item_Wh(R_ * LevelSets))
        LevelSets = (1-Z_)*ht_ + Z_*LevelSets
        return LevelSets

    def forwardn(self, LevelSets, Images):
        for i in range(self.forwardNum):
            LevelSets = self.forward1(LevelSets, Images)
        return LevelSets

def showImageVisdom(Result_pre, CaptionName ='Input Image', TitleName = 'Origin RGB Image'):
    Result_pre = np.squeeze(Result_pre)
    ResultImg = Image.fromarray(Result_pre)
    ResultImg = ResultImg.resize([200, 200])
    ResultImg = np.asarray(ResultImg)
    # A = A[np.newaxis,:,:]
    viz.image(
        ResultImg,
        opts=dict(title=TitleName, caption=CaptionName)
    )

