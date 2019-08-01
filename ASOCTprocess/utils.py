
import numpy as np
import os.path
import random
import cv2
from skimage.segmentation import find_boundaries

import scipy.io as sci

# Rectancle :[x,y,width,height]   x is row, y is col
# img: grey scale Image_
def DrawSlidingWindow(Rectancle, img):
    x = Rectancle[0]
    y = Rectancle[1]
    width = Rectancle[2]
    height = Rectancle[3]
    NewImage = np.zeros(img.shape)
    where255 = np.where(img>0)
    NewImage[where255] = 255
    #Draw rectangle
    NewImage[x:x+height, y:y+width] = 255
    return  NewImage

def GetEdgeNumFromRectangle(Rectancle, img):
    x = Rectancle[0]
    y = Rectancle[1]
    width = Rectancle[2]
    height = Rectancle[3]

    PatchSize = (height, width)
    ImagePatch = np.zeros(PatchSize)
    ImagePatch = img[x:x + height, y:y + width]
    SumNum = np.sum(np.sum(ImagePatch))
    return SumNum

def TransformToBinaryImage(Image_):
    BinaryMap = np.zeros(Image_.shape)
    HasValue = np.where(Image_>0)
    BinaryMap[HasValue] = 1
    return BinaryMap

def Binary2Grey(Image_):
    NewImage = Image_ * 255
    return NewImage

def ChargeLenSegment(LineList):
    listLength = len(LineList)
    difflist = []
    for i in range(listLength-1):
        difflist.append(LineList[i+1]-LineList[i])

    #count the length of different segments:
    position =[]
    counts = []
    count = 0
    begincount = 0
    for j in range(len(difflist)):
        if difflist[j] == 1 and begincount == 0:
            position.append(j)
            begincount = 1
            count = 1

        elif difflist[j] == 1 and begincount == 1:
            count = count + 1
            if j == len(difflist)-1:
                counts.append(count)
        else:
            begincount = 0
            if count > 0:
                counts.append(count)
                count = 0
            continue

    if len(counts)==0:
        return -1

    #Find the longest segment:
    maxIndice = np.argmax(counts)
    Length = counts[maxIndice]
    BeginN = position[maxIndice]

    #Select The Middle
    MiddleCount = LineList[BeginN + Length / 2]

    return MiddleCount

def ChargeLenSegment2(LineList):
    minnum = np.min(LineList)
    return minnum

# Divided the rectancle into two regions, calculate the difference of edge Num / region area
# Rectancle :[x,y,width,height]   x is row y is col  x, y is the left top
def CalculateRegionDensityDiffereneUpDown(Rectancle, img):
    x = Rectancle[0]
    y = Rectancle[1]
    width = Rectancle[2]
    height = Rectancle[3]

    # Rectancle :[x,y,width,height]   x is row, y is col
    RectancleUp = [x,y,width,int(height/2)]
    RectancleDown = [int(x + height/2),y,width,int(height/2)]

    UpNum = GetEdgeNumFromRectangle(RectancleUp, img)
    DownNum = GetEdgeNumFromRectangle(RectancleDown, img)

    Difference = (UpNum - DownNum)/width * height
    return Difference

def CalculateRegionDensityDiffereneLeftRight(Rectancle, img):
    x = Rectancle[0]
    y = Rectancle[1]
    width = Rectancle[2]
    height = Rectancle[3]

    # Rectancle :[x,y,width,height]   x is row, y is col
    RectancleLeft = [x,y,int(width/2),height]
    RectancleRight = [x,int(y+width/2),int(width/2),height]

    LeftNum = GetEdgeNumFromRectangle(RectancleLeft, img)
    RightNum = GetEdgeNumFromRectangle(RectancleRight, img)

    Difference = (LeftNum - RightNum)/width * height
    return Difference

def CalculateRegionDensityDiffereneRightLeft(Rectancle, img):
    x = Rectancle[0]
    y = Rectancle[1]
    width = Rectancle[2]
    height = Rectancle[3]

    # Rectancle :[x,y,width,height]   x is row, y is col
    RectancleLeft = [x,y,int(width/2),height]
    RectancleRight = [x,int(y+width/2),int(width/2),height/2]

    LeftNum = GetEdgeNumFromRectangle(RectancleLeft, img)
    RightNum = GetEdgeNumFromRectangle(RectancleRight, img)

    Difference = (RightNum - LeftNum)/width * height
    return Difference

#get the average value of mid 50%
def Get50percentAverage(linelist):
    sortedlist = np.sort(linelist)
    allylen = len(linelist)
    allylen25 = int(allylen / 4)
    allylen75 = int(allylen / 4 * 3)
    xy50 = sortedlist[allylen25:allylen75]
    final = np.average(xy50)
    return final

#score every line
def ScoreLeftRight(edgeMap,Lines):
    scores = []
    imshape = edgeMap.shape
    for i in range(len(Lines)):
        R1BeginX = imshape[0] / 3 * 2
        R1BeginY = Lines[i]-200
        R1Width = 200
        R1Height = 200
        R1=[R1BeginX,R1BeginY,R1Width,R1Height]
        EdgeNumR1 = GetEdgeNumFromRectangle(R1, BinaryMap_)

        R2BeginX = imshape[0] / 3 * 2
        R2BeginY = Lines[i]
        R2Width = 200
        R2Height = 200
        R2 = [R2BeginX, R2BeginY, R2Width, R2Height]
        EdgeNumR2 = GetEdgeNumFromRectangle(R2, BinaryMap_)

        Score = np.square((EdgeNumR1 - EdgeNumR2)) / 4000
        scores.append(Score)
    return scores

#score every line, Linesx and Linesy is the centre of the rect
def ScoreTopDown(edgeMap, Linesx, Linesy):
    scores = []
    BinaryMap_ = edgeMap
    imshape = edgeMap.shape
    #width, height
    DetectBoxShape = [10, 10]
    #x, y is the centre of the rect
    for i in range(len(Linesx)):
        R1BeginX = Linesx[0] - DetectBoxShape[1]
        R1BeginY = Linesy[0]
        R1Width = DetectBoxShape[0]
        R1Height = DetectBoxShape[1]
        R1=[R1BeginX,R1BeginY,R1Width,R1Height]
        EdgeNumR1 = GetEdgeNumFromRectangle(R1, BinaryMap_)

        R2BeginX = Linesx[0]
        R2BeginY = Linesy[0]
        R2Width = DetectBoxShape[0]
        R2Height = DetectBoxShape[1]
        R2 = [R2BeginX, R2BeginY, R2Width, R2Height]
        EdgeNumR2 = GetEdgeNumFromRectangle(R2, BinaryMap_)

        Score = (EdgeNumR2 - EdgeNumR1 ) / 100
        scores.append(Score)
    return scores

#Decide the boundary box
#bounding box: x, y, width , height , x is row , y is col
#boundingBoxSize [0]width  [1]height
def DecideBoundaryLeftRight(Binaryedgemap,boundingBoxSize):

    BinaryMap_ = Binaryedgemap
    imshape = Binaryedgemap.shape
    edges = Binaryedgemap * 255
    ally = []
    # Left:
    # Begin at  x: 1/4, y: 1/5
    BeginX = imshape[0] / 3 * 2
    EndX = BeginX + 200
    BeginY = imshape[1] / 6
    EndY = imshape[1] / 2
    LengthY = int(EndY - BeginY)
    LengthX = int(EndX - BeginX)
    allxLeftRight = []
    allyLeftRight = []
    #for i in range(1, LengthX, 3):
    for i in range(1,LengthX, 3):
        #Every X correspond to a Y
        EdgeNumList = []
        EdgeNumDifferentList = []
        correspondy_ = []
        for j in range(LengthY):
            SlidingWindowPos = [BeginX+i, BeginY+j, boundingBoxSize[0], boundingBoxSize[1]]
            diff = CalculateRegionDensityDiffereneRightLeft(SlidingWindowPos, BinaryMap_)
            EdgeNum = GetEdgeNumFromRectangle(SlidingWindowPos, BinaryMap_)
            EdgeNumList.append(EdgeNum)
            EdgeNumDifferentList.append(diff)
            correspondy_.append(j)

        MaxEdgeNum = np.max(EdgeNumList)
        MaxDifference = np.max(EdgeNumDifferentList)
        #print('max difference is %f, max EdgeNum is %f' % (MaxDifference, MaxEdgeNum))

        # Divided the different into two classes,
        IntoClasses = 2
        EdgeUnit = MaxEdgeNum / IntoClasses
        DifferenceUnit = MaxDifference / IntoClasses

        # process the different part of the line
        #  EdgeNum > EdgeUnit/2  EdgeDiff > DifferenceUnit : boundary
        #  EdgeNum > EdgeUnit :  content
        #  EdgeNum < EdgeUnit/2:           blank
        EdgeNumArray = np.asarray(EdgeNumList)
        EdgeDiffArray = np.asarray(EdgeNumDifferentList)
        ConditionOne = np.where(EdgeNumArray > (EdgeUnit / 2))
        ConditionTwo = np.where(EdgeDiffArray > DifferenceUnit)

        #the points satisfy the condition
        CandidatePos = []
        for j_ in range(len(ConditionTwo[0])):
            data_j = ConditionTwo[0][j_]
            where_j = np.where(ConditionOne == data_j)
            if len(where_j[0]) > 0:
                CandidatePos.append(data_j)

        # the longest segment satisfy the segment
        # MedNum = ChargeLenSegment(CandidatePos)
        # if MedNum == -1:
        #     continue
        if len(CandidatePos) == 0:
            continue
        MedNum = ChargeLenSegment2(CandidatePos)

        cx = i
        cy = correspondy_[MedNum]
        allxLeftRight.append(BeginX+cx)
        allyLeftRight.append(BeginY+cy)
    #final = Get50percentAverage(allyLeftRight)
    Linescores = ScoreLeftRight(edges,allyLeftRight)
    maxscores = Linescores.index(np.max(Linescores))
    final = allyLeftRight[maxscores]
    ally.append(int(final))

    #Right
    # Begin at  x: 1/4, y: 1/5
    BeginX = imshape[0] / 3 * 2
    EndX = BeginX + 200
    BeginY = imshape[1] / 6 * 5
    EndY = imshape[1] / 2
    LengthY = int(BeginY -EndY)
    LengthX = int(EndX - BeginX)
    allxLeftRight = []
    allyLeftRight = []
    for i in range(1, LengthX, 3):
        #Every X correspond to a Y
        EdgeNumList = []
        EdgeNumDifferentList = []
        correspondy_ = []
        for j in range(LengthY):
            SlidingWindowPos = [BeginX+i, BeginY-j-boundingBoxSize[0], boundingBoxSize[0], boundingBoxSize[1]]
            #NewMap = DrawSlidingWindow(SlidingWindowPos, edges)
            #cv2.imwrite('F:\\MyCode\\LensSegmentation\\result\\imageResult.png', NewMap)

            diff = CalculateRegionDensityDiffereneLeftRight(SlidingWindowPos, BinaryMap_)
            EdgeNum = GetEdgeNumFromRectangle(SlidingWindowPos, BinaryMap_)
            EdgeNumList.append(EdgeNum)
            EdgeNumDifferentList.append(diff)
            correspondy_.append(j)

        MaxEdgeNum = np.max(EdgeNumList)
        MaxDifference = np.max(EdgeNumDifferentList)
        #print('max difference is %f, max EdgeNum is %f' % (MaxDifference, MaxEdgeNum))

        # Divided the different into two classes,
        IntoClasses = 2
        EdgeUnit = MaxEdgeNum / IntoClasses
        DifferenceUnit = MaxDifference / IntoClasses

        # process the different part of the line
        #  EdgeNum > EdgeUnit/2  EdgeDiff > DifferenceUnit : boundary
        #  EdgeNum > EdgeUnit :  content
        #  EdgeNum < EdgeUnit/2:           blank
        EdgeNumArray = np.asarray(EdgeNumList)
        EdgeDiffArray = np.asarray(EdgeNumDifferentList)
        ConditionOne = np.where(EdgeNumArray > EdgeUnit * 2/3)
        ConditionTwo = np.where(EdgeDiffArray > DifferenceUnit* 2/3)

        CandidatePos = []
        for j_ in range(len(ConditionTwo[0])):
            data_j = ConditionTwo[0][j_]
            where_j = np.where(ConditionOne == data_j)
            if len(where_j[0]) > 0:
                CandidatePos.append(data_j)

        # MedNum = ChargeLenSegment(CandidatePos)
        # if MedNum == -1:
        #     continue
        if len(CandidatePos) == 0:
            continue
        MedNum = ChargeLenSegment2(CandidatePos)

        cx = i
        cy = correspondy_[MedNum]
        allxLeftRight.append(BeginX+cx)
        allyLeftRight.append(BeginY-cy)

    # Get 50% in the middle for average
    #final = Get50percentAverage(allyLeftRight)
    #final = np.max(allyLeftRight)
    Linescores = ScoreLeftRight(edges, allyLeftRight)
    maxscores = Linescores.index(np.max(Linescores))
    final = allyLeftRight[maxscores]
    ally.append(int(final))

    return ally

#bounding box: x, y, width , height , x is row , y is col
#boundingBoxSize [0]width  [1]height
def DecideBoundaryTopDown(Binaryedgemap,boundingBoxSize,LeftP, RightP):

    BinaryMap_ = Binaryedgemap
    imshape = Binaryedgemap.shape
    edges = Binaryedgemap * 255
    allx = []
    ally = []

    # Top Right:
    # Begin at  x: 1/4, y: 1/5
    BeginX = imshape[0] / 3
    EndX = imshape[0] / 3 * 2
    BeginY = imshape[1] / 2
    EndY = RightP
    LengthY = int(EndY - BeginY)
    LengthX = int(EndX - BeginX)

    edges[:,BeginY] = 255
    edges[BeginX,:] = 255

    for i in range(1,LengthY, 3):
        #Every X correspond to a Y
        EdgeNumList = []
        EdgeNumDifferentList = []
        correspondx_ = []
        for j in range(LengthX):
            #Begin X and Begin Y is the centre of the rect,  bounding box: x, y, width , height , x is row , y is col
            #x, y is the left top
            SlidingWindowPos = [BeginX+j, BeginY+i, boundingBoxSize[0], boundingBoxSize[1]]
            #NewMap = DrawSlidingWindow(SlidingWindowPos, edges)
            #cv2.imwrite('F:\\MyCode\\LensSegmentation\\result\\imageResult.png', NewMap)
            diff = -CalculateRegionDensityDiffereneUpDown(SlidingWindowPos, BinaryMap_)
            EdgeNum = GetEdgeNumFromRectangle(SlidingWindowPos, BinaryMap_)
            EdgeNumList.append(EdgeNum)
            EdgeNumDifferentList.append(diff)
            correspondx_.append(j)

        MaxEdgeNum = np.max(EdgeNumList)
        MaxDifference = np.max(EdgeNumDifferentList)
        #print('max difference is %f, max EdgeNum is %f' % (MaxDifference, MaxEdgeNum))

        # Divided the different into two classes,
        IntoClasses = 2
        EdgeUnit = MaxEdgeNum / IntoClasses
        DifferenceUnit = MaxDifference / IntoClasses

        # process the different part of the line
        #  EdgeNum > EdgeUnit/2  EdgeDiff > DifferenceUnit : boundary
        #  EdgeNum > EdgeUnit :  content
        #  EdgeNum < EdgeUnit/2:           blank
        EdgeNumArray = np.asarray(EdgeNumList)
        EdgeDiffArray = np.asarray(EdgeNumDifferentList)
        ConditionOne = np.where(EdgeNumArray > (EdgeUnit/2))
        ConditionTwo = np.where(EdgeDiffArray > DifferenceUnit * 4/3)

        #the points satisfy the condition
        CandidatePos = []
        for j_ in range(len(ConditionTwo[0])):
            data_j = ConditionTwo[0][j_]
            where_j = np.where(ConditionOne == data_j)
            if len(where_j[0]) > 0:
                CandidatePos.append(data_j)

        if len(CandidatePos) == 0:
            continue
        #MedNum = ChargeLenSegment2(CandidatePos)

        allxtemp = []
        allytemp = []
        #X and Y is the left top point of the rect
        for i_t in range(len(CandidatePos)):
            allxtemp.append(BeginX+CandidatePos[i_t])
            allytemp.append(BeginY+i)
            #edges[BeginX+CandidatePos[i_t], :] = 255
            #cv2.imwrite('F:\\MyCode\\LensSegmentation\\result\\imageResult.png', edges)

        #final = Get50percentAverage(allyLeftRight)
        #allx and ally is the centre of the rect
        Linescores = ScoreTopDown(Binaryedgemap, allxtemp, allytemp)
        maxscores = Linescores.index(np.max   (Linescores))
        finalx = allxtemp[maxscores] + boundingBoxSize[1]/2
        finaly = allytemp[maxscores]

        #SlidingWindowPos = [BeginX + j - boundingBoxSize[1] / 2, BeginY + i - boundingBoxSize[0] / 2,
        #                    boundingBoxSize[0], boundingBoxSize[1]]
        #SlidingWindowPos = [finalx, finaly, boundingBoxSize[0], boundingBoxSize[1]]
        #NewMap = DrawSlidingWindow(SlidingWindowPos, edges)
        # NewMap = edges
        # NewMap[int(finalx)+boundingBoxSize[1]/2, :] = 255
        # NewMap[:, imshape[1] / 2] = 255
        # cv2.imwrite('F:\\MyCode\\LensSegmentation\\result\\imageResult2.png', NewMap)

        allx.append(int(finalx))
        ally.append(int(finaly))

    return allx,ally


def process_csv(data):
    for key in data.keys():
        data[key] = data[key].astype(np.float32)
    # data = data.sort_values(by='0')
    r_sum = data.iloc[:, 1:].apply(lambda x: x.sum(), axis=1).values.astype(np.bool)
    data = data.iloc[r_sum, :]
    return data

def get_pixel_label(img):
    p_label = np.zeros(img.shape[:3]).astype(np.uint8)
    rows,cols,dims=img.shape
    for row in range(rows):
        for col in range(cols):
            if (img[row,col]==[255,255,0]).all():
                p_label[row,col]=1
            elif(img[row,col]==[255,0,0]).all():
                p_label[row,col]=2
            elif(img[row,col]==[0,0,255]).all():
                p_label[row,col]=3
    if p_label.max()!=3:
        print('fail %s'%(p_label))
    return p_label


def my_ployfit(x,y,num,start,end,ratio=2):
    z1 = np.polyfit(x, y, ratio)
    p1 = np.poly1d(z1)
    plt_x = np.linspace(start, end, num)
    plt_y = np.polyval(p1, plt_x)
    return plt_x,plt_y

def new_ployfit(x,y,num,start,end,ratio=2):
    z1 = np.polyfit(x, y, ratio)
    p1 = np.poly1d(z1)
    plt_x = np.linspace(start, end, num)
    plt_y = np.polyval(p1, plt_x)
    return z1,plt_x,plt_y

def compute_intersection(z_1, z_2):
    a, b, c = z_1 - z_2
    if a < 1e-8:
        return 0, 0
    deta = b ** 2 - (4 * a * c)
    if deta < 1e-8:
        return -b / (2 * a), -b / (2 * a)
    elif deta > 0:
        return (-b - np.sqrt(deta)) / (2 * a), (-b + np.sqrt(deta)) / (2 * a)
    elif deta < 0:
        return 0, 0

def process_x_y(csv_data,idx,img_shape, start,end,flag=False):
    z1 = np.array([0.0, 0.0, 0.0])
    x = csv_data['0'].values * img_shape[1] / 16.0
    y = csv_data[str(idx + 1)].values * img_shape[0] / 14.0
    mask = map(lambda i: not np.isnan(i), y)
    y = y[mask]
    x = x[mask]
    if flag:
        z1, x, y = new_ployfit(x, y, num=img_shape[1] - 1, start=start, end=end)
    else:
        x, y = my_ployfit(x, y, num=img_shape[1] - 1, start=start, end=end)
    return z1,x,y

def not_csv_process(x,y,img_shape, start,end,flag=False):
    z1 = np.array([0.0, 0.0, 0.0])
    if flag:
        z1, x, y = new_ployfit(x, y, num=img_shape[1] - 1, start=start, end=end)
    else:
        x, y = my_ployfit(x, y, num=img_shape[1] - 1, start=start, end=end)
    return z1,x,y

def split_dataset(root_path, split_idx=0.7):
    img_list = os.listdir(root_path)
    random.shuffle(img_list)
    tmp_list = []
    for img in img_list:
        data = img.split('_')
        # idx = data[0] + '_' + data[1]
        idx = data[0]
        if not idx in tmp_list:
            tmp_list.append(idx)
    random.shuffle(tmp_list)
    train_num = int(len(tmp_list) * split_idx)
    val_tmp_list = tmp_list[train_num:]
    train_list = []
    val_list = []
    for img in img_list:
        data = img.split('_')
        # idx = data[0] + '_' + data[1]
        idx = data[0]
        if idx in val_tmp_list:
            val_list.append(img)
        else:
            train_list.append(img)
    return train_list, val_list

def split_dataset_1(root_path, tain_percent=0.6, test_percent=0.2, val_percent=0.2):
    """
    To split dataset into train, test, val.
    Author: Shihao Zhang
    Data: 2018/12/17
    :param root_path:
    :param tain_percent:
    :param test_percent:
    :param val_percent:
    :return:
    """
    # img_list=[]
    # with open(root_path) as fin:
    #     for line in fin:
    #         img_list.append(line.strip('\n'))
    #
    img_list = os.listdir(root_path)

    random.shuffle(img_list)
    tmp_list = []
    for img in img_list:
        data = img.split('_')
        idx = data[0] + '_' + data[1]
        #     idx = data[0]
        if not idx in tmp_list:
            tmp_list.append(idx)
    random.shuffle(tmp_list)
    train_num = int(len(tmp_list) * tain_percent)
    test_num = int(len(tmp_list) * test_percent)
    val_num = int(len(tmp_list) * val_percent)
    train_tmp_list = tmp_list[:train_num]
    val_tmp_list = tmp_list[train_num+test_num:]

    train_list = []
    test_list = []
    val_list = []
    for img in img_list:
        data = img.split('_')
        idx = data[0] + '_' + data[1]
        # idx = data[0]
        if idx in val_tmp_list:
            val_list.append(img)
        elif idx in train_tmp_list:
            train_list.append(img)
        else:
            test_list.append(img)
    return train_list, test_list, val_list


def distance_map(data_path, distance_path):
    img_list = os.listdir(data_path)
    for idx, img_name in enumerate(img_list):
        path = os.path.join(data_path, img_name)
        img = cv2.imread(path, 0)
        # img = cv2.resize(img,(size,size))
        tmp_img = []
        for i in range(1, 4, 1):
            up_img = np.zeros_like(img)
            down_img = np.zeros_like(img)
            up_img[img != i] = 0
            up_img[img == i] = 1
            down_img[img != i] = 1
            down_img[img == i] = 0
            up = cv2.distanceTransform(up_img, cv2.DIST_L2, 5)
            down = cv2.distanceTransform(down_img, cv2.DIST_L2, 5)
            newImage = up + down
            tmp_img.append(newImage)
        tmp_img = np.stack(tmp_img)
        save_name = os.path.join(distance_path, img_name[:-4])
        np.save(save_name, tmp_img)


def distance_map_new(data_path, distance_path):
    img_list = os.listdir(data_path)
    for idx, img_name in enumerate(img_list):
        path = os.path.join(data_path, img_name)
        img = cv2.imread(path, 0)
        # img = cv2.resize(img,(size,size))
        tmp_img = []
        for i in range(1, 4, 1):
            up_img = np.zeros_like(img)
            down_img = np.zeros_like(img)
            up_img[img != i] = 0
            up_img[img == i] = 1

            boundary1 = find_boundaries(up_img, mode='inner').astype(np.uint8)
            up_img = up_img - boundary1

            down_img[img != i] = 1
            down_img[img == i] = 0

            # boundary2 = find_boundaries(down_img, mode='inner').astype(np.uint8)
            # down_img = down_img - boundary2

            up = cv2.distanceTransform(up_img, cv2.DIST_L2, 5)
            down = cv2.distanceTransform(down_img, cv2.DIST_L2, 5)
            newImage = up + down
            tmp_img.append(newImage)
        tmp_img = np.stack(tmp_img)
        save_name = os.path.join(distance_path, img_name[:-4])
        np.save(save_name, tmp_img)
