import cv2
import numpy as np
import os.path
import pandas as pd
import cPickle as pkl



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
def ScoreLeftRight(BinaryMap_,edgeMap,Lines):
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
    Linescores = ScoreLeftRight(BinaryMap_,edges,allyLeftRight)
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
    Linescores = ScoreLeftRight(BinaryMap_,edges, allyLeftRight)
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
    return p_label


def my_ployfit(x,y,num,start,end,ratio=2):
    z1 = np.polyfit(x, y, ratio)
    p1 = np.poly1d(z1)
    plt_x = np.linspace(start, end, num)
    plt_y = np.polyval(p1, plt_x)
    return plt_x,plt_y

def process_x_y(csv_data,idx,img,start,end,flag=False):
    x = csv_data['0'].values * img.shape[1] / 16.0
    y = csv_data[str(idx + 1)].values * img.shape[0] / 14.0
    mask = map(lambda i: not np.isnan(i), y)
    y = y[mask]
    x = x[mask]
    if flag:
        start = x[5]
        end = x[-5:-4]
    x, y = my_ployfit(x, y, num=img.shape[1] - 1, start=start, end=end)
    return x,y



def main():
    all_img_dir = './data/eyes/'
    train_label_path = './data/train_label'
    visual_data_path = './data/visual_data'
    train_data_path = './data/train_data'
    three_color = [(255,255,0), (255,0,0), (0,0,255)]  # yellow, red, blue
    # three_color = ['#FFFF00', '#FF0000', '#0000FF']
    if not os.path.isdir(train_label_path):
        os.mkdir(train_data_path)
        os.mkdir(train_label_path)
        os.mkdir(visual_data_path)

    img_infor = {}
    patient_idxs = os.listdir(all_img_dir)
    for path in patient_idxs:
        cur_path = os.path.join(all_img_dir, path)
        img_list = os.listdir(cur_path)

        csv_path = [x for x in img_list if x.endswith('.csv')]
        if not csv_path:
            continue
        csv_path = os.path.join(cur_path, csv_path[0])
        img_list = [x for x in img_list if x.endswith('.jpg')]
        img_list = np.sort(img_list)

        ## read and process the csv datas
        csv_data = pd.read_csv(csv_path)
        # Lens = pd.concat([csv_data[:800], csv_data[4015:4815]])
        # Cortex = pd.concat([csv_data[803:1603], csv_data[3212:4012]])
        # Nucleus = pd.concat([csv_data[1606:2406], csv_data[2409:3209]])
        # Lens = process_csv(Lens)
        # Cortex = process_csv(Cortex)
        # Nucleus = process_csv(Nucleus)

        Lens_front = csv_data[:800]
        Lens_back = csv_data[4015:4815]
        Lens1 = process_csv(Lens_front)
        Lens2 = process_csv(Lens_back)
        Lens2 = Lens2.sort_index(ascending=False)
        Lens = pd.concat([Lens1, Lens2])

        Cortex_front = csv_data[803:1603]
        Cortex_back = csv_data[3212:4012]
        Cortex1 = process_csv(Cortex_front)
        Cortex2 = process_csv(Cortex_back)
        Cortex2 = Cortex2.sort_index(ascending=False)
        Cortex = pd.concat([Cortex1, Cortex2])

        Nucleus_front = csv_data[1606:2406]
        Nucleus_back = csv_data[2409:3209]
        Nucleus1 = process_csv(Nucleus_front)
        Nucleus2 = process_csv(Nucleus_back)
        Nucleus2 = Nucleus2.sort_index(ascending=False)
        Nucleus = pd.concat([Nucleus1, Nucleus2])
        print 'success %s' % path

        ## read the images
        for idx, img in enumerate(img_list):
            ori_img_path = os.path.join(cur_path, img)
            patient_idx, _, _, eye_idx, _, _, img_name = img.split('_')
            patient_idx = patient_idx.split('(')[0]
            img_name = patient_idx + '_' + eye_idx + '_' + img_name.split('.')[0] + '.png'
            img_infor[img_name] = {}
            img = cv2.imread(ori_img_path, 0)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            edges = cv2.Canny(img, 35, 110)
            #edges = cv2.Canny(img, 90, 150)
            # Sliding Window
            SlidingWindowSize = (20,40)
            #SlidingWindowSize = (20,50)
            #Get Binary Map
            BinaryMap_ = TransformToBinaryImage(edges)
            imshape = edges.shape

            img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            #topx, topy = DecideBoundaryTopDown(BinaryMap_, SlidingWindowSize, 580, 1462)
            ally = DecideBoundaryLeftRight(BinaryMap_, SlidingWindowSize)
            #Draw The boundary
            BeginY = imshape[1] / 3
            newImage = img2[BeginY:,ally[0]:ally[1]]
            img_infor[img_name]['position'] = [ally[0],ally[1],BeginY] # left,right,top
            img_infor[img_name]['size'] = newImage.shape

            BW_img = np.zeros(newImage.shape[:2], dtype=np.uint8)
            BW_img1 = BW_img.copy()
            data_savename = './data/train_data/' + img_name
            visual_name = './data/visual_data/' + img_name
            label_savename = './data/train_label/' + img_name
            cv2.imwrite(data_savename, newImage)
            # cv2.imshow(savename,newImage)

            for color_id,csv_data in enumerate([[Lens1,Lens2],[Cortex1,Cortex2], [Nucleus1,Nucleus2]]):
                front,back = csv_data

                front_x, front_y = process_x_y(front, idx,img, start=ally[0], end=ally[1], flag=True)
                back_x, back_y = process_x_y(back, idx,img, start=ally[1], end=ally[0], flag=True)
                x = np.stack([front_x, back_x]).reshape([-1])
                y = np.stack([front_y, back_y]).reshape([-1])
                # x = x - ally[0]
                # y = y - BeginY
                new_index = zip(x,y)
                new_index=np.array(new_index,np.int32).reshape([-1,1,2])

                # save the 2 class
                # cv2.fillPoly(BW_img, [new_index], 1)
                # cv2.fillPoly(BW_img1, [new_index], 255)

                # cv2.polylines(img2,[np.int32(new_index)],False,(255,0,0),8)
                cv2.fillPoly(img2,[new_index],three_color[color_id])
                x = x - ally[0]
                y = y - BeginY
                new_index = zip(x, y)
                new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
                cv2.fillPoly(BW_img, [new_index], color_id+1)

                # cv2.polylines(BW_img,[np.int32(new_index)],False,0,4)
                # cv2.fillPoly(img2,[new_index],three_color[color_id])
                # cv2.fillPoly(newImage, [new_index], three_color[color_id])

            cv2.imwrite(label_savename, BW_img)
            # cv2.imwrite(visual_name, BW_img1)
            cv2.imwrite(visual_name, img2)
            # pixel_label = get_pixel_label(newImage)
            # cv2.imwrite(label_savename, pixel_label)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)

            # for color_id, csv_data in enumerate([Lens, Cortex, Nucleus]):
            #     x = csv_data['0'].values * img.shape[1] / 16.0
            #     y = csv_data[str(idx + 1)].values * img.shape[0] / 14.0
            #     mask = map(lambda i: not np.isnan(i), y)
            #     y = y[mask]
            #     x = x[mask]
            #     x = x - ally[0]
            #     y = y - BeginY
            #     new_index = zip(x, y)
            #     new_index = np.array(new_index, np.int32).reshape([-1, 1, 2])
            #
            #     cv2.polylines(BW_img,[np.int32(new_index)],False,255,8)
            #     # cv2.fillPoly(img2,[new_index],three_color[color_id])
            #     # cv2.fillPoly(newImage, [new_index], three_color[color_id])
            # cv2.imwrite(visual_name, BW_img)
            # # cv2.imwrite(visual_name, img2)
            # # pixel_label = get_pixel_label(newImage)
            # # cv2.imwrite(label_savename, pixel_label)
            # # cv2.imshow("img", img)
            # # cv2.waitKey(0)

            # plt.plot(x, y,three_color[color_id])
            # cv2.line(newImage, new_index[0], new_index[100], green, 10)
            # cv2.imshow("img",newImage)
            # cv2.waitKey(0)
            # cv2.imwrite(savename, newImage)

            # ## np.polyfit
            # for color_id,csv_data in enumerate([[Lens1,Lens2],[Cortex1,Cortex2], [Nucleus1,Nucleus2]]):
            #     front,back = csv_data
            #     if color_id<2:
            #         front_x,front_y = process_x_y(front, idx, start=ally[0], end=ally[1])
            #         back_x, back_y = process_x_y(back, idx, start=ally[1], end=ally[0])
            #     else:
            #         front_x, front_y = process_x_y(front, idx, start=ally[0], end=ally[1], flag=True)
            #         back_x, back_y = process_x_y(back, idx, start=ally[1], end=ally[0], flag=True)
            #     x = np.stack([front_x, back_x]).reshape([-1])
            #     y = np.stack([front_y, back_y]).reshape([-1])
            #     x = x - ally[0]
            #     y = y - BeginY
            #     new_index = zip(x,y)
            #     new_index=np.array(new_index,np.int32).reshape([-1,1,2])
            #
            #     # cv2.polylines(img2,[np.int32(new_index)],False,(255,0,0),8)
            #     # cv2.fillPoly(img2,[new_index],three_color[color_id])
            #     cv2.fillPoly(newImage, [new_index], three_color[color_id])
            # cv2.imwrite(visual_name, newImage)
            # # cv2.imwrite(visual_name, img2)
            # pixel_label = get_pixel_label(newImage)
            # cv2.imwrite(label_savename, pixel_label)
            # # cv2.imshow("img", img)
            # # cv2.waitKey(0)

    with open('./train.pkl','w') as f:
        pkl.dump(img_infor,f)

if __name__ == "__main__":
    main()