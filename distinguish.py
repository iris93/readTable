# coding=utf8

import os
import cv2
import numpy as np
import classifier
import pandas as pd


class Distinguish(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.is_A4 = False
        self.testdir = os.path.join(os.path.dirname(__file__),"test/")
        self.picedir = os.path.join(os.path.dirname(__file__),"pice/")
        self.outdir = os.path.join(os.path.dirname(__file__),"output/")
        self.table_setting  = {"min_area" : 4000, "max_area" : 7000, "xoffset" : 2, "yoffset" : 2}

        # this dir just for test
        if not os.path.exists(self.testdir):
            os.mkdir(self.testdir)

    def imageRead(self):
        image = cv2.imread(self.file_path)
        # 边缘检测算法需要灰度图作为输入
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(self.testdir + "gray.jpg", gray_image)
        copy_image = gray_image.copy()
        copy_image = cv2.medianBlur(copy_image, 3)
        cv2.imwrite(self.testdir + "blur.jpg", copy_image)

        # 采用领域内自适应预支函数处理，为的就是准确的找出想要的矩形边框
        # 第一个参数是原始图像
        # 第二个参数像素值上限
        # 第三个参数自适应方法Adaptive Method:
        #   — cv2.ADAPTIVE_THRESH_MEAN_C ：领域内均值
        #   —cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：领域内像素点加权和，权 重为一个高斯窗口
        # 第四个参数值的赋值方法：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV
        # 第五个参数Block size:规定领域大小（一个正方形的领域）
        # 第六个参数常数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值 就是求得领域内均值或者加权值, 这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值。
        adaptive_binary = cv2.adaptiveThreshold(copy_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # adaptive_binary = cv2.Threshold(gab, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # 灰度图二值化， 二值化的图像用于裁剪图片
        ret, thresh = cv2.threshold(adaptive_binary, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(self.testdir + "thresh.jpg", thresh)
        erode = cv2.erode(thresh, None, iterations = 1)
        cv2.imwrite(self.testdir + "erode.jpg", erode)
        # 检测图像中所有的闭合曲线
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 画出包含闭合曲线的图像用于调试
        # 画出轮廓，-1,表示所有轮廓，画笔颜色为(0, 255, 0)，即Green，粗细为3
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
        # cv2.imwrite(self.testdir + "contours.jpg", image)

        point_list = self.findRect(contours, image)
        # print len(point_list)
        # print point_list



        self.reset()

        # 裁剪图像，每个单元格切为一幅图像
        pice_list = self.cutImage(point_list, image)
        # print len(pice_list)
        # 对于裁剪的图像进行识别
        self.distinguish(pice_list)
        # 将识别的数据拼接起来
        j=0
        result = self.stitchData(point_list)
        result = pd.DataFrame(result)
        print result
        result.to_excel('final-result/result.xls')


    def reset(self):
        os.system("rm pice/*")
        os.system("rm output/*")
        os.system("rm template/*")

    def findRect(self, contours, image):
        file_name = "rects.jpg"
        # os.remove(self.testdir + file_name)
        point_list = []
        # area_list = []
        for cont in contours:
            # 计算矩形框的面积，采用面积大小来过滤不需要识别的矩形框
            area = cv2.contourArea(cont)
            # area_list.append(area)
            if area < self.table_setting["min_area"] or area > self.table_setting["max_area"]:
                continue
            # 找到图像中的所有的矩形框
            x, y, w, h = cv2.boundingRect(cont)
            point = [x, y, w, h]

            # 测试使用，在图像上画出矩形框，用于调整参数
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            point_list.append(point)
        # area_list.sort()
        # print area_list
        # 保存之前在图像上的操作
        cv2.imwrite(self.testdir + file_name, image)
        return point_list

    def createList(self, point_list):
        xlist = []
        ylist = []
        wlist = []
        hlist = []
        for point in point_list:
            xlist.append(point[0])
            ylist.append(point[1])
            wlist.append(point[2])
            hlist.append(point[3])
        xlist.sort()
        ylist.sort()
        np.array(wlist)
        np.array(hlist)
        wmean_val = np.mean(wlist)
        hmean_val = np.mean(hlist)
        return xlist, ylist, wmean_val, hmean_val

    def splitImage(self,imgSrc):
        grayImg = cv2.cvtColor(imgSrc,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grayImg, 75, 255, cv2.THRESH_BINARY_INV)
        ret,blackImg = cv2.threshold(grayImg, 95, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.dilate(closed, None, iterations = 3)
        contours, hierarchy = cv2.findContours(closed,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print len(contours)
        area = []
        point =[]
        areaMax = 0
        indexMax = 0
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
            x, y, w, h = cv2.boundingRect(contours[i])
            point.append([x,y,w,h])
            if areaMax<area[i]:
                areaMax = area[i]
                indexMax = i
        x, y, w, h = point[indexMax]
        newImg = blackImg[y:y+h,x:x+w]
        newImg = cv2.erode(newImg, None, iterations = 1)
        return newImg

    def cutImage(self, point_list, image):
        xlist, ylist, wmean_val, hmean_val = self.createList(point_list)

        # 拿到x, y的梯度，也就是对x, y进行分类，将分类结果组装起来就获得了表格
        xthresh = []
        xladder = self.getLadder(xlist)
        for i in xrange(len(xlist)-1):
            if xlist[i+1]-xlist[i]>xladder:
                xthresh.append((xlist[i]+xlist[i+1])/2)
        ythresh = []
        yladder = self.getLadder(ylist)
        for i in xrange(len(ylist)-1):
            if ylist[i+1]-ylist[i]>yladder:
                ythresh.append((ylist[i]+ylist[i+1])/2)
        # print len(xthresh), len(ythresh)
        pice_list = []
        for point in point_list:
            # 遍历所有的矩形框， 获取矩形框的坐标(x1y1)表示左下角开始的第一列第一行
            # print point[0]
            xlabel = self.getLabel(point[0], xthresh)
            ylabel = self.getLabel(point[1], ythresh)
            ythresh.sort()
            xthresh.sort()
            if not ylabel and ylabel <> 0:
                continue
            # print xlabel, ylabel
            # 在原图像上截取出每个矩形框用于识别
            pice = image[(point[1] + self.table_setting["yoffset"]) : (point[1] + point[3] - self.table_setting["yoffset"]),\
             (point[0] + self.table_setting["xoffset"]) : (point[0] + point[2] - self.table_setting["xoffset"])]
            picename = self.picedir + 'x' + str(xlabel) + 'y' + str(ylabel) + '.jpg'
            pice = self.splitImage(pice)
            cv2.imwrite(picename ,pice)
            pice_list.append(picename)
        return pice_list

    #　对ｘｙ进行分类，分成的结果应该是ｘ３５, ｙ１６对应着３５列１６行
    def getLadder(self,eList):
        retList = []
        for i in xrange(len(eList)-1):
            retList.append(eList[i+1]-eList[i])
            retList.sort()
            for i in xrange(len(retList) - 1):
                if retList[i] >0 and retList[i+1] > 5*retList[i]:
                    return (retList[i+1]+retList[i])/2

    # 拿到矩形框在图像上的坐标，左下角开始x1y1代表第一列第一行
    def getLabel(self,e,eThresh):
        if e < min(eThresh):
            return 1
        elif e > max(eThresh):
            return len(eThresh) + 1
        else:
            for i in xrange(1,len(eThresh)):
                if e > eThresh[i-1] and e < eThresh[i]:
                    return i+1

    def sort(self,point_list):
        for i in range(len(point_list)-1,0,-1):
            for j in range(i):
                if point_list[j][0]>point_list[j+1][0]:
                    temp = point_list[j]
                    point_list[j] = point_list[j + 1]
                    point_list[j + 1] = temp
        return point_list

    def splitNumber(self,pice_name):
        imgSrc = cv2.imread(pice_name,1);
        grayImg = cv2.cvtColor(imgSrc,cv2.COLOR_BGR2GRAY)
        imgCopy = imgSrc
        pice = pice_name.split("/")[-1][:-4]
        ret, thresh = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        point_list = []
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            point_list.append([x,y,w,h])
        # print point_list
        point_list = self.sort(point_list)
        # print point_list
        numbers = []
        for i in range(len(point_list)):
            x, y, w, h = point_list[i]
            if h >13:
                newImg = imgCopy[y:y+h,x:x+w]
                cv2.imwrite("template/"+str(i)+"-"+str(pice)+".jpg",newImg)
                numbers.append(newImg)
        return numbers

    def write(self,result,output):
        f = open(output,'w')
        for item in result:
            f.write(str(item))
        f.close()

    # 识别函数，采用多进程(进程池)
    def distinguish(self, pice_list):
        for pice in pice_list:
            numbers = self.splitNumber(pice)
            img_input = classifier.loadTestImg(numbers)
            result = classifier.classifier(img_input)
            # print result
            outfile = self.outdir + pice.split("/")[-1][:-4]+".txt"
            self.write(result,outfile)

    def getThresh(self, elist):
        ethresh = []
        eladder = self.getLadder(elist)
        for i in xrange(len(elist)-1):
            if elist[i + 1] - elist[i] > eladder:
                ethresh.append((elist[i] + elist[i + 1]) / 2)
        return ethresh

    # 数据拼接函数，将识别结果按照坐标位置拼接
    def stitchData(self, point_list):
        xlist, ylist, wmean_val, hmean_val = self.createList(point_list)
        # 拿到x, y的梯度，也就是对x, y进行分类，将分类结果组装起来就获得了表格
        xthresh = self.getThresh(xlist)
        ythresh = self.getThresh(ylist)
        xlabel_list = []
        ylabel_list = []
        # 这里是为了组建图像上的坐标系
        for point in point_list:
            xlabel = self.getLabel(point[0], xthresh)
            ylabel = self.getLabel(point[1], ythresh)
            xlabel_list.append(xlabel)
            ylabel_list.append(ylabel)
        row_list = []
        # 选出不同的坐标比如x上的(1,2,3)y(1,2,3)
        xlabel_list = list(set(xlabel_list))
        ylabel_list = list(set(ylabel_list))
        xlabel_tmp = []
        ylabel_tmp = []
        # 因为实际操作的时候出现label出现None的情况
        for xlabel in xlabel_list:
            if xlabel:
                xlabel_tmp.append(xlabel)
        for ylabel in ylabel_list:
            if ylabel:
                ylabel_tmp.append(ylabel)
        xlabel_list = xlabel_tmp
        ylabel_list = ylabel_tmp
        xlen = len(set(xlabel_list))
        ylen = len(set(ylabel_list))
        # 使用两个列表的乘积组成图像坐标系,进行数据的拼接
        for y in xrange(1,ylen+1):
            col_list = []
            for x in xrange(1,xlen+1):
                picename = self.outdir + 'x' + str(x) + 'y' + str(y) + '.txt'
                try:
                    fp = open(picename, "rb")
                    content = fp.readline().strip()
                    fp.close()
                except Exception, e:
                    content = "null"
                col_list.append(content)
            row_list.append(col_list)
        return row_list
