import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.image as img
import matplotlib.image as mpimg
import datetime
import calendar
import time 


class Car_Object():

    def __init__(self, car_id):
        
        self.car_id = car_id
        self.final_number_plate = []
        self.alphabet = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L'
            ,'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

        # 進入時間
        self.entry_Year = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[0].split('-')[0])
        self.entry_Month = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[0].split('-')[1])
        self.entry_Day = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[0].split('-')[2])
        self.entry_Hour = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[1].split(':')[0])
        self.entry_Minute = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[1].split(':')[1])
        self.entry_Second = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[1].split(':')[2])
        
        self.Month_num = 12
        self.Hour_num = 24
        self.Minute_num = 60
        
        self.an_Hour_Pay = 30
        self.amounts_payable = 0
        
        self.is_paid = 0

    
    def clip_image(self, gray):

        at_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 333, 1)
        temp_img = at_img.copy()

        # 因為照片可能會傾斜，所以左右兩邊會有白邊
        # 切割左右
        white = []
        black = []

        height = temp_img.shape[0]
        width = temp_img.shape[1]

        white_max = 0
        black_max = 0

        for i in range(width):
            s = 0  
            t = 0  
            for j in range(height):
                if temp_img[j][i] == 255:
                    s += 1
                if temp_img[j][i] == 0:
                    t += 1
            white_max = max(white_max, s)
            black_max = max(black_max, t)
            white.append(s)
            black.append(t)
        # 左
        b_left = 0
        for b_left in range(len(black)):
            if black[b_left] > white_max * 0.75:
                break
        # 右
        b_right = 0
        for b_right in range(len(black) - 1, 0, -1):
            if black[b_right] > white_max * 0.75:
                break

        img_copy = temp_img.copy()
        width = img_copy.shape[1]
        height = img_copy.shape[0]

        left_top = (b_left, 0)
        right_bottom = (b_right, height)

        clipped_img = img_copy[left_top[1]:right_bottom[1], left_top[0]: right_bottom[0]]
        temp_img = clipped_img.copy()


        # 切割上下(螺絲)
        white = []
        black = []

        height = temp_img.shape[0]
        width = temp_img.shape[1]

        white_max = 0
        black_max = 0

        for i in range(height):
            s = 0  
            t = 0  
            for j in range(width):
                if temp_img[i][j] == 255:
                    s += 1
                if temp_img[i][j] == 0:
                    t += 1
            white_max = max(white_max, s)
            black_max = max(black_max, t)
            white.append(s)
            black.append(t)

        # 上
        screw_start_top = 0
        for screw_start_top in range(30, len(white)):
            # 螺絲開始
            if white[screw_start_top] > black_max * 0.05:
                break
        screw_end_top = 0
        for screw_end_top in range(screw_start_top, len(white)):
            # 螺絲結束
            if white[screw_end_top] < black_max * 0.03:
                break
                
        # 下
        screw_start_down = 0
        for screw_start_down in range(len(white) - 25, 0, -1):
            # 螺絲開始
            if white[screw_start_down] > black_max * 0.1:
                break
        screw_end_down = 0
        for screw_end_down in range(screw_start_down, 0, -1):
            # 螺絲結束
            if white[screw_end_down] < black_max * 0.1:
                break

        img_copy = temp_img.copy()
        width = img_copy.shape[1]
        height = img_copy.shape[0]

        left_top = (0, screw_end_top)
        right_bottom = (width, screw_end_down - 10)

        clipped_img = img_copy[left_top[1]:right_bottom[1], left_top[0]: right_bottom[0]]

        return clipped_img
    
    def find_end(self, start, width, black, black_max):
        
        end = start + 1
        for m in range(start + 1, width - 1):
            # 如果columns幾乎是黑的，就認定結束(邊界)
            if black[m] >= black_max * 0.95:  # 可調整參數
                end = m
                break
        return end
    
    def split_image(self, final_img):

        # 在每個columns裡白色的數量
        white = []
        # 在每個columns裡黑色的數量
        black = []

        height = final_img.shape[0]
        width = final_img.shape[1]

        white_max = 0
        black_max = 0
        # 每個columns裡黑白的總數
        for i in range(width):
            white_num = 0  
            black_num = 0  
            for j in range(height):
                if final_img[j][i] == 255:
                    white_num += 1
                if final_img[j][i] == 0:
                    black_num += 1
            white_max = max(white_max, white_num)
            black_max = max(black_max, black_num)
            white.append(white_num)
            black.append(black_num)
            
        temp = 1

        # 開始、結束位置
        start = 2
        end = 3

        # 被分割的數量
        num = 0
        
        # 創建新的資料夾
        os.mkdir("C:/Users/user/Downloads/ANPR/entry/display_image/Car" + str(self.car_id))
        os.mkdir("C:/Users/user/Downloads/ANPR/entry//train_image/Car" + str(self.car_id))

        # train/display
        display_output_path = "C:/Users/user/Downloads/ANPR/entry/display_image/Car" + str(self.car_id)
        train_output_path = "C:/Users/user/Downloads/ANPR/entry/train_image/Car" + str(self.car_id)


        # 從左至右
        while temp < width - 2:
            temp += 1

            # 找到白邊即開始
            if white[temp] > 0.05 * white_max:
                start = temp
                # 找結束位置
                end = self.find_end(start, width, black, black_max)
                # 下一個開始點
                temp = end

                # 確保是車牌號碼(太小為雜質)
                if end - start > 20:
                    # 分割
                    if (end + 2) > width:
                        result = final_img[1:height, start - 1:end]
                
                    if (start - 2) < 0:
                        result = final_img[1:height, start :end + 2]

                    result = final_img[1:height, start - 2:end + 2]

                    # 儲存分割結果到資料夾
                    cv2.imwrite(display_output_path+ '/' + str(num) + '.jpg', result)

                    # resize (因為訓練模組的input需要(784,)的大小，所以先存為(28,28))
                    result = cv2.resize(result, (28, 28), interpolation=cv2.INTER_CUBIC)

                    cv2.imwrite(train_output_path + '/' + str(num) + '.jpg', result)

                    num += 1
        self.lic_num = num
          
    def process_image(self):

        # 讀取照片
        img = cv2.imread('entry/origin_image/image_' + str(self.car_id) + '.jpg')
        self.origin_img = img
        # 轉灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 統一大小
        gray = cv2.resize(gray, (1000, 500))
        
        final_img = self.clip_image(gray)
        
        self.split_image(final_img)
        
    def get_image(self):

        for i in range(0, self.lic_num):
            # 取得照片
            globals()['img_' + str(i)] = img.imread('entry/train_image/Car' + str(self.car_id) + '/' + str(i) + '.jpg')
            # reshape to 28x28
            globals()['img_' + str(i)] = globals()['img_' + str(i)].reshape([28,28])
            # 左右反轉
            globals()['img_' + str(i)] = np.fliplr(globals()['img_' + str(i)])
            # 轉90度
            globals()['img_' + str(i)] = np.rot90(globals()['img_' + str(i)])
            # reshape to 784x1
            globals()['img_' + str(i)] = globals()['img_' + str(i)].reshape([784,])
        
    def find_hyphen(self, img, number_plate, hyphen): # '-'

        two_list = []
        # 把照片兩行兩行分別加總，找到差距最大的值即為'-'符號
        for i in range(10, 18):
            total = np.mean(img.reshape([28, 28])[:, i])
            total += np.mean(img.reshape([28, 28])[:, i + 1])
            two_list.append(total)
            
        two = np.array(two_list)
        sort_index = np.argsort(two)
        
        max_pixel = two[sort_index[-1]]
        min_pixel = two[sort_index[0]]

        # is hyphen
        if (max_pixel - min_pixel) > 400:
            hyphen.append(img)
            return 1, number_plate, hyphen
        # not hyphen
        else:
            number_plate.append(img)
            return 0, number_plate, hyphen
        
    def input_data(self, parking_df):

        # list變成字串
        self.number_plate_str = ''
        for i in self.final_number_plate:
            self.number_plate_str += i
            
        parking_df.loc[self.car_id - 1] = [self.car_id, self.number_plate_str, self.entry_Year, self.entry_Month, self.entry_Day,
                                  self.entry_Hour, self.entry_Minute, self.entry_Second, np.nan, np.nan
                                , np.nan, np.nan, np.nan, np.nan, np.nan, 0]
        
        return parking_df

    def identify_number_plate(self, parking_df):

        # load model
        rfc_model = joblib.load('rfc_model')
        
        self.get_image()
        
        number_plate = []
        hyphen = []
        hyphen_id = 0
        # find hyphen
        for i in range(self.lic_num):
            is_hyphen, number_plate, hyphen = self.find_hyphen(globals()['img_' + str(i)], number_plate, hyphen)
            if is_hyphen == 1:
                hyphen_id = i
                
        number_plate_arr = np.array(number_plate ,dtype='int64')
        
        # 預測車牌
        predictions = rfc_model.predict(number_plate_arr)

        # append to final number plate
        count = 0
        for index in predictions:
            count += 1
            if count == hyphen_id:
                self.final_number_plate.append(self.alphabet[index])
                self.final_number_plate.append('-')
                
            else:
                self.final_number_plate.append(self.alphabet[index])
        # input data        
        parking_df = self.input_data(parking_df)

        return self.final_number_plate, parking_df 
    


class Leave_sys():

    def __init__(self, leave_Id):
        self.leave_Id = leave_Id
        self.alphabet = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L'
            ,'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.final_number_plate = []
        

    def leave_cam(self):

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        cap.set(3, 1240)
        cap.set(4, 640)

        num = 0

        while True:
            success, img = cap.read()
            if success:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                break
            
            # 取得輪廓
            gray = cv2.GaussianBlur(gray, (15, 15), 0)
            gray = cv2.bilateralFilter(gray, 40, 40, 500)
            edged = cv2.Canny(gray, 12, 35)
            cv2.imshow('Edge', edged)

            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                cv2.imshow('Video', img)
                x, y, w, h = cv2.boundingRect(contour)
                if ((x + w) > 950) and ((y + h) > 300)  and (w > 850) and (h > 300):
                    num += 1
                # 重複多次一點確保為車牌
                if num > 10:
                    clipped_img = img[y : y + h, x : x + w]
                    cv2.imwrite('leave/origin_image/image_'+ str(self.leave_Id) + '.jpg', clipped_img)
                    break
            if num > 10:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def clip_image(self, gray):

        at_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 333, 1)
        temp_img = at_img.copy()

        # 因為照片可能會傾斜，所以左右兩邊會有白邊
        # 切割左右
        white = []
        black = []

        height = temp_img.shape[0]
        width = temp_img.shape[1]

        white_max = 0
        black_max = 0

        for i in range(width):
            s = 0  
            t = 0  
            for j in range(height):
                if temp_img[j][i] == 255:
                    s += 1
                if temp_img[j][i] == 0:
                    t += 1
            white_max = max(white_max, s)
            black_max = max(black_max, t)
            white.append(s)
            black.append(t)
        # 左
        b_left = 0
        for b_left in range(len(black)):
            if black[b_left] > white_max * 0.75:
                break
        # 右
        b_right = 0
        for b_right in range(len(black) - 1, 0, -1):
            if black[b_right] > white_max * 0.75:
                break
        img_copy = temp_img.copy()
        width = img_copy.shape[1]
        height = img_copy.shape[0]

        left_top = (b_left, 0)
        right_bottom = (b_right, height)

        clipped_img = img_copy[left_top[1]:right_bottom[1], left_top[0]: right_bottom[0]]
        temp_img = clipped_img.copy()


        # 切割上下(螺絲)
        white = []
        black = []

        height = temp_img.shape[0]
        width = temp_img.shape[1]

        white_max = 0
        black_max = 0

        for i in range(height):
            s = 0  
            t = 0  
            for j in range(width):
                if temp_img[i][j] == 255:
                    s += 1
                if temp_img[i][j] == 0:
                    t += 1
            white_max = max(white_max, s)
            black_max = max(black_max, t)
            white.append(s)
            black.append(t)

        # 上
        screw_start_top = 0
        for screw_start_top in range(30, len(white)):
            # 螺絲開始
            if white[screw_start_top] > black_max * 0.05:
                break
        screw_end_top = 0
        for screw_end_top in range(screw_start_top, len(white)):
            # 螺絲結束
            if white[screw_end_top] < black_max * 0.03:
                break
                
        # 下
        screw_start_down = 0
        for screw_start_down in range(len(white) - 25, 0, -1):
            # 螺絲開始
            if white[screw_start_down] > black_max * 0.1:
                break
        screw_end_down = 0
        for screw_end_down in range(screw_start_down, 0, -1):
            # 螺絲結束
            if white[screw_end_down] < black_max * 0.1:
                break

        img_copy = temp_img.copy()
        width = img_copy.shape[1]
        height = img_copy.shape[0]

        left_top = (0, screw_end_top)
        right_bottom = (width, screw_end_down - 10)

        clipped_img = img_copy[left_top[1]:right_bottom[1], left_top[0]: right_bottom[0]]

        return clipped_img
    
    def find_end(self, start, width, black, black_max):

        end = start + 1
        for m in range(start + 1, width - 1):
            # 如果columns幾乎是黑的，就認定結束(邊界)
            if black[m] >= black_max * 0.95:  # 可調整參數
                end = m
                break
        return end
    
    def split_image(self, final_img):

        # 在每個columns裡白色的數量
        white = []
        # 在每個columns裡黑色的數量
        black = []

        height = final_img.shape[0]
        width = final_img.shape[1]

        white_max = 0
        black_max = 0
        # 在每個columns裡黑白的總數
        for i in range(width):
            white_num = 0  
            black_num = 0  
            for j in range(height):
                if final_img[j][i] == 255:
                    white_num += 1
                if final_img[j][i] == 0:
                    black_num += 1
            white_max = max(white_max, white_num)
            black_max = max(black_max, black_num)
            white.append(white_num)
            black.append(black_num)
            
        temp = 1

        # 開始、結束位置
        start = 2
        end = 3

        # 被分割的數量
        num = 0
        
        # 創建新的資料夾
        os.mkdir("C:/Users/user/Downloads/ANPR/leave/display_image/Car" + str(self.leave_Id))
        os.mkdir("C:/Users/user/Downloads/ANPR/leave//train_image/Car" + str(self.leave_Id))

        # train/display
        display_output_path = "C:/Users/user/Downloads/ANPR/leave/display_image/Car" + str(self.leave_Id)
        train_output_path = "C:/Users/user/Downloads/ANPR/leave/train_image/Car" + str(self.leave_Id)


        # 從左至右
        while temp < width - 2:
            temp += 1

            # 找到白邊即開始
            if white[temp] > 0.05 * white_max:
                start = temp
                # 找結束位置
                end = self.find_end(start, width, black, black_max)
                # 下一個開始點
                temp = end

                # 確保是車牌號碼(太小為雜質)
                if end - start > 20:
                    # split
                    if (end + 2) > width:
                        result = final_img[1:height, start - 1:end]
                
                    if (start - 1) < 0:
                        result = final_img[1:height, start :end + 2]

                    result = final_img[1:height, start - 1:end + 2]

                    # 儲存分割結果到資料夾
                    cv2.imwrite(display_output_path + '/' + str(num) + '.jpg', result)

                    # resize (因為訓練模組的input需要(784,)的大小，所以先存為(28,28))
                    result = cv2.resize(result, (28, 28), interpolation=cv2.INTER_CUBIC)

                    cv2.imwrite(train_output_path + '/' + str(num) + '.jpg', result)

                    num += 1

        self.lic_num = num
        
         
    def process_image(self):

        # 讀取照片
        img = cv2.imread('leave/origin_image/image_' + str(self.leave_Id) + '.jpg')
        self.origin_img = img
        # 轉灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 統一大小
        gray = cv2.resize(gray, (1000, 500))

        final_img = self.clip_image(gray)
        
        self.split_image(final_img)
        
        
    def get_image(self):

        for i in range(self.lic_num):
            # 取得照片
            globals()['leave_img_' + str(i)] = img.imread('leave/train_image/Car' + str(self.leave_Id) + '/' + str(i) + '.jpg')
            # reshape to 28x28
            globals()['leave_img_' + str(i)] = globals()['leave_img_' + str(i)].reshape([28,28])
            # 左右翻轉
            globals()['leave_img_' + str(i)] = np.fliplr(globals()['leave_img_' + str(i)])
            # 旋轉90度
            globals()['leave_img_' + str(i)] = np.rot90(globals()['leave_img_' + str(i)])
            # reshape to 784x1
            globals()['leave_img_' + str(i)] = globals()['leave_img_' + str(i)].reshape([784,])
        
    def find_hyphen(self, img, number_plate, hyphen): # '-'

        two_list = []
        # 把照片兩行兩行分別加總，找到差距最大的值即為'-'符號
        for i in range(10, 18):
            total = np.mean(img.reshape([28, 28])[:, i])
            total += np.mean(img.reshape([28, 28])[:, i + 1])
            two_list.append(total)
            
        two = np.array(two_list)
        sort_index = np.argsort(two)
        

        max_pixel = two[sort_index[-1]]
        min_pixel = two[sort_index[0]]

        # is hyphen
        if max_pixel - min_pixel > 400:
            hyphen.append(img)
            return 1, number_plate, hyphen
        # not hyphen
        else:
            number_plate.append(img)
            return 0, number_plate, hyphen

    def identify_number_plate(self):

        # load model
        rfc_model = joblib.load('rfc_model')
        
        self.get_image()
        
        number_plate = []
        hyphen = []
        hyphen_id = 0
        # find hyphen
        for i in range(self.lic_num):
            is_hyphen, number_plate, hyphen = self.find_hyphen(globals()['leave_img_' + str(i)], number_plate, hyphen)
            if is_hyphen == 1:
                hyphen_id = i
                
        number_plate_arr = np.array(number_plate ,dtype='int64')
        # 預測車牌
        predictions = rfc_model.predict(number_plate_arr)
        # append to final number plate
        count = 0
        for index in predictions:
            count += 1
            if count == hyphen_id:
                self.final_number_plate.append(self.alphabet[index])
                self.final_number_plate.append('-')
                
            else:
                self.final_number_plate.append(self.alphabet[index])
                
        return self.final_number_plate
    
    def check_database(self, parking_df):
        
        detected = 0
        number_plate_str = ''
        for i in self.final_number_plate:
            number_plate_str += i
        # 從dataframe中找車牌號碼
        for i in range(len(parking_df)):
            if parking_df['Number_Plate'][i] == number_plate_str:
                detected = 1
                if parking_df['is_Paid'][i] == 1:
                    open_gate()
                    break
                else:
                    print("Unpaid")
        if detected == 0:
            print('Not found your car')




def open_gate():

    print("Open the Gate")
    time.sleep(8)
    print('Close the Gate')



def entry_cam(index):

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        cap.set(3, 1240)
        cap.set(4, 640)

        num = 0

        while True:
            success, img = cap.read()
            if success:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                break

            # 取得輪廓
            gray = cv2.GaussianBlur(gray, (15, 15), 0)
            gray = cv2.bilateralFilter(gray, 40, 40, 500)
            edged = cv2.Canny(gray, 12, 35)
            cv2.imshow('Edge', edged)

            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            

            for contour in contours:
                cv2.imshow('Video', img)
                x, y, w, h = cv2.boundingRect(contour)
                if ((x + w) > 950) and ((y + h) > 300)  and (w > 850) and (h > 300):
                    num += 1
                # 重複多次一點確保為車牌
                if num > 10:
                    clipped_img = img[y : y + h, x : x + w]
                    cv2.imwrite('entry/origin_image/image_'+ str(index) + '.jpg', clipped_img)
                    break
            if num > 10:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()



def payment_interface(amounts_payable):

    is_paid = 0
    print("Amounts Payable : {}".format(amounts_payable))
    customer_paid = int(input('You Pay : '))
    while is_paid == 0:
        if customer_paid < amounts_payable:
            customer_paid += int(input('You Pay : '))
            continue
        else:
            change = customer_paid - amounts_payable
            is_paid = 1
            print("Change {} dollar.".format(change))
            print('Paid !')
            break
    return is_paid



def payment_machine1(parking_df):

    found = 0
    while found == 0:
        customer_number = []
        customer_number.append(input('Please enter your license plate number(before -) : '))

        # 根據輸入的數字，從dataframe裡面找可能的車牌號碼
        possible_Id = []
        for i in range(0, len(parking_df)):
            if customer_number[0] == parking_df['Number_Plate'][i].split('-')[0]:
                possible_Id.append(i)
        if len(possible_Id) == 0:
            print("Can't find your car, Please enter your number plate again.")
            continue
        else:
            found = 1

    return possible_Id
        


def payment_machine2(possible_Id):

    # 顯示可能的車牌照片給客人看
    print('Choose Your Car')
    for i in range(0, len(possible_Id)):
        image = mpimg.imread('entry/origin_image/image_{}.jpg'.format(possible_Id[i] + 1))
        plt.title('NO.{}'.format(i + 1))
        plt.imshow(image)
        plt.show()

    while True:
        num = int(input('Enter Your Car Number : '))
        if num > len(possible_Id):
            print('It out of range, please enter the number of your car.')
            continue
        final_Id = possible_Id[num - 1]
        break

    return final_Id

        
def check_out(parking_df, final_Id):

    # 離開時間
    leave_Year = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[0].split('-')[0])
    leave_Month = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[0].split('-')[1])
    leave_Day = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[0].split('-')[2])
    leave_Hour = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[1].split(':')[0])
    leave_Minute = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[1].split(':')[1])
    leave_Second = int(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split()[1].split(':')[2])

    entry_Year = parking_df.loc[final_Id]['Entry_Year']
    entry_Month = parking_df.loc[final_Id]['Entry_Month']
    entry_Day = parking_df.loc[final_Id]['Entry_Day']
    entry_Hour = parking_df.loc[final_Id]['Entry_Hour']
    entry_Minute = parking_df.loc[final_Id]['Entry_Minute']
    entry_Second = parking_df.loc[final_Id]['Entry_Second']

    Month_num = 12
    Hour_num = 24
    Minute_num = 60
    
    an_Hour_Pay = 30
    amounts_payable = 0

    year = -1
    month = -1
    day = -1

    year = leave_Year - entry_Year
    start_Year = entry_Year
    
    if year < 1:
        month = leave_Month - entry_Month
        start_Month = entry_Month
        if month < 1:
            day = leave_Day - entry_Day
            start_Day = entry_Day

    # w, d = calendar.monthrange(start_Year, entry_Month) d表示在這一年的這一個月裡的天數 

    # 超過一年
    if year >= 1:
        # 大於一年
        if year > 1:
            while year > 1:
                # 一整年
                for m in range(13):
                    w, d = calendar.monthrange(start_Year, m)
                    amounts_payable += d * Hour_num * an_Hour_Pay
                year -= 1
                start_Year += 1
            # 剩下的
            w, d = calendar.monthrange(start_Year, entry_Month)
            amounts_payable += (d - entry_Day + 1) * Hour_num * an_Hour_Pay

            amounts_payable += (Hour_num - (entry_Hour + 1)) * an_Hour_Pay

            for m in range(entry_Month + 1, 13):
                w, d = calendar.monthrange(start_Year, m)
                amounts_payable += d * Hour_num * an_Hour_Pay

            start_Year += 1
            for m in range(0, leave_Month):
                w, d = calendar.monthrange(start_Year, m)
                amounts_payable += d * Hour_num * an_Hour_Pay

            amounts_payable += (leave_Day - 1) * Hour_num * an_Hour_Pay

            amounts_payable += leave_Hour * an_Hour_Pay
        # 一年
        else:
            w, d = calendar.monthrange(start_Year, entry_Month)
            amounts_payable += (d - entry_Day + 1) * Hour_num * an_Hour_Pay

            amounts_payable += (Hour_num - (entry_Hour + 1)) * an_Hour_Pay
        
            for m in range(entry_Month + 1, 13):
                w, d = calendar.monthrange(start_Year, m)
                amounts_payable += d * Hour_num * an_Hour_Pay
                
            start_Year += 1
            for m in range(0, leave_Month):
                w, d = calendar.monthrange(start_Year, m)
                amounts_payable += d * Hour_num * an_Hour_Pay

            amounts_payable += (leave_Day - 1) * Hour_num * an_Hour_Pay

            amounts_payable += leave_Hour * an_Hour_Pay      
    # 超過一個月
    elif month >= 1:
        # 大於一個月
        if month > 1:
            w, d = calendar.monthrange(start_Year, entry_Month)
            amounts_payable += (d - (entry_Day + 1)) * Hour_num * an_Hour_Pay

            amounts_payable += (Hour_num - (entry_Hour + 1)) * an_Hour_Pay

            for m in range(start_Month, leave_Month):
                w, d = calendar.monthrange(start_Year, m)
                amounts_payable += d * Hour_num * an_Hour_Pay

            amounts_payable += (leave_Day - 1) * Hour_num * an_Hour_Pay

            amounts_payable += leave_Hour * an_Hour_Pay
        # 一個月
        else:
            w, d = calendar.monthrange(start_Year, entry_Month)
            amounts_payable += (d - (entry_Day + 1)) * Hour_num * an_Hour_Pay

            amounts_payable += (Hour_num - (entry_Hour + 1)) * an_Hour_Pay
            
            amounts_payable += (leave_Day - 1) * Hour_num * an_Hour_Pay

            amounts_payable += leave_Hour * an_Hour_Pay
    # 超過一天
    elif day >= 1:
        # 大於一天
        if day > 1:
            amounts_payable += (day - 1) * Hour_num * an_Hour_Pay
            amounts_payable += (Hour_num - (entry_Hour + 1)) * an_Hour_Pay
            amounts_payable += leave_Hour * an_Hour_Pay
        # 一天
        else:
            amounts_payable += (Hour_num - (entry_Hour + 1)) * an_Hour_Pay
            amounts_payable += leave_Hour * an_Hour_Pay
    # 小於一天
    elif day == 0:
        if entry_Minute > leave_Minute:
            amounts_payable += (leave_Hour - entry_Hour) * an_Hour_Pay
        else:
            amounts_payable += ((leave_Hour - entry_Hour) + 1) * an_Hour_Pay

    is_paid = payment_interface(amounts_payable)

    # input data
    parking_df.loc[final_Id] = [parking_df.loc[final_Id]['Car_Id'], parking_df.loc[final_Id]['Number_Plate'],
                                entry_Year, entry_Month, entry_Day, entry_Hour, entry_Minute, entry_Second,
                                leave_Year, leave_Month, leave_Day, leave_Hour, leave_Minute, leave_Second,
                                amounts_payable, is_paid]

    # 把dataframe變成excel
    writer = pd.ExcelWriter('Parking_df.xlsx') 
    parking_df.to_excel(writer, sheet_name='dataframe', index=False, na_rep='NaN')

    # 客製化寬度
    for column in parking_df:
        column_width = max(parking_df[column].astype(str).map(len).max(), len(column)) 
        col_idx = parking_df.columns.get_loc(column)                                   
        writer.sheets['dataframe'].set_column(col_idx, col_idx, column_width)
    writer.save()

    return parking_df