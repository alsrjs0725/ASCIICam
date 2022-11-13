import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

FONT = 'JetBrainsMono-Medium.ttf'
COLOR = (0, 255, 0)
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 10
ASCII_WIDTH = 64
ASCII_HEIGHT = 36
ASCII = '$8#hpZLYvr/)[_>I"\               '[::-1]
REMOVE_LIGHT = True


def num2char(num):
    return ASCII[round((len(ASCII) - 1) * num / 255)]

def text_to_image(
text: str,
font_filepath: str,
font_size: int,
font_align="center"):  # https://stackoverflow.com/a/72615131/16614074

   font = ImageFont.truetype(font_filepath, size=font_size)
   box = font.getsize_multiline(text)
   img = Image.new('RGB', (box[0], box[1]))
   draw = ImageDraw.Draw(img)
   draw_point = (0, 0)
   draw.multiline_text(draw_point, text, font=font, fill=(0, 255, 0), align=font_align)
   return img


if __name__ == '__main__':
    while True:
        try:
            camidx = int(input('Choose a integer that greater than 0 (cam index): '))
            if camidx < 1:
                raise ValueError()
            break
        except ValueError:
            print('Not a integer that greater than 0')

    capture = cv2.VideoCapture(0)
    num2charvec = np.vectorize(num2char)

    while cv2.waitKey(33) < 0:
        ret, frame_before_resizing = capture.read()
        frame = cv2.resize(frame_before_resizing, (ASCII_WIDTH, ASCII_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if REMOVE_LIGHT:
            inv_gray = cv2.bitwise_not(gray)
            th_ret, th_gray = cv2.threshold(inv_gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        else:
            th_ret, th_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

        char_arr = num2charvec(th_gray)
        char_arr_endl = np.pad(char_arr, ((0, 0), (0, 1)), 'constant', constant_values='\n')
        str_arr = np.reshape(char_arr_endl, -1)
        _str = ''.join(str_arr)
        str_frame = np.array(text_to_image(_str, FONT, 15))
        str_frame = cv2.resize(str_frame, (CAM_WIDTH, CAM_HEIGHT))
        
        # cv2.imshow('th_gray', th_gray)
        cv2.imshow('ASCIICam', str_frame)
        cv2.imshow('Cam', frame_before_resizing)
        time.sleep(1 / CAM_FPS)
