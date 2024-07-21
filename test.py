import cv2
import pyzbar.pyzbar as pyzbar
from PIL import Image, ImageDraw, ImageFont


def decodeDisplay(img_path):
    img_data = cv2.imread(img_path)
    # 转为灰度图像
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    barcodes = pyzbar.decode(gray)
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        cv2.rectangle(img_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        a = [x+w/2, y+h/2]
        print("center:", a)
        # 条形码数据为字节对象，所以如果我们想在输出图像上
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # 不能显示中文
        # 绘出图像上条形码的数据和条形码类型
        # 更换为：
        img_PIL = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
        # 参数（字体，默认大小）
        font = ImageFont.truetype('msyh.ttc', 35)
        # 字体颜色（rgb)
        fillColor = (0, 255, 255)
        # 文字输出位置
        position = (x, y - 10)
        # 输出内容
        str = barcodeData
        # 需要先把输出的中文字符转换成Unicode编码形式(  str.decode("utf-8)   )
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position, str, font=font, fill=fillColor)
        # 使用PIL中的save方法保存图片到本地
        img_PIL.save('result.jpg', 'jpeg')
        # 向终端打印条形码数据和条形码类型
        return a


# a = decodeDisplay("C:/Users/11478/Desktop/par/0-right.jpg")
# print(a)
# a = decodeDisplay("C:/Users/11478/Desktop/par/0-left.jpg")
# print(a)

