import os
import aiohttp
import cv2
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from PIL import Image,ImageEnhance
from pydantic import BaseModel
from fastapi import FastAPI,HTTPException
from fastapi.responses import StreamingResponse
 
 
app = FastAPI(
    title="sqy-image-scorer",
    description="Use this API to get the image score",
    version="2.0.1",
)

 
class URL(BaseModel):
    url_: str
 

def extract_filename(URL):
    parsed = urlparse(URL)
    return os.path.basename(parsed.path)


@app.get("/sqy_image")
async def image_scorer(URL1): 

    '''This function get image from your system or
       take input as original image
    '''
#     URL1 = check_image.url_

    try:
        filename = extract_filename(URL1)
        filename = filename.strip()

    except Exception:
        ##logger.info("Error: HTTPException(status_code=406, detail=Not a valid URL)")
        raise HTTPException(status_code=406, detail="Not a valid URL")
 
    if URL1.lower().endswith((".jpg", ".png", ".jpeg", ".gif", ".webp",".jfif")) == False:
        ##logger.info("Error: HTTPException(status_code=406, detail=Not a valid URL)")
        raise HTTPException(status_code=406, detail="Not a valid URL")

    async with aiohttp.ClientSession() as session:
        async with session.get(URL1) as resp:
            contents = await resp.read()
  
    async with aiohttp.ClientSession() as session:
        async with session.get(URL1) as resp:
            contents = await resp.read()

    if contents == None:
        raise HTTPException(status_code=406, detail="No image found.")

    image = Image.open(BytesIO(contents))

    #this function get the format type of input image
    def get_format(filename):
        format_ = filename.split(".")[-1]
        if format_.lower() == "jpg":
            format_ = "jpeg"
        elif format_.lower == "webp":
            format_ = "WebP"
    
        return format_
    
    format_ = get_format(filename) #here format_ store the type of image by filename

    def calculate_brightness(image):
        greyscale_image = image.convert('L')
        histogram = greyscale_image.histogram()
        pixels = sum(histogram)
        brightness = scale = len(histogram)

        for index in range(0, scale):
            ratio = histogram[index] / pixels
            brightness += ratio * (-scale + index)

        return 1 if brightness == 255 else brightness / scale
    bright1 = calculate_brightness(image)
    print("b_bright",calculate_brightness(image))

    def calculate_sharpness(image): #here calculate the sharpness 
        image = Image.open(BytesIO(contents))
        image.save("original_img."+format_)

        try:
            img = cv2.imread("original_img."+format_, cv2.IMREAD_GRAYSCALE)
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            print("sharpness try", laplacian_var)

            img_c = cv2.imread("original_img."+format_)
            Y = cv2.cvtColor(img_c, cv2.COLOR_BGR2YUV)[:,:,0]
            # compute min and max of Y
            min = np.min(Y)
            max = np.max(Y)

            # compute contrast
            contrast = (max-min)/(max+min)
            print("try min=",min)

            img_s = cv2.imread("original_img."+format_)
            img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
            saturation = img_hsv[:, :, 1].mean()
            print("saturation try",saturation)

        except:
            img_s = cv2.imread("original_img."+format_)
            img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
            saturation = img_hsv[:, :, 1].mean()
            print("saturation",saturation)

            img = cv2.imread("original_img."+format_, cv2.IMREAD_GRAYSCALE)
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            print("lap except", laplacian_var)
        
        if laplacian_var > 500:
            result = 10

        if laplacian_var > 290 and laplacian_var < 500:
            result = 9

        if laplacian_var > 135 and laplacian_var < 290:
            result = 8

        if laplacian_var > 105 and laplacian_var < 135:
            result = 7

        if laplacian_var > 80 and laplacian_var < 105:
            result = 6

        if laplacian_var > 70  and laplacian_var < 80:
            result = 5

        if laplacian_var > 60 and laplacian_var < 70:
            result = 4

        if laplacian_var > 50 and laplacian_var < 60:
            result = 3

        if laplacian_var > 45 and laplacian_var < 50:
            result = 2

        if laplacian_var > 1 and  laplacian_var < 45:
            result = 1  

        if min < 9 and laplacian_var < 100 and laplacian_var >40:
            result = 8

        if min >3 and laplacian_var >250:
            result = 8

        if saturation > 115 and laplacian_var >400:
            result = 9

        if saturation > 130 and laplacian_var >300:
            result = 8

        if saturation < 85 and laplacian_var < 50:
            result = 3

        if saturation < 103 and saturation > 85 and laplacian_var < 60 and laplacian_var >40:
            result = 6

        if min < 5 and laplacian_var < 30:
            result = 4

        if saturation > 147 and saturation <165:
            result = 7

        if saturation > 175:
            result = 3

        if saturation < 85 and laplacian_var < 50 and min > 1:
            result=3
            
        if min < 1  and laplacian_var < 40:
            result = 5

        if bright1 > 0.63 and laplacian_var < 40 and saturation < 40:
            result = 7

        if laplacian_var < 50 and min < 1 and saturation <50:
            result = 7

        if bright1 < 0.3:
            result = 3
 
        return result

    
    result_check1 = calculate_sharpness(image)
    s2 = slice(0,6)

    buffer = BytesIO()
    image.save(buffer, format=format_)
    buffer.seek(0)

    return ({"score_":result_check1/10})

class URL1(BaseModel):
    url_: list
 
@app.post("/sqy_image")
async def image_scorer(image_url:URL1): 

    '''This function get image from your system or
       take input as original image
    '''

    URL1 = image_url.url_

    list_1 = URL1
    a_list = []

    for i in range (len(list_1)):
        URL1 = list_1[i]

        async with aiohttp.ClientSession() as session:
            async with session.get(URL1) as resp:
                contents = await resp.read()
    
        async with aiohttp.ClientSession() as session:
            async with session.get(URL1) as resp:
                contents = await resp.read()
 
        if contents == None:
            raise HTTPException(status_code=406, detail="No image found.")

        image = Image.open(BytesIO(contents))

        #this function get the format type of input image
        def get_format(filename):
            format_ = filename.split(".")[-1]
            if format_.lower() == "jpg":
                format_ = "jpeg"
            elif format_.lower == "webp":
                format_ = "WebP"
        
            return format_

        image = PIL.Image.open(BytesIO(contents))
        format_ =image.format #here format_ store the type of image by filename

        if(image):
            greyscale_image = image.convert('L')
            histogram = greyscale_image.histogram()
            pixels = sum(histogram)
            brightness = scale = len(histogram)

            for index in range(0, scale):
                ratio = histogram[index] / pixels
                brightness += ratio * (-scale + index)

        bright1 = brightness / scale
        # print(bright1)

        if(image): #here calculate the sharpness 
            image = Image.open(BytesIO(contents))
            image.save("original_img."+format_)
            # print("fifth")

            try:
                img = cv2.imread("original_img."+format_, cv2.IMREAD_GRAYSCALE)
                laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                # print("sharpness try", laplacian_var)

                img_c = cv2.imread("original_img."+format_)
                Y = cv2.cvtColor(img_c, cv2.COLOR_BGR2YUV)[:,:,0]
                # compute min and max of Y
                min = np.min(Y)
                max = np.max(Y)

                # compute contrast
                contrast = (max-min)/(max+min)
                # print("try min=",min)

                img_s = cv2.imread("original_img."+format_)
                img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
                saturation = img_hsv[:, :, 1].mean()
                # print("saturation try",saturation)

            except:
                img_s = cv2.imread("original_img."+format_) 
                img_hsv = cv2.cvtColor(img_s, cv2.COLOR_BGR2HSV)
                saturation = img_hsv[:, :, 1].mean()
                # print("saturation",saturation)

                img = cv2.imread("original_img."+format_, cv2.IMREAD_GRAYSCALE)
                laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                # print("lap except", laplacian_var)
            
            if laplacian_var > 500:
                result = 10

            if laplacian_var > 290 and laplacian_var < 500:
                result = 9

            if laplacian_var > 135 and laplacian_var < 290:
                result = 8

            if laplacian_var > 105 and laplacian_var < 135:
                result = 7

            if laplacian_var > 80 and laplacian_var < 105:
                result = 6

            if laplacian_var > 70  and laplacian_var < 80:
                result = 5

            if laplacian_var > 60 and laplacian_var < 70:
                result = 4

            if laplacian_var > 50 and laplacian_var < 60:
                result = 3

            if laplacian_var > 45 and laplacian_var < 50:
                result = 2

            if laplacian_var > 1 and  laplacian_var < 45:
                result = 1  

            if min < 9 and laplacian_var < 100 and laplacian_var >40:
                result = 8

            if min >3 and laplacian_var >250:
                result = 8

            if saturation > 115 and laplacian_var >400:
                result = 9

            if saturation > 130 and laplacian_var >300:
                result = 8

            if saturation < 85 and laplacian_var < 50:
                result = 3

            if saturation < 103 and saturation > 85 and laplacian_var < 60 and laplacian_var >40:
                result = 6

            if min < 5 and laplacian_var < 30:
                result = 4

            if saturation > 147 and saturation <165:
                result = 7

            if saturation > 175:
                result = 3

            if saturation < 85 and laplacian_var < 50 and min > 1:
                result=3
                
            if min < 1  and laplacian_var < 40:
                result = 5

            if bright1 > 0.63 and laplacian_var < 40 and saturation < 40:
                result = 7

            if laplacian_var < 50 and min < 1 and saturation <50:
                result = 7

            if bright1 < 0.3:
                result = 3
            
            integers_to_append = result
            a_list.append(integers_to_append)
                    
            # print("result",a_list)
        
    sum1 = 0
    a_list.sort(reverse = True)
    print(a_list)

    if (len(a_list) > 10):
        print("greater than 10")

        for i in range (10):
            sum1 += a_list[i]
            # print(a_list[i])

    if (len(a_list) < 10):
        # print("less than 10")

        for i in range (len(a_list)):
            sum1 += a_list[i]
            
    result_check1 = sum1
    total_url = (len(list_1))
    result1 = result_check1/total_url
    result = (round(result1)/10)
    
    # result = str(result)
    # s1 = slice(3)
    # print(result[s1])
    print("Score = ",result)
    
    return ({"score_":result})
