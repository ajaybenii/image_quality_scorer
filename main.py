import os
import aiohttp
import cv2
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from PIL import Image
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


@app.post("/sqy_image")
async def image_scorer(check_image: URL): 

    '''This function get image from your system or
       take input as original image
    '''
    URL1 = check_image.url_
    filename = extract_filename(URL1)
    filename = filename.strip()
  
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
    
    def calculate_contrast (image):
        image.save("original_img."+format_)  
        img = cv2.imread("original_img."+format_)
        Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]

        # compute min and max of Y
        min = np.min(Y)
        max = np.max(Y)

        # compute contrast
        contrast = (max-min)/(max+min)
        print(min,max,contrast)
        return (min,max,contrast)


    def calculate_sharpness(image): #here calculate the sharpness 

        image.save("original_img."+format_)
        img = cv2.imread("original_img."+format_, cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        if laplacian_var > 500:
            result = 10
        if laplacian_var > 300 and laplacian_var < 500:
            result = 9
        if laplacian_var > 200 and laplacian_var < 300:
            result = 8
        if laplacian_var > 115 and laplacian_var < 200:
            result = 7
        if laplacian_var > 100 and laplacian_var < 115:
            result = 6
        if laplacian_var > 90  and laplacian_var < 100:
            result = 5
        if laplacian_var > 80 and laplacian_var < 90:
            result = 4
        if laplacian_var > 70 and laplacian_var < 80:
            result = 3
        if laplacian_var > 60 and laplacian_var < 70:
            result = 2
        if laplacian_var > 1 and  laplacian_var < 60:
            result = 1  
        
        return result
#         return laplacian_var
    
      
    def calculate_brightness(image): #here calculate the brightness
        greyscale_image = image.convert('L')
        histogram = greyscale_image.histogram()
        pixels = sum(histogram)
        brightness = scale = len(histogram)

        for index in range(0, scale):
            ratio = histogram[index] / pixels
            brightness += ratio * (-scale + index)

        return 1 if brightness == 255 else brightness / scale
    
    result_check3 = ("contrast level",calculate_contrast(image))
    result_check1 = calculate_sharpness(image)
    s2 = slice(0,6)
    result_check2 = ("Brightness Level",calculate_brightness(image))
    
    buffer = BytesIO()
    image.save(buffer, format=format_)
    buffer.seek(0)

    return result_check1

