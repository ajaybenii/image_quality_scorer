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
    title="sqy-watermark-engine",
    description="Use this API to paste Square Yards logo as a watermark at the center of input images",
    version="2.0.1",
)
 
class URL(BaseModel):
    url_: str
 

def extract_filename(URL):
    parsed = urlparse(URL)
    return os.path.basename(parsed.path)

async def get_image_properties(URL):
    filename = None
    try:
        filename = extract_filename(URL)
        filename = filename.strip()
    except Exception as e:
        raise HTTPException(status_code=406, detail="Not a valid URL")
 
    if URL.lower().endswith((".jpg", ".png", ".jpeg", ".gif", ".webp")) == False:
        raise HTTPException(status_code=406, detail="Not a valid URL")
    
    contents = None
    original_image = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(URL) as resp:
                contents = await resp.read()
 
        if contents == None:
            raise HTTPException(status_code=406, detail="No image found.")
 
        original_image = Image.open(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error while reading the image. Make sure that the URL is a correct image link.")
    
    return filename, original_image
 


@app.post("/uploadfile")
async def image_enhance(Enhance_image: URL): 

 
    '''This function get image from your system or
       take input as original image
    '''
    URL1 = Enhance_image.url_
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


    #this function for gave the same type of format to output
    def get_content_type(format_):
        type_ = "image/jpeg"
        if format_ == "gif":
            type_ = "image/gif"
        elif format_ == "webp":
            type_ = "image/webp"
        elif format_ == "png":
            type_ = "image/png"

        return type_

    format_ = get_format(filename) #here format_ store the type of image by filename

    def calculate_quality(image):
        path = image.save("original_img.jpg")
        
        img = cv2.imread("original_img.jpg", cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        if laplacian_var > 200:
            result = "Good Quality Image"
            
        if laplacian_var > 91 and laplacian_var < 200 :
            result = "Average Quality Image"

        if laplacian_var > 1 and laplacian_var < 50 :
            result = "Poor Quality Image"

        print(laplacian_var)
        
        return result

    def calculate_contrast (image):
        # path = image.save("original_img.jpg")  
        img = cv2.imread("original_img.jpg")
        Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]

        # compute min and max of Y
        min = np.min(Y)
        max = np.max(Y)

        # compute contrast
        contrast = (max-min)/(max+min)
        print(min,max,contrast)
        return (min,max,contrast)


    def calculate_qual(image):

        path = image.save("original_img.jpg")
        img = cv2.imread("original_img.jpg", cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        return laplacian_var
    
      
    def calculate_brightness(image):
        greyscale_image = image.convert('L')
        histogram = greyscale_image.histogram()
        pixels = sum(histogram)
        brightness = scale = len(histogram)

        for index in range(0, scale):
            ratio = histogram[index] / pixels
            brightness += ratio * (-scale + index)

        return 1 if brightness == 255 else brightness / scale
    
    result_check3 = ("contrast level",calculate_contrast(image))
    result_check1 = (calculate_qual(image)/10)
    s2 = slice(0,6)

    result_check = ("Image Quality is ...",calculate_quality(image))
    result_check2 = ("Brightness Level",calculate_brightness(image))
    
    buffer = BytesIO()
    image.save(buffer, format=format_)
    buffer.seek(0)

    return result_check1

