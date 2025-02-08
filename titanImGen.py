import boto3
import json
import os
import base64
import sys
from PIL import Image
from io import BytesIO


bedrock = boto3.client(service_name="bedrock-runtime")


prompt = "Generate image a book with title 'Murmures du coeur ' and a man with maroon color seating  a side "
body = json.dumps(
    {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,   # Required
#           "negativeText": "<text>"  # Optional
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,   # Range: 1 to 5 
            "quality": "premium",  # Options: standard or premium
            "height": 768,         # Supported height list in the docs 
            "width": 1280,         # Supported width list in the docs
            "cfgScale": 7.5,       # Range: 1.0 (exclusive) to 10.0
            "seed": 42             # Range: 0 to 214783647
        }
    }
)

response = bedrock.invoke_model(
    body = body,#"{\"textToImageParams\":{\"text\":\"angry lion eating snake\"},\"taskType\":\"TEXT_IMAGE\",\"imageGenerationConfig\":{\"cfgScale\":8,\"seed\":42,\"quality\":\"standard\",\"width\":1024,\"height\":1024,\"numberOfImages\":3}}",
    modelId = "amazon.titan-image-generator-v2:0",
    accept = "application/json",
    contentType = "application/json"

)
response_boy = json.loads(response.get("body").read())
images = [Image.open(BytesIO(base64.b64decode(base64_image))) for base64_image in response_boy.get('images')]

for img in images:
    img.show()

