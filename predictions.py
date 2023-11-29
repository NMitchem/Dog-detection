from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision import transforms
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained("PetModel")
import urllib.request

from PIL import Image, ImageDraw, ImageFont
import json
id2label = json.load(open("id2label.json","r"))
import torch
yolomodel = torch.hub.load('ultralytics/yolov5', 'custom','best.pt')
import sys
import os
def prediction(img_path):
   results = yolomodel(img_path)
   results.show()
   for i in results.crop(save = False):
      box = [j.item() for j in i["box"]]
      try:
         im=Image.open(img_path).convert("RGB")
      except:
         urllib.request.urlretrieve(img_path,"test.jpg")
         im = Image.open("test.jpg")
         os.remove("test.jpg")
      im = im.crop(box)
      target_size = (224, 224)
      #transform = transforms.Resize(target_size)
      #im_trans = transforms.Compose([transform, transforms.ToTensor()])
      #img_tensor = im_trans(im)
      encoding = feature_extractor(images=im, return_tensors="pt")
      pixel_values = encoding['pixel_values']
      outputs = model(pixel_values)
      result = outputs.logits.softmax(1).argmax(1)
      new_result = result.tolist()
      label = "Prediction: " + id2label[str(new_result[0])][4:]
      draw = ImageDraw.Draw(im)
      font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 24)
      pos = (10, 10)
      color = (255, 0, 0)
      draw.text(pos, label, font=font, fill=color)
      im.show()
      return label

print(prediction(sys.argv[1]))
