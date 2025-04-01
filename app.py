import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from math import atan2
from os import listdir, path
from PIL import Image as PImage

OUT_W = 130
OUT_H = 170
OUT_EYE_SPACE = 64
OUT_NOSE_TOP = 72

EYE_0_IDX = 36
EYE_1_IDX = 45

haarcascade = "./models/haarcascade_frontalface_alt2.xml"
face_detector = cv2.CascadeClassifier(haarcascade)

LBFmodel = "./models/lbfmodel.yaml"
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

NUM_OUTS = 16
all_outputs = [gr.Image(format="jpeg", visible=False) for _ in range(NUM_OUTS)]

def face(img_in):
  out_pad = NUM_OUTS * [gr.Image(visible=False)]
  if img_in is None:
    return out_pad

  pimg = img_in.convert("L")
  pimg.thumbnail((1000,1000))
  imgg = np.array(pimg).copy()

  iw,ih = pimg.size
  
  faces = face_detector.detectMultiScale(imgg)

  if len(faces) < 1:
    return out_pad

  biggest_faces = faces[np.argsort(-faces[:,2])]
  _, landmarks = landmark_detector.fit(imgg, biggest_faces)

  if len(landmarks) < 1:
    return out_pad

  out_images = []
  for landmark in landmarks:
    eye0 = np.array(landmark[0][EYE_0_IDX])
    eye1 = np.array(landmark[0][EYE_1_IDX])
    mid = np.mean([eye0, eye1], axis=0)

    eye_line = eye1 - eye0
    tilt = atan2(eye_line[1], eye_line[0])
    tilt_deg = 180 * tilt / np.pi

    scale = OUT_EYE_SPACE / abs(eye0[0] - eye1[0])
    pimgs = pimg.resize((int(iw * scale), int(ih * scale)), resample=PImage.Resampling.LANCZOS)

    # rotate around nose
    new_mid = [int(c * scale) for c in mid]
    crop_box = (new_mid[0] - (OUT_W // 2),
                new_mid[1] - OUT_NOSE_TOP,
                new_mid[0] + (OUT_W // 2),
                new_mid[1] + (OUT_H - OUT_NOSE_TOP))

    img_out = pimgs.rotate(tilt_deg, center=new_mid, resample=PImage.Resampling.BICUBIC).crop(crop_box)
    out_images.append(gr.Image(img_out, visible=True))

  out_images += out_pad
  return out_images[:NUM_OUTS]


with gr.Blocks() as demo:
  gr.Markdown("""
              # 9103H 2024F Face Alignment Tool.
              ## Interface for face detection, alignment, cropping\
              to help create dataset for [HW10](https://github.com/DM-GY-9103-2024F-H/HW10).
              """)

  gr.Interface(
    face,
    inputs=gr.Image(type="pil"),
    outputs=all_outputs,
    cache_examples=True,
    examples=[["./imgs/03.webp"], ["./imgs/11.jpg"]],
    allow_flagging="never",
  )

if __name__ == "__main__":
   demo.launch()
