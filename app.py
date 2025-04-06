import cv2
import gradio as gr
import numpy as np

from huggingface_hub import hf_hub_download
from math import atan2
from PIL import Image as PImage
from ultralytics import YOLO

OUT_W = 130
OUT_H = 170

OUT_EYE_SPACE = 64
OUT_FACE_WIDTH = 89
OUT_NOSE_TOP = 72

EYE_0_IDX = 36
EYE_1_IDX = 45

TEMPLE_0_IDX = 0
TEMPLE_1_IDX = 16

yolo_model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
face_detector = YOLO(yolo_model_path)

LBFmodel = "./models/lbfmodel.yaml"
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

NUM_OUTS = 16
all_outputs = [gr.Image(format="jpeg", visible=False) for _ in range(NUM_OUTS)]

def face(img_in):
  out_pad = NUM_OUTS * [gr.Image(visible=False)]
  if img_in is None:
    return out_pad

  img = img_in.copy()
  img.thumbnail((1000,1000))
  img_np = np.array(img).copy()

  iw,ih = img.size

  output = face_detector.predict(img, verbose=False)

  if len(output) < 1 or len(output[0]) < 1:
    return out_pad

  faces_xyxy = output[0].boxes.xyxy.numpy()
  faces = np.array([[x0, y0, (x1 - x0), (y1 - y0)] for x0,y0,x1,y1 in faces_xyxy])

  biggest_faces = faces[np.argsort(-faces[:,2])]
  _, landmarks = landmark_detector.fit(img_np, biggest_faces)

  if len(landmarks) < 1:
    return out_pad

  out_images = []
  for landmark in landmarks:
    eye0 = np.array(landmark[0][EYE_0_IDX])
    eye1 = np.array(landmark[0][EYE_1_IDX])
    temple0 = np.array(landmark[0][TEMPLE_0_IDX])
    temple1 = np.array(landmark[0][TEMPLE_1_IDX])

    mid = np.mean([eye0, eye1], axis=0)

    eye_line = eye1 - eye0
    tilt = atan2(eye_line[1], eye_line[0])
    tilt_deg = 180 * tilt / np.pi


    scale = min(OUT_EYE_SPACE / np.linalg.norm(eye1 - eye0),
                OUT_FACE_WIDTH / np.linalg.norm(temple1 - temple0))

    img_s = img.resize((int(iw * scale), int(ih * scale)))

    # rotate around nose
    new_mid = [int(c * scale) for c in mid]
    crop_box = (new_mid[0] - (OUT_W // 2),
                new_mid[1] - OUT_NOSE_TOP,
                new_mid[0] + (OUT_W // 2),
                new_mid[1] + (OUT_H - OUT_NOSE_TOP))

    img_out = img_s.rotate(tilt_deg, center=new_mid, resample=PImage.Resampling.BICUBIC).crop(crop_box).convert("L")
    out_images.append(gr.Image(img_out, visible=True))

  out_images += out_pad
  return out_images[:NUM_OUTS]


with gr.Blocks() as demo:
  gr.Markdown("""
              # 5020A Face Alignment Tool.
              ## Interface for face detection, alignment, cropping\
              to help create dataset for [WK11](https://github.com/PSAM-5020-2025S-A/WK11) / [HW11](https://github.com/PSAM-5020-2025S-A/Homework11).
              """)

  gr.Interface(
    face,
    inputs=gr.Image(type="pil"),
    outputs=all_outputs,
    cache_examples=True,
    examples=[["./imgs/03.webp"], ["./imgs/11.jpg"], ["./imgs/people.jpg"]],
    allow_flagging="never",
  )

if __name__ == "__main__":
   demo.launch()
