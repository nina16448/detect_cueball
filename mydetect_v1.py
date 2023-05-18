import torch
from pathlib import Path
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
import cv2


def run(weights="../v8_m.pt", source="../test4.mp4", conf_thres=0.5):
    device = select_device("")
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    dataset = LoadImages(source, img_size=(640, 640), stride=stride)
    vid_path, vid_writer = None, None

    for path, img, im0s, vid_cap, s in dataset:  # add 's' here
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres)

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator = Annotator(im0s, line_width=3, example=str(names))
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
                    print(f"{label} {xyxy}")

                im0s = annotator.result()

                if vid_path != path:  # new video
                    vid_path = path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(
                        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )

                vid_writer.write(im0s)

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()  # release final video writer


if __name__ == "__main__":
    run()
