import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.plots import Annotator, colors
from pathlib import Path


def run(frame, weights="../v8_m.pt", conf_thres=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names

    # Prepare frame
    img = torch.from_numpy(frame).to(device).permute(2, 0, 1)  # HWC to CHW
    img = img.half() if img.is_floating_point() else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Inference
    pred = model(img)

    # NMS
    pred = non_max_suppression(pred, conf_thres)

    # Process predictions
    det = pred[0]  # per image
    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(frame, line_width=3, example=str(names))

    labels_info = []
    if len(det):
        # Rescale boxes from img_size to frame size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = f"{names[c]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(c, True))
            xywh = (
                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            )  # normalized xywh
            labels_info.append((cls, *xywh, conf))

    # Stream results
    frame = annotator.result()

    return frame, labels_info
