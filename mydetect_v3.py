import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.plots import Colors, Annotator


def run(img, model, imgsz=640, conf_thres=0.5):
    device = select_device("")
    half = device.type != "cpu"  # half precision only supported on CUDA

    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Convert
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres)

    # Process detections
    im0 = img.clone().detach().cpu().numpy()
    im0 = (im0 * 255).astype("uint8")  # convert to 8-bit unsigned integer for OpenCV
    annotator = Annotator(im0)
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]} {conf:.2f}"
                annotator.box_label(xyxy, label=label, color=Colors()(int(cls), True))

    return annotator.im, pred
