import cv2
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_sync
import mydetect as MD


def main(source="../test4.mp4", weights="../v8_m.pt", imgsz=640):
    device = select_device("")
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = (
        DetectMultiBackend(weights, device=device).half()
        if half
        else DetectMultiBackend(weights, device=device)
    )

    # Set Dataloader
    vid_path, vid_writer = None, None
    if isinstance(source, str):
        dataset = LoadImages(source, img_size=imgsz)
    else:
        dataset = LoadStreams(source, img_size=imgsz)

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0  # You may need to adjust this value depending on your video
    frame_size = (640, 480)  # You may need to adjust this value depending on your video
    out = cv2.VideoWriter("output.mp4", fourcc, fps, frame_size)

    # Run inference
    for path, img, im0s, vid_cap, s in dataset:
        # Preprocess the frame here if necessary

        # Pass the frame to mydetect.py for object detection
        result_frame, labels_info = MD.run(img, model=model)

        for i, det in enumerate(labels_info):
            if len(det):
                for *xyxy, conf, cls in det:
                    print(f"Object {i}:")
                    print(f"Coordinates: {xyxy}")
                    print(f"Confidence: {conf}")
                    print(f"Class: {cls}")

        # Write the result frame to the video file
        result_frame = cv2.cvtColor(
            result_frame, cv2.COLOR_RGB2BGR
        )  # convert color space from RGB to BGR
        out.write(result_frame.astype("uint8"))

    # Release the VideoWriter when done
    out.release()


if __name__ == "__main__":
    main()
