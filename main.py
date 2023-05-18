import mydetect as MD
import sys
import cv2
import numpy as np

def main():
    input_video_path = "../test4.mp4"
    output_video_path = "./output.mp4"
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        sys.exit()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height)
    )
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        warped_frame = MD.run(frame)


        out.write(warped_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
