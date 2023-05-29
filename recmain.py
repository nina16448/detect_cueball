import edge_detect as ED
import Transformation as TS
import sys
import cv2
import numpy as np
import torch


def main():
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    model = torch.hub.load("ultralytics/yolov5", "custom", path="../v8_m.pt")
    # 設定 IoU 門檻值
    model.iou = 0.9

    # 設定信心門檻值
    model.conf = 0.9

    input_video_path = "../test5.mp4"
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        sys.exit()
    flag = False
    intersections = np.zeros((4, 2), dtype=np.float32)
    counter = 0
    while True:
        ret, frame = cap.read()

        height, width, channels = frame.shape

        if not ret:
            break  # 如果讀不到幀，退出循環

        # 對每一幀執行邊緣提取和透視變換
        if flag == False:
            warped_frame, intersections, flag = TS.prev_exe(frame)
        else:
            warped_frame = TS.exe_pic(frame, intersections)
            warped_frame = cv2.filter2D(warped_frame, -1, kernel)

        warped_frame = cv2.resize(
            warped_frame, (640, 640), interpolation=cv2.INTER_AREA
        )

        # 將像素值範圍調整為 [0, 1]
        warped_frame = warped_frame / 255.0

        # 將圖像的維度從 [height, width, channels] 調整為 [1, channels, height, width]
        warped_frame = np.transpose(warped_frame, (2, 0, 1))
        warped_frame = np.expand_dims(warped_frame, axis=0)

        # 確保 warped_frame 是 float32 類型
        warped_frame = warped_frame.astype(np.float32)

        # 將 warped_frame 轉換為 PyTorch tensor
        warped_frame = torch.from_numpy(warped_frame)

        results = model(warped_frame)

        # 如果有檢測到物體，則繪製檢測結果並顯示圖像
        if results[0].shape[0] > 0:
            # 將 warped_frame 轉換為 OpenCV 圖像
            warped_frame = warped_frame.permute(0, 2, 3, 1).squeeze().cpu().numpy()
            warped_frame = (warped_frame * 255).astype(np.uint8)
            for *xyxy, conf, cls in results[0]:
                # 繪製邊界框
                cv2.rectangle(
                    warped_frame,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    (255, 0, 0),
                    2,
                )
                # 繪製類別標籤和信心度
                cv2.putText(
                    warped_frame,
                    f"{int(cls)} {conf:.2f}",
                    (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        warped_frame = cv2.resize(
            warped_frame, (1280, 640), interpolation=cv2.INTER_AREA
        )

        # 顯示變換後的幀
        cv2.imshow("Warped Frame", warped_frame)
        print(results)
        # 等待 1 毫秒，並檢查是否有按下 q 鍵，如果有，則退出循環
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        print(counter)
        counter += 1

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
