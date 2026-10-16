import cv2
import argparse
from ultralytics import YOLO


def main():
    # 1. 设置参数（默认使用 best.pt 和 test.mp4）
    parser = argparse.ArgumentParser(description="YOLO 装甲板推理与关键点可视化")
    parser.add_argument('--model', type=str, default='best.pt', help='模型权重路径')
    parser.add_argument('--video', type=str, default='test.mp4', help='输入视频路径')
    parser.add_argument('--output', type=str, default='result.mp4', help='输出视频路径')
    args = parser.parse_args()

    # 2. 加载模型
    print(f"正在加载模型: {args.model}")
    model = YOLO(args.model)

    # 3. 读取视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {args.video}，请检查文件名是否为 test.mp4！")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("开始推理，请稍等...")
    # 4. 逐帧推理
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.5, verbose=False)
        result = results[0]

        # 画检测框和标签
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 画4个关键点 (红点)
            if result.keypoints is not None and result.keypoints.xy is not None:
                for kpt in result.keypoints.xy:
                    for i, point in enumerate(kpt):
                        px, py = int(point[0]), int(point[1])
                        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                        cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"推理完成！结果视频已保存为: {args.output}")


if __name__ == '__main__':
    main()