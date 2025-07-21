import os
from ultralytics import YOLO
import supervision as sv
import cv2
from collections import Counter

# Load mô hình YOLOv5
model = YOLO('yolov5s.pt')

# Thư mục chứa ảnh gốc
input_folder = r'D:\Hoc\NhapMonXuLyAnhSo\DoAn\Nhom10_XLAS_HK243_yolov5-master\data\images'

# Thư mục lưu kết quả
output_folder = r'D:\Hoc\NhapMonXuLyAnhSo\DoAn\Nhom10_XLAS_HK243_yolov5-master\result_of_count_object'
os.makedirs(output_folder, exist_ok=True)

# Xóa nội dung cũ của file log/ count_log.txt (nếu có)
open(os.path.join(output_folder, "count_log.txt"), "w").close()

# Xử lý từng ảnh
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)

        # Dự đoán
        results = model(image_path)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Đếm số lượng theo lớp
        class_counts = Counter(detections.class_id)
        names = model.model.names

        # ======== Ghi ra file txt ==========
        # Đếm số lượng theo lớp
        class_counts = Counter(detections.class_id)
        names = model.model.names

        # Chuỗi hiển thị
        text_lines = [f"{count} {names[class_id]}s" for class_id, count in class_counts.items()]
        summary_text = ", ".join(text_lines)

        # Ghi log .txt
        log_file_path = os.path.join(output_folder, "count_log.txt")
        line_items = [f"{names[class_id]}={count}" for class_id, count in class_counts.items()]
        log_line = f"{filename}: " + ", ".join(line_items)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
        # ======== Ghi ra file txt ==========

        # Tạo chuỗi tóm tắt số lượng
        text_lines = [f"{count} {names[class_id]}s" for class_id, count in class_counts.items()]
        summary_text = ", ".join(text_lines)

        # Lấy ảnh có bounding box
        image_with_boxes = results.plot()

        # Vẽ chuỗi lên ảnh
        cv2.putText(image_with_boxes, summary_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


        # ========== THÊM BAR CHART NHỎ ==========

        # Vị trí bắt đầu vẽ chart
        chart_x = 30
        chart_y = 80
        bar_height = 25
        spacing = 10
        max_bar_width = 200

        # Tìm số lượng lớn nhất để scale độ dài thanh
        max_count = max(class_counts.values()) if class_counts else 1

        for idx, (class_id, count) in enumerate(class_counts.items()):
            class_name = names[class_id]
            bar_width = int((count / max_count) * max_bar_width)

            # Tính tọa độ từng thanh
            top_left = (chart_x, chart_y + idx * (bar_height + spacing))
            bottom_right = (chart_x + bar_width, top_left[1] + bar_height)

            # Vẽ khung màu xanh lá
            cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), -1)

            # Ghi tên lớp và số lượng bên trái thanh
            label_text = f"{class_name}: {count}"
            cv2.putText(image_with_boxes, label_text, (chart_x + bar_width + 10, top_left[1] + bar_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # ========== THÊM BAR CHART NHỎ ==========

        # Lưu ảnh kết quả
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image_with_boxes)

print("Done!! Ảnh được lưu vào:", output_folder)
