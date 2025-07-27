# 📦 Đồ án: Đếm và Hiển thị Số Lượng Vật Thể trên Ảnh bằng YOLOv5

## 🎓 Môn học: Nhập môn Xử lý ảnh số  
**Nhóm 10 – HK243**

---

## 📌 Mục tiêu đề tài

Ứng dụng mô hình học sâu YOLOv5 để:
- Phát hiện các vật thể trong ảnh đầu vào
- Đếm số lượng từng loại vật thể
- Ghi kết quả ra file `.txt`
- Hiển thị trực tiếp thông tin lên ảnh (label + bar chart trực quan)
- Lưu ảnh kết quả vào thư mục riêng

---

## 🛠️ Công nghệ sử dụng

| Thành phần       | Công cụ |
|------------------|--------|
| Mô hình AI       | `YOLOv5s` (qua thư viện `ultralytics`) |
| Xử lý ảnh        | `OpenCV` |
| Vẽ trực quan     | `cv2.putText`, `cv2.rectangle` |
| Log dữ liệu      | Ghi file `.txt` tự động |
| Thư viện hỗ trợ  | `supervision` để chuyển đổi output YOLO |

---

## 📂 Cấu trúc thư mục

├── count_objects.py # File chính chạy xử lý
├── data/
│ └── images/ # Thư mục chứa ảnh gốc
├── result_of_count_object/
│ ├── image1.jpg # Ảnh sau khi xử lý
│ ├── count_log.txt # Log kết quả đếm
│ └── contains_special/ # (nếu có) ảnh chứa vật thể đặc biệt
## ▶️ Hướng dẫn chạy

### 1. Cài đặt thư viện cần thiết:
```bash
pip install ultralytics supervision opencv-python
pip install -r requirements.txt

python count_objects.py
