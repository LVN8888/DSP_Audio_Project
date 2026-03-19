# DSP Audio Project (SVM Only)

Dự án này dùng **SVM duy nhất** cho bài phân loại âm thanh với UrbanSound8K.  
Mọi phần CNN cũ đã được loại bỏ khỏi luồng sử dụng chính. Lệnh quan trọng nhất bây giờ là `train`, vì nó chạy toàn bộ pipeline trong **một lệnh**:

- sinh ảnh theo yêu cầu assignment
- chạy cross-validation
- export bảng kết quả
- lưu model SVM cuối để predict
- tạo manifest kiểm tra file đầu ra

## Cấu trúc thư mục

```text
DSP_Audio_Project/
├── analysis/
├── datasets/
├── experiments/
├── features/
├── models/
│   └── classical_models.py
├── preprocessing/
├── utils/
├── visualization/
├── main.py
├── requirements.txt
└── README.md
```

## Dataset layout

```text
data/
└── UrbanSound8K/
    ├── audio/
    │   ├── fold1/
    │   ├── fold2/
    │   └── ...
    └── metadata/
        └── UrbanSound8K.csv
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Lệnh sử dụng

### 1) Train full pipeline bằng một lệnh
Đây là lệnh chính để nộp bài.

```bash
python main.py train --out-dir outputs --pipeline both
```

Lệnh này sẽ:
- tạo ảnh trong `outputs/analysis_required_by_pdf/`
- tạo bảng kết quả trong `outputs/results/`
- tạo bảng so sánh Raw vs DSP trong `outputs/comparisons/`
- lưu model cuối ở `outputs/artifacts/dsp_svm.joblib`
- tạo manifest ở `outputs/submission/submission_manifest.csv`

### 2) Train chỉ DSP
Nếu bạn không cần so sánh Raw vs DSP:

```bash
python main.py train --out-dir outputs --pipeline dsp
```

### 3) Train toàn bộ dataset
Nếu muốn dùng full dataset, chỉ cần bỏ `--max-files`:

```bash
python main.py train --out-dir outputs --pipeline both
```

Nếu muốn giới hạn số file để chạy thử nhanh:

```bash
python main.py train --out-dir outputs --pipeline both --max-files 2000
```

### 4) Chỉnh số fold
```bash
python main.py train --out-dir outputs --pipeline both --kfolds 5
```

### 5) Fit riêng model cuối
Lệnh này là tùy chọn, chỉ dùng khi bạn muốn train riêng model deploy mà không chạy lại toàn bộ bảng và ảnh.

```bash
python main.py fit-final --pipeline dsp --out-dir outputs
```

### 6) Predict file mới
Sau khi train xong, test file mới bằng:

```bash
python main.py predict --file path/to/test.wav --pipeline dsp --artifact outputs/artifacts/dsp_svm.joblib
```

### 7) Phân tích DSP cho một file
```bash
python main.py analyze --file path/to/sample.wav --out-dir outputs/analysis_example
```

## Ý nghĩa các lệnh

### `train`
Lệnh all-in-one. Dùng để:
- làm report
- sinh hình
- xuất bảng
- lưu model cuối

### `fit-final`
Chỉ train model cuối. Không sinh đầy đủ bảng và ảnh như `train`.

### `predict`
Dùng model đã train để dự đoán một file âm thanh mới.

### `analyze`
Chạy phân tích tín hiệu DSP cho một file đơn lẻ.

## File đầu ra quan trọng

Sau khi chạy `train`, bạn sẽ thường dùng các file này:

```text
outputs/
├── analysis_required_by_pdf/
├── artifacts/
│   ├── dsp_svm.joblib
│   └── dsp_svm_metadata.json
├── comparisons/
├── results/
└── submission/
    ├── submission_manifest.csv
    └── submission_manifest.json
```

## Gợi ý dùng để nộp bài

Để ra đủ hình + bảng + model cuối, dùng:

```bash
python main.py train --out-dir outputs --pipeline both
```

Để demo predict:

```bash
python main.py predict --file path/to/test.wav --pipeline dsp --artifact outputs/artifacts/dsp_svm.joblib
```

## Ghi chú

- Project này là **SVM-only**
- Không cần dùng CNN
- `train` là lệnh chính thay cho `submit`
- Seed mặc định là `42`
