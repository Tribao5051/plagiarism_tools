# Hướng dẫn chi tiết hệ thống phát hiện đạo văn

## 1. Mục đích của tài liệu

Tài liệu này giải thích cấu trúc thư mục, luồng chạy của hệ thống, sáu task sửa
lỗi gần nhất, cách kiểm tra và các giới hạn vẫn còn. Mục tiêu là giúp đọc code,
chạy dự án và trình bày trước giảng viên mà không nhầm unit test với độ chính
xác thực tế.

## 2. Cấu trúc thư mục mới

```text
plagiarism_runtime/
|-- runtime/       Mã nguồn và file cấu hình cần để chạy hệ thống
|-- tests/         Unit test và regression test
|-- performance/   Script benchmark và kết quả benchmark
|-- docs/          Tài liệu hướng dẫn, giải thích và báo cáo dạng text/Markdown
|-- test_data/     Manifest, metadata và dữ liệu dành cho đánh giá
|-- faculty_indexes/  Vector DB và normalized text đang sử dụng
|-- project_cache/    Model cache của Hugging Face/BGE-M3
|-- venv/             Môi trường Python hiện tại
|-- AGENTS.md         Quy tắc làm việc với project
`-- .gitignore        Quy tắc không commit dữ liệu lớn hoặc thông tin nhạy cảm
```

Ba thư mục `faculty_indexes`, `project_cache` và `venv` được giữ nguyên vị trí.
Chúng có tổng dung lượng lớn, đang được config hiện tại sử dụng và việc di
chuyển không giúp cải thiện thuật toán. `project_config.py` xác định project root
là thư mục cha của `runtime`, vì vậy các path tương đối trong config vẫn trỏ về
đúng corpus, cache và vector DB cũ.

## 3. Các file runtime quan trọng

- `runtime/api_service.py`: FastAPI backend nhận PDF/TXT và faculty, chọn index,
  gọi detector và trả JSON.
- `runtime/detection_module_diff_p4.py`: retrieval, semantic matching, verbatim
  matching, merge và chọn evidence.
- `runtime/pdf_extract.py`: trích xuất PDF/TXT, giữ thứ tự và text phục vụ offset.
- `runtime/vector_db.py`: extract, chunk, embed BGE-M3 và ghi ChromaDB.
- `runtime/evidence_reranker.py`: tạo feature và xếp hạng evidence.
- `runtime/model_module.py`: singleton BGE-M3 dùng chung cho API.
- `runtime/chunking_config.py`: tạo Chonkie chunker từ config.
- `runtime/project_config.py`: đọc config và resolve tất cả path từ project root.
- `runtime/corpus_metadata.py`: chuẩn hóa faculty, metadata, checksum và source.
- `runtime/dspace_sync.py`: lấy metadata/bitstream và đồng bộ corpus DSpace.
- `runtime/dspace_vector_pipeline.py`: nối DSpace sync với vector indexing.
- `runtime/corpus_scheduler.py`: kiểm tra thay đổi và chạy lại index theo chu kỳ.
- `runtime/vector_db_readable.py`: phiên bản đơn giản để đọc flow vector DB.
- `runtime/plagiarism_config.toml`: model, chunking, index, detection, API và DSpace.

## 4. Luồng chạy của hệ thống

### 4.1. Chuẩn bị corpus và vector DB

1. `dspace_sync.py` duyệt community, collection, item, bundle và bitstream.
2. File được kiểm tra metadata/checksum trước khi tải hoặc cập nhật.
3. `pdf_extract.py` trích xuất và normalize nội dung.
4. Chonkie chia nội dung theo strategy trong `plagiarism_config.toml`.
5. BGE-M3 biến mỗi chunk thành embedding vector.
6. `vector_db.py` lưu vector, metadata nguồn và normalized text theo faculty.
7. Scheduler chỉ index lại faculty có file mới, sửa hoặc xóa.

### 4.2. Scan một submission

1. Frontend gửi file và faculty đến `/scan`.
2. API kiểm tra extension, kích thước và faculty hợp lệ.
3. PDF/TXT được extract thành nội dung có thể so sánh.
4. API chọn đúng TF-IDF index, Chroma collection và cache của faculty.
5. Retrieval lấy các source candidate trong đúng khoa.
6. Phase semantic tìm nội dung cùng nghĩa hoặc paraphrase.
7. Phase verbatim dùng fingerprint để tìm phần sao chép chính xác.
8. `merge_evidence` chỉ gộp hai phase khi cùng source và cùng vùng ở cả hai văn
   bản.
9. Evidence được consolidate/rerank để giảm trùng và ưu tiên bằng chứng mạnh.
10. API trả phần trăm, evidence text, source, offset, loại evidence và performance.

## 5. Chi tiết sáu task đã sửa

### Task 1 - Chặn oracle candidate sources

**Trước khi sửa:** client có thể truyền `candidate_sources` để bỏ qua retrieval.
Danh sách rỗng hoặc source không tồn tại bị lọc âm thầm, làm benchmark có thể
đánh giá một flow khác với scan thực tế.

**Nguyên nhân:** API chưa phân biệt tính năng oracle dành cho benchmark với
request production.

**Sau khi sửa:** `allow_oracle_candidate_sources` mặc định là `false`. API trả
403 khi oracle tắt, trả 400 cho danh sách rỗng/không hợp lệ, và chỉ forward source
hợp lệ khi bật. Request bình thường không truyền field này vẫn giữ nguyên.

**Commit:** `0837a3f`.

### Task 2 - Tách semantic/fingerprint cache theo khoa

**Trước khi sửa:** tên file có thể được dùng như định danh toàn hệ thống. Hai
khoa cùng có `paper.pdf` có nguy cơ dùng nhầm cached text; source được cập nhật
cũng có thể trả nội dung cũ.

**Nguyên nhân:** cache thiếu faculty/index namespace và phiên bản nội dung.

**Sau khi sửa:** mỗi faculty index sở hữu semantic cache và fingerprint cache
riêng. Namespace gồm faculty, DB path và index version. Mỗi source có SHA-256
version; thay đổi source hoặc index làm cache tương ứng hết hiệu lực.

**Commit:** `0837a3f`.

### Task 3 - Giữ nội dung trước References

**Trước khi sửa:** gặp `References` trên một trang có thể làm mất cả phần nội
dung hợp lệ nằm phía trên tiêu đề.

**Nguyên nhân:** rule cũ loại theo trang, không cắt theo vị trí heading.

**Sau khi sửa:** chỉ heading References đứng riêng mới được xem là điểm kết thúc.
Phần trước heading trên cùng trang được giữ, trang chỉ có References bị loại và
từ `references` trong câu bình thường không làm cắt tài liệu.

**Commit:** `0837a3f`.

### Task 4 - Sửa fingerprint lệch vị trí

**Trước khi sửa:** fingerprint dùng `step=5`. Chỉ cần thêm 1-4 ký tự ở đầu có
thể làm toàn bộ cửa sổ lệch pha. Hash chỉ giữ một occurrence và chưa kiểm chứng
nội dung sau khi hash trùng.

**Nguyên nhân:** stride lớn tạo alignment dependency; hash bị xem như bằng chứng
thay vì chỉ là bộ lọc ứng viên; normalized offset chưa map đủ về original text.

**Sau khi sửa:** mặc định `step=1`, mọi hash lưu tất cả occurrence, hash hit phải
được xác minh bằng nội dung và offset được map về text gốc. Whitespace thay đổi
không còn làm highlight sai, đoạn lặp có thể trả đúng vị trí.

**Chi phí:** thử nghiệm 200.000 ký tự cho thấy `step=1` tạo gần 5 lần số cửa sổ,
dùng khoảng 4,9 lần RSS và mất khoảng 3,6 lần wall time so với `step=5`. Đây là
bản correctness-first; Winnowing là hướng tối ưu sau khi có benchmark đại diện.

**Commit:** `0837a3f`.

### Task 5 - Không gộp nhầm evidence

**Trước khi sửa:** semantic và verbatim evidence có thể bị gộp vì gần nhau trong
submission, dù source offset nằm ở hai vùng hoàn toàn khác nhau. Text của một
span đôi khi bị gắn với offset của span khác.

**Nguyên nhân:** merge chỉ dựa nhiều vào student position/text similarity, chưa
yêu cầu source span cùng vùng.

**Sau khi sửa:** chỉ merge khi cùng source và span giao nhau ở cả submission lẫn
source. Evidence không đạt điều kiện được giữ riêng cho bước consolidate.
`student_match_text` và `source_match_text` luôn đi cùng offset tương ứng.

**Commit:** `2af12c3`.

### Task 6 - Sửa nhãn và metric của reranker

**Trước khi sửa:** evidence có thể là positive nếu student passage khớp ground
truth dù source passage sai. Tên `precision` không cho biết đang đo evidence hay
ký tự.

**Nguyên nhân:** source score chỉ được dùng cộng điểm, không phải điều kiện bắt
buộc. Benchmark và evaluator có thể dùng hai cách chấm khác nhau.

**Sau khi sửa:** positive yêu cầu cả student passage và source passage vượt cùng
threshold của evaluator. `evidence_hit_precision` là số evidence đúng chia tổng
evidence được chọn. `precision` được giữ làm compatibility alias.
`character_precision` hiện là `None`, vì XML offset thuộc raw PAN text trong khi
tool đang có normalized offset; cần mapping raw-to-normalized trước khi chấm.
Split validator cấm trùng train/validation/test. Chỉ train và validation được
dùng tune; test chỉ dùng một lần để báo cáo cuối.

**Commit:** `f895a26`.

## 6. Cách chạy sau khi sắp xếp

Mở CMD:

```bat
cd /d D:\hochanh\hk252\plagiarism_runtime
runtime\setup_runtime.cmd
runtime\run_api.cmd
```

Đồng bộ DSpace và index một lần:

```bat
runtime\run_dspace_once.cmd
```

Chạy scheduler 30 ngày:

```bat
runtime\run_monthly_dspace_sync.cmd
```

Chạy toàn bộ test không cần DSpace thật:

```bat
tests\run_all_tests.cmd
```

Chạy benchmark PDF:

```bat
venv\Scripts\python.exe performance\benchmark_pdf_extraction.py --sample-size 15 --repeats 2
```

Kết quả benchmark mặc định được ghi trong `performance/results/` và không được
Git track.

## 7. Ý nghĩa các phép đo

- Evidence-hit precision: trong tất cả evidence tool hiển thị, bao nhiêu evidence
  khớp đúng cả vùng submission và vùng source của ground truth.
- Recall: trong tất cả vùng đạo văn ground truth, tool bắt được bao nhiêu vùng.
- F1: trung bình điều hòa giữa evidence-hit precision và recall.
- Character precision: tỷ lệ ký tự tool đánh dấu thực sự nằm trong vùng ground
  truth. Chưa chấm cho tới khi có mapping offset về raw text.
- API latency: thời gian từ lúc API nhận request đến lúc trả JSON.
- Phase timing: thời gian extraction, retrieval, semantic, verbatim, merge và
  evidence selection; dùng để xác định bottleneck nhưng không chứng minh accuracy.

## 8. Cách biết hệ thống hoạt động đúng

1. Syntax/import test phải qua sau mỗi thay đổi cấu trúc.
2. Regression test phải tái hiện lỗi cũ và đạt sau khi sửa.
3. API smoke test phải load đúng faculty index và giữ response schema.
4. Benchmark phải dùng cùng submission/source/split và SHA-256 đầu vào.
5. Accuracy chỉ được kết luận từ tập test giữ riêng, không từ unit test hoặc tập
   đã dùng tune.

Sau Task 6, bộ test liên quan có 53 test đạt. Sau khi tổ chức lại thư mục và thêm
regression test cho project-root path, toàn bộ 54 test tiếp tục đạt bằng cả
Python trực tiếp và `tests\\run_all_tests.cmd`.

## 9. Giới hạn còn lại

- Chưa có raw-text offset mapping nên chưa có character precision đáng tin cậy.
- Kết quả benchmark cũ phải chạy lại vì Task 6 thay đổi cách tạo positive label.
- `step=1` tăng độ đúng Phase 4 nhưng có thể tốn RAM/thời gian trên file dài.
- Unit test chứng minh lỗi logic đã được chặn, không chứng minh accuracy tăng.
- Cần benchmark PAN held-out và sau đó thêm dataset khác để đánh giá khả năng
  tổng quát hóa.

## 10. Mốc Git để rollback

- `0837a3f`: Task 1-4.
- `2af12c3`: Task 5.
- `f895a26`: Task 6.
- Commit tổ chức thư mục được tạo sau khi toàn bộ test đạt.
