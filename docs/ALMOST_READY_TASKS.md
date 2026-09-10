# Almost Ready: mốc ổn định và các task tiếp theo

Cập nhật: 10/09/2026. Thư mục làm việc: `D:\hochanh\hk252\plagiarism_runtime`.

**Đây là tài liệu giao việc, không phải danh sách các cải tiến đã triển khai.**
Chỉ thực hiện task có mã `Nxx` mà người dùng giao. Các task 1-9 của đợt trước
đã nằm trong lịch sử Git; không làm lại toàn bộ các task đó.

## 1. Mốc cần giữ khi sắp hết thời gian

Commit mới có message chính xác `almost ready`. Mã production của mốc này
giữ nguyên so với commit cha `33741fc` (`fix: harden pagination and lifecycle rollback`).
Commit mới bổ sung hướng dẫn, script benchmark oracle và các file baseline nhỏ.
Không tuyên bố rằng commit tài liệu này làm precision hoặc recall tăng lên.

Mốc này phù hợp để giữ bản chạy được cho demo và làm baseline so sánh. Nó vẫn có
các hạn chế accuracy mô tả bên dưới; tên commit không có nghĩa hệ thống đã đạt
chất lượng production hoặc đã vượt mọi kiểm thử.

Git không lưu `venv`, model cache, corpus, normalized TXT hoặc vector DB lớn.
Các dữ liệu ấy phải được giữ riêng. Không xóa index hiện có khi đổi branch.
Index đang dùng: `bge_m3_semantic_potion384_v1`, khoa `pan-evaluation`, 614 nguồn,
187.483 vector. Nếu code/config tương lai không tương thích index, phải báo rõ;
không tự rebuild hoặc bỏ qua kiểm tra hash.

Các file được đóng gói cùng mốc:

- `docs/ALMOST_READY_TASKS.md`: hướng dẫn giao từng task cho model khác.
- `docs/ORACLE_NOISE_BASELINE.txt`: nhận xét chi tiết, cách chấm và giới hạn.
- `performance/run_oracle_noise.py`: bản script đã dùng cho 50 request oracle.
- `test_data/oracle_noise_baseline/manifest.json`: bài nộp, nguồn XML/nhiễu và SHA-256.
- `test_data/oracle_noise_baseline/summary.json`: kết quả tổng hợp.
- `test_data/oracle_noise_baseline/per_case.csv`: số liệu từng bài/điều kiện.
- `test_data/oracle_noise_baseline/per_case.json`: số liệu từng phase và các XML ID bị mất.
- `test_data/oracle_noise_baseline/run_environment.json`: môi trường của lượt đo.

Response đầy đủ, mẫu tải máy và HTML vẫn ở:
`D:\hochanh\hk252\plagiarism_performance\oracle_noise_pan18_20260909\results`.
Không đưa chúng hoặc các PDF/TXT corpus vào Git chỉ để tạo checkpoint.

## 2. Hiểu đúng baseline trước khi sửa

50/50 request oracle hoàn tất, một request tại một thời điểm. So sánh cùng 14 bài
có đạo văn, chứa 117 đoạn XML; thêm 4 bài âm tính để kiểm tra cảnh báo sai.

| Điều kiện | Evidence đúng/tổng | Precision evidence-hit | Recall theo đoạn | API giây/bài, 14 bài dương tính |
|---|---:|---:|---:|---:|
| Oracle thuần | 76/78 | 97,44% | 69/117 = 58,97% | 6,00 |
| Oracle +5 nhiễu | 76/83 | 91,57% | 69/117 = 58,97% | 10,86 |
| Oracle +10 nhiễu | 76/84 | 90,48% | 69/117 = 58,97% | 19,50 |

Cả +5 và +10 đều cảnh báo 2/4 bài âm tính. Tính đủ 18 bài, precision lần lượt
là 89,41% và 88,37%. Không chỉ báo điểm trên bài dương tính rồi bỏ qua bài âm tính.

Ở cả ba điều kiện oracle: evidence thô bắt 114/117 đoạn; sau consolidate vẫn
114/117; sau chọn hiển thị còn 69/117. **45 đoạn có bằng chứng hỗ trợ bị mất ở
Phase 7.** Cấu hình `agreement.max_per_source` thực tế là **4**, không phải giá
trị mặc định 8 viết trong Python. Ngưỡng chọn agreement hiện là 0,40.

Precision mới tính mọi evidence khớp ít nhất một cặp XML ở cả hai phía. Báo cáo
cũ từng chỉ ghi công evidence được chọn cho đoạn XML. Chấm lại cùng JSON cũ theo
cách mới ra 86,25% thay vì 70%; đó là khác biệt phép chấm, không phải nâng cấp model.
Không đổi định nghĩa metric giữa hai nhánh để làm điểm trông cao hơn.

Phép chấm này dùng containment nội dung và chưa có mapping offset raw. Nó không
đo mọi ký tự highlight có đúng hay không, cũng không phải PAN plagdet chính thức.
Nhiều evidence lặp vẫn có thể được tính đúng; phải báo thêm mức trùng lặp và độ phủ.

Cache chạy tự nhiên; máy không được cô lập khỏi ứng dụng khác. Thời gian không
phải cold/warm có kiểm soát. Build index có thể ghi là dự trù khoảng 90 phút khi
đa nhiệm; không sửa số thực đo thành 90 phút. Chưa cần tối ưu build để xử lý các
vấn đề Phase 7, vì build là tác vụ định kỳ.

## 3. Prompt chung để giao cho model khác

Dán đoạn dưới cùng với phần mô tả MỘT task cụ thể:

```text
Dự án: plagiarism detection.
Workspace: D:\hochanh\hk252\plagiarism_runtime.
Đọc AGENTS.md và docs\ALMOST_READY_TASKS.md, rồi đọc đúng các hàm liên quan.
Chỉ làm task Nxx được giao; không tự thực hiện các task tiếp theo.
Không sửa D:\hochanh\hk252\doan\project. Được đọc dữ liệu benchmark ở đó.
Nếu prompt này chưa cấp quyền sửa file, hỏi quyền theo AGENTS.md trước khi ghi.
Không thay embedding model, Chonkie, threshold, chunking hoặc API response fields.
Không xóa/rebuild corpus hay vector DB thật. Không reset Git hoặc ghi đè thay đổi sẵn có.
Với lỗi hành vi: viết regression test bằng dữ liệu giả lập trước, chứng minh test
thất bại vì đúng lỗi, rồi mới sửa và chạy lại. Unit test không cần GPU/DSpace thật.
Không đưa XML/ground truth vào thuật toán production hoặc dùng nhãn để chọn evidence.
Chỉ tune trên train/validation mới được chỉ định; không tune bằng tập test cuối.
Giữ evidence text và offset thuộc đúng cặp submission/source; không sửa offset
để khớp XML, không mở rộng highlight tùy ý để tăng recall.
Mọi kết quả performance ghi vào một thư mục MỚI ở plagiarism_performance.
Không ghi đè baseline, không đổi metric cũ dưới cùng một tên.
Cuối task: báo file sửa, nguyên nhân, thay đổi hành vi, test, giới hạn và lệnh tái lập.
Nếu được yêu cầu commit, chỉ stage file thuộc task này; không dùng git add .
```

Lệnh kiểm tra chung từ CMD ở runtime:

```cmd
cd /d D:\hochanh\hk252\plagiarism_runtime
git status --short
venv\Scripts\python.exe -B -X utf8 -m unittest tests.test_evidence_selection tests.test_benchmark_evidence_rerankers -v
```

Thêm test module mới vào lệnh nếu task tạo test riêng. Khi sửa code dùng chung,
chạy thêm `tests\run_all_tests.cmd`. Không cần chạy benchmark GPU sau mọi thay
đổi tài liệu; chỉ chạy khi hành vi hoặc phép đo liên quan thực sự thay đổi.

## 4. Thứ tự và mức ưu tiên

| Task | Việc cần làm | Phụ thuộc | Mục tiêu |
|---|---|---|---|
| N00 | Cố định phép chấm và khả năng chạy lại baseline | Không | So sánh đúng |
| N01 | Truy vết từng evidence bị loại ở Phase 7 | N00 | Xác định đúng nguyên nhân |
| N02 | Bỏ cảnh báo từ boilerplate chắc chắn | N01 | Tăng precision ít rủi ro |
| N03 | Khử trùng phải kiểm tra cả hai span | N01 | Không bỏ nhầm evidence |
| N04 | Chọn theo độ phủ và thử giới hạn mỗi nguồn offline | N03 | Tăng recall có kiểm soát |
| N05 | Kiểm tra semantic sai và tài liệu tham khảo | N01, N02 | Giảm false positive khó |
| N06 | Giới hạn cache theo bộ nhớ và đo fingerprint | Không; nên sau N04 | Giảm RAM/thời gian API |
| N07 | Chấm độ phủ ký tự trên TXT có mapping raw | N00 | Đo chất lượng highlight |
| N08 | Mở rộng bộ đánh giá và giữ tập test mới | N00; trước tune lớn | Tránh tối ưu theo vài bài |
| N09 | Chạy đối chứng và quyết định dùng bản mới | Các task đã chọn | Chốt bản nộp |

Nếu sắp hết thời gian: giữ mốc almost ready; ưu tiên N00-N03 và kiểm tra N09.
Không cần triển khai đủ mọi ý tưởng mới có thể giữ một bản nộp ổn định.
N04/N05 chưa được phép tự thay config production hoặc đổi model.

## N00. Cố định phép chấm, tách chấm điểm khỏi model

**Prompt giao task:**
"Giữ nguyên định nghĩa chấm hai phía của baseline oracle. Tách phần chấm thuần
thành helper trong performance nếu cần, để chấm JSON đã lưu không cần API/GPU.
Không đổi threshold hay cách tính positive label dùng để huấn luyện reranker."

Đọc: `performance/run_oracle_noise.py` (load_scoring, records_for, metrics),
`performance/benchmark_evidence_rerankers.py` (prediction_text,
annotate_truth_matches, evidence_hit_metrics_from_predictions,
assign_best_candidate_labels), `tests/test_benchmark_evidence_rerankers.py`.
Baseline script còn đọc helper từ evaluator ngoài runtime; N00 được đưa những
helper thuần cần thiết vào performance, nhưng không sửa bản ngoài runtime.

Phạm vi: các file performance trên, helper mới nếu cần và test tương ứng.
Không sửa detector, API, PDF extraction hay vector_db.

Cách làm:

1. Viết test ba cặp: đúng hai phía; đúng submission nhưng sai file nguồn; đúng
   file nguồn nhưng sai đoạn nguồn. Chỉ cặp đầu được hit.
2. Test hai evidence cùng khớp một XML: evidence-hit có thể là 2/2 nhưng recall
   chỉ tăng một đoạn. Không nhầm với nhãn best-candidate của training.
3. Test XML rỗng: recall N/A, có evidence là cảnh báo trên bài âm tính.
4. Báo riêng raw/consolidated/display, semantic/verbatim/mixed; cùng metric
   cho mọi điều kiện. Character precision tiếp tục null cho đến N07.
5. Lưu phiên bản schema của phép chấm và hash mã chấm trong output.
6. Re-score 50 JSON cũ, so từng chỉ số với baseline trước/sau refactor.

Test đạt: số đếm của 50 JSON không đổi; sai nguồn/sai đoạn nguồn không là hit;
chạy offline không khởi động model. Bất kỳ khác biệt nào phải giải thích trước
khi coi task đạt. Không sửa baseline để khớp kết quả mới.

## N01. Truy vết lý do evidence bị loại

**Prompt giao task:**
"Thêm khả năng trace lựa chọn Phase 7 dành riêng cho benchmark offline. Ghi
từng ứng viên, điểm, span và lý do giữ/loại; mặc định production không trace.
Lập bảng XML ID mất ở khâu nào, dùng ground truth chỉ trong evaluator."

Đọc `runtime/detection_module_diff_p4.py`: select_display_evidence,
_assign_agreement_scores và hàm reject nội bộ; đọc `runtime/evidence_reranker.py`.
Input chính là `raw_json/oracle_10/*.json` của baseline, nhất là 00320.

Phạm vi: detector chỉ thêm đường trace không đổi mặc định; script offline trong
performance; `tests/test_evidence_selection.py` hoặc test trace mới.
Không thêm field vào response API, không gọi API thật chỉ để lấy lại dữ liệu.

Cách làm:

1. Tạo khóa ứng viên ổn định từ namespace bài/khoa, nguồn, type, cả hai span và
   thứ tự ứng viên để các bản lặp nguyên văn không trùng ID.
2. Replay select_display_evidence từ consolidated_evidence đã lưu với đúng config.
3. Ghi score, rank, lý do rejection thực tế, ID evidence giữ lại nếu bị coi là
   duplicate/competition, số evidence nguồn đã chọn lúc chạm cap.
4. Ở evaluator, gắn mỗi ứng viên với XML ID bằng N00; tìm đủ các ứng viên hỗ trợ
   từng đoạn bị mất. Một XML có thể bị mất vì nhiều nguyên nhân khác nhau.
5. Báo số ứng viên và số XML riêng; không gọi 1.172 ứng viên bị cap là 1.172 đoạn.
6. Khi trace tắt, không cấp phát danh sách/log trace và không thay API output.

Test đạt: bật/tắt trace chọn cùng evidence theo cùng thứ tự; mỗi ứng viên có
trạng thái cuối; các nguyên nhân khớp tổng rejection counters. Báo được phân bố
lý do cho 45 XML mất ở Phase 7 mà không cho ground truth đi vào production.

## N02. Chặn cảnh báo từ boilerplate chắc chắn

**Prompt giao task:**
"Dùng trace để loại khỏi phần tính đạo văn các evidence chỉ chứa boilerplate
đã nhận diện chắc chắn, trước bước chọn hiển thị. Giữ nội dung và offset gốc;
không sửa extraction, không xóa văn bản trong corpus."

Ví dụ cần tái hiện: bài 00322 trùng dòng ghi công Online Distributed Proofreading
Team. Đây là exact match thật nhưng không phải đoạn đạo văn được XML gán nhãn.

Phạm vi: helper nhỏ trong detector hoặc module policy riêng nếu có lý do rõ;
đường chọn evidence, tests; benchmark offline. Không sửa pdf_extract/vector_db.

Cách làm:

1. Tạo fixture ngắn về dòng ghi công và một đoạn nội dung thật có cụm từ giống nó.
2. Chỉ loại evidence khớp mẫu boilerplate rõ và không có nội dung chính khác.
   Không lọc mọi câu có tên người, tên sách, từ references hoặc Gutenberg.
3. Kiểm tra cả text match ở submission và source; không dùng expanded context
   dài để suy rằng toàn bộ evidence là boilerplate.
4. Giữ raw/consolidated cho audit. Ghi lý do chi tiết trong trace N01; không đổi
   schema response. Dùng counter hiện có nếu cần tương thích API.
5. Re-score cùng JSON baseline, báo FP giảm và kiểm tra có XML đúng nào bị mất.

Test đạt: fixture ghi công không còn được cảnh báo; đoạn văn nội dung thật vẫn
giữ; offset/text không đổi; không làm mất XML đúng trong tập validation đã chọn.
Không tự xử lý mọi danh mục tham khảo ở task này; trường hợp mơ hồ thuộc N05.

## N03. Khử trùng phải xét cả submission và source

**Prompt giao task:**
"Sửa riêng duplicate suppression của Phase 7 để không loại hai evidence chỉ
vì chúng gần nhau trong submission. Cần cùng nguồn và cùng vùng ở cả hai phía.
Không thay merge_evidence hoặc threshold của model."

Đọc `_same_source_evidence_is_duplicate` và `select_display_evidence` trong
`runtime/detection_module_diff_p4.py`. Hàm duplicate hiện chỉ lấy student span;
không nhầm nó với merge_evidence đã sửa trong Task 5 của đợt trước.

Phạm vi: hai hàm trên và `tests/test_evidence_selection.py`.

Cách làm:

1. Viết regression fixture cùng source: student spans giao/ở gần nhau nhưng
   source spans 100-200 và 9000-9100. Hai evidence không được gọi là duplicate.
2. Fixture thứ hai giao ở cả hai phía: giữ bản tốt hơn theo thứ tự điểm hiện có.
3. Fixture hai vùng student độc lập trích cùng source region: giữ vị trí lặp
   trong bài nộp, không chỉ lấy lần xuất hiện đầu.
4. Chỉ xét duplicate khi có giao thật ở cả hai phía; khoảng cách gần nhau một
   mình không đủ. Dùng helper interval sẵn có, quy ước [start, end).
5. Không lấy text của evidence A gắn offset B; lấy nguyên evidence được giữ.
6. Replay trên JSON đã lưu, báo XML khôi phục, FP mới và độ trùng evidence.

Test đạt: các fixture trên đúng, edge-touch không là overlap, lặp test cho
cùng output. Test pipeline merge/fingerprint trước đây vẫn đạt. Unit pass không
được dùng để tự khẳng định precision tăng.

## N04. Chọn evidence theo độ phủ; thử cap offline

**Prompt giao task:**
"Sau N03, thử cách chọn evidence có thêm vùng nội dung mới thay vì để nhiều
ứng viên cùng vùng chiếm suất. So sánh offline với cap=4 hiện tại. Giữ nguyên
agreement score, semantic thresholds và API config cho đến khi được duyệt."

Phạm vi: script thí nghiệm performance, test selection; chỉ thêm thuật toán
thử nghiệm có cờ tắt mặc định trong runtime nếu cần. Không đổi API mặc định.

Cách làm:

1. Dùng consolidated candidates và score hiện tại, không embed lại.
2. Khử trùng dựa cả hai span trước khi phân bổ suất theo nguồn.
3. Đánh giá phần vùng match mới mà một candidate bổ sung, không dùng text context
   mở rộng để thổi phồng độ phủ. Duy trì các khoảng được chọn theo từng nguồn.
4. Thử riêng cap 4/8/12 trong bản copy cấu hình offline; không sửa TOML production
   và không đồng thời đổi threshold, thuật toán điểm hoặc chunking.
5. Có thể thử phân bổ theo vùng, nhưng không ghép hai đoạn xa bằng một span bao
   trùm để làm XML recall trông cao. Selected text/offset giữ nguyên.
6. Tune chỉ trên train/validation được chỉ định. Bộ 18 cũ dùng chẩn đoán; không
   dùng 4 bài test cũ để chọn cap rồi gọi điểm ấy là kiểm thử độc lập.
7. Báo precision, recall, FP trên bài âm tính, số evidence và thời gian; có cả
   cấu hình thua. Bản tốt nhất phải có bằng chứng hơn baseline trên cùng phép chấm.

Test đạt: evidence lặp không chiếm hết suất; hai vùng độc lập được giữ; cap vẫn
được tuân thủ; không tăng điểm chỉ vì nối khoảng trống. Quyết định dùng cấu hình
mới thuộc N09 sau khi có số liệu, không tự deploy khi unit test đạt.

## N05. Semantic sai và bằng chứng trong tài liệu tham khảo

**Prompt giao task:**
"Phân loại các false positive còn lại sau N02 bằng cặp text cụ thể. Đánh giá
cách kiểm chứng semantic cuối pipeline, giữ nguyên các phase tạo evidence.
Không mặc định tăng threshold toàn cục hoặc thay embedding model."

Đọc `agreement_score`, `evidence_feature_vector`, `_evidence_confidence`,
`_passes_evidence_profile`; xem các nguồn nhiễu của 00241 và các ca tương tự.

Phạm vi đầu tiên: script phân tích performance, fixture tests và tài liệu nhãn.
Không sửa scoring production trước khi người dùng duyệt phương án có số liệu.

Cách làm:

1. Tách nhóm: boilerplate; cùng tên tài liệu/tham khảo; cùng chủ đề nhưng khác
   nội dung; trùng thật ngoài XML; sai ranh giới highlight. Không gọi mọi trường
   hợp ngoài XML là lỗi embedding khi chưa đọc hai phía.
2. Với danh mục tham khảo, cần quy tắc chính sách rõ: có bỏ khỏi điểm đạo văn hay
   chỉ tách để người dùng xem? Nếu chưa được quyết định, nêu câu hỏi cụ thể;
   không tự xóa các câu có references hoặc tên sách.
3. Chuẩn bị positive paraphrase khó và hard negatives có cùng chủ đề trong
   train/validation mới. Không lấy toàn bộ hard cases từ test cuối để tune.
4. Thử một thay đổi mỗi lần: kiểm chứng từ vựng/ngữ cảnh hoặc reranker có sẵn.
   Phân biệt tính tương đồng chủ đề với quan hệ sao chép giữa hai đoạn cụ thể.
5. Nếu muốn dùng cross-encoder/model mới: trước hết báo model, dữ liệu train,
   tài nguyên, giấy phép và thời gian dự kiến để người dùng duyệt. Chưa cài hoặc
   tải model chỉ vì tài liệu này có đề cập.

Test đạt: báo cáo có cặp ví dụ, nhãn và lý do, kết quả cùng candidate pool trên
validation, không làm mất paraphrase đúng chỉ để tăng precision. Giữ riêng
semantic/verbatim/mixed và thời gian tăng thêm của reranker.

## N06. Cache có giới hạn RAM và tối ưu fingerprint

**Prompt giao task:**
"Đo kích thước semantic/fingerprint cache, thêm giới hạn theo dung lượng có
kiểm soát. Giữ cache riêng từng index/khoa và source version. Không thay thuật
toán exact matching hoặc dùng lại step=5 để tăng tốc."

Đọc `source_cache_key`, `get_cached_semantic_source`, `get_cached_fingerprints`,
`get_fingerprints`, `build_text_index`, `refresh_corpus_state`.
Cache hiện giới hạn 64 nguồn, không phải 64 nguồn có cùng kích thước.

Phạm vi: cache helpers trong detector/API và regression tests. Không viết
persistent cache vào vector DB thật, không thay lifecycle hoặc rebuild index.

Cách làm:

1. Viết test với nhiều nguồn nhỏ và một nguồn rất lớn; ghi memory-accounting của
   từng entry. Phân biệt ước lượng Python object với RSS thực của process.
2. Dùng LRU có budget byte; entry lớn hơn budget được xử lý nhưng không cache,
   không tạo vòng lặp eviction vô hạn. Giới hạn cả số entry nếu cần tương thích.
3. Giữ key namespace/khoa + source + version. Nội dung đổi hoặc nguồn bị xóa
   phải invalidation; không tái sử dụng chỉ vì cùng tên paper.pdf.
4. Giữ get_fingerprints step=1, mọi lần xuất hiện và kiểm chứng nội dung sau hash.
5. So output trước/sau cache bật/tắt; đo cùng file, một lượt cold được xác định
   và một lượt warm, ghi API RSS và tải hệ thống riêng nhau.
6. Nếu cần định dạng fingerprint gọn hoặc persistent cache, lập task riêng sau
   khi có profile CPU/RAM; không ghép tối ưu thuật toán và cache trong một patch.

Test đạt: hai khoa không dùng nhầm cache; update nguồn không trả text cũ; entry
lớn không vượt budget đã quy định cho dữ liệu cache; evidence và offset không
đổi. Đừng hứa giới hạn tổng RSS bằng đúng cache budget vì còn model và thư viện.

## N07. Đo character precision/recall trên PAN TXT

**Prompt giao task:**
"Bổ sung phép đo ký tự riêng cho PAN TXT với mapping normalized-to-raw rõ ràng.
Không so trực tiếp offset normalized của API với raw offset XML. Giữ nguyên
các metric evidence-hit hiện có và API response fields."

Phạm vi ban đầu: evaluator/helper mapping trong performance và tests. Đọc
`pdf_extract.extract_university_content`, `compact_with_index_map` để hiểu các
biến đổi nhưng không đổi extraction/version đang phục vụ index trong task này.
Nếu không dựng lại đúng phép biến đổi hiện tại, báo unavailable thay vì đo giả.

Cách làm:

1. Ghi chính xác mỗi bước biến đổi TXT trước khi scan: whitespace, Unicode,
   loại nội dung, v.v. Theo dõi chỉ số raw khi thực hiện phép biến đổi ấy.
2. Tạo mapping riêng cho submission và từng source; không dùng str.find lấy
   lần đầu vì một câu có thể lặp ở nhiều vị trí.
3. Test Unicode nhiều ký tự sau chuẩn hóa, CRLF/LF, nhiều space, dòng bị bỏ,
   đầu/cuối span và các đoạn lặp. Mapping giữ đúng thứ tự và quy ước end exclusive.
4. Tính giao/hợp interval trên raw theo cùng nguồn; gộp các interval chồng nhau
   trước khi đếm để không nhân đôi ký tự. Báo riêng phía submission/source.
5. Gắn schema mới, giữ evidence-hit/segment recall riêng. Không dùng vùng context
   hiển thị để tính matched characters; không lấy bounding span qua vùng đã bị xóa.

Test đạt: biết chính xác TP/FP/FN character trên fixture; span không thể map phải
được báo và không âm thầm tính đúng. Không mở rộng task sang OCR/PDF đa cột ở đây.

## N08. Mở rộng đánh giá, tạo tập test cuối chưa dùng để tune

**Prompt giao task:**
"Lập manifest PAN mở rộng có train/validation/test tách biệt, đủ positive và
negative, chia theo độ dài và loại obfuscation. Chỉ chuẩn bị dữ liệu và lệnh;
không tự chạy một chiến dịch GPU dài hoặc tune trên test."

Phạm vi: script lấy mẫu performance và manifest test_data; không sửa detector.
Input: các XML/TXT hiện có và manifest 18 bài đã được dùng để chẩn đoán.

Cách làm:

1. Kiểm kê các bài có đủ submission/XML/source thực sự đã indexed; báo thiếu.
   Không âm thầm bỏ nguồn XML thiếu rồi vẫn gọi là oracle đủ nguồn.
2. Đánh dấu 18 bài cũ đã được xem; không quảng bá chúng là test độc lập mới.
3. Giữ seed, SHA-256 và partition. Xem trùng văn bản/họ nguồn giữa partitions;
   ưu tiên chia theo nhóm nguồn khi khả thi, báo nếu nhóm quá lớn làm mất cân bằng.
4. Có bài không đạo văn và các loại none/low/high/translation, nhiều độ dài.
   Không mặc định cùng khoa đồng nghĩa cùng nội dung hoặc chắc chắn chứa nguồn đúng.
5. Dùng cùng 5/10 distractors cho các cấu hình đối chứng; ghi rõ cách chọn nhiễu.
6. Báo số bài/tổng dung lượng/ước tính thời gian để người dùng duyệt trước khi
   chạy toàn bộ. Khi chạy, lưu checkpoint và tách lỗi khỏi accuracy denominator.

Test đạt: partitions không trùng bài, noise không trùng nguồn XML, +5 là tập con
của +10, rerun seed tái tạo manifest. Không tự loại bài khó để tăng điểm báo cáo.

## N09. Kiểm thử đối chứng và quyết định chốt bản nộp

**Prompt giao task:**
"So bản almost ready và bản ứng viên trên cùng manifest/phép chấm. Báo đầy đủ
trade-off accuracy, thời gian và RAM; chỉ đề xuất đổi bản mặc định khi có bằng
chứng, chưa merge hoặc thay config production nếu chưa được yêu cầu."

Phạm vi: tests và performance reports; ghi báo cáo ngoài runtime. Không chạy hai
API GPU đồng thời trên máy 4 GB chỉ để so tốc độ; dùng cùng chế độ cache/tải máy.

Cách làm:

1. Chạy unit/regression liên quan rồi toàn bộ test chính thức.
2. Chạy oracle 0/5/10 với đúng nguồn của manifest, profile/config mỗi nhánh được
   lưu lại; source input, model, chunking và metric giữ như nhau khi so selection.
3. Báo P/R/F1, counts, negative files with evidence, mức trùng lặp, per-phase time,
   mean/median/p95, RAM API, RAM hệ thống và GPU; ghi sample size.
4. Với tối ưu cache, API/evidence phải tương đương; với tối ưu accuracy, báo mọi
   trường hợp tăng/giảm và lý do, không chỉ lấy trung bình che bài bị hỏng.
5. Mục tiêu đề xuất để người dùng quyết định, không phải chuẩn tốt nghiệp:
   giảm false positives trên negatives và noise mà không giảm recall; hoặc tăng
   recall rõ ràng mà precision không thấp hơn baseline trên cùng validation.
   Có thể đặt mục tiêu +5 điểm phần trăm recall cho thử nghiệm N04, nhưng đây
   là mục tiêu, không phải kết quả hứa trước hoặc lý do để tune vào test.
6. Chỉ chạy test cuối mới sau khi chốt lựa chọn bằng validation. Không tiếp tục
   chọn model theo điểm test; nếu dùng test để sửa, phải ghi nhận nó đã bị dùng.
7. Nếu không tốt hơn hoặc sắp hết thời gian, giữ almost ready và viết giới hạn
   trung thực. Không merge thử nghiệm chỉ vì đã tốn công chạy.

Đầu ra cuối: báo cáo so sánh, manifest/config/hash, regression tests, lệnh chạy,
commit ứng viên và đề xuất giữ/đổi. Không dùng câu 'unit tests pass nên accuracy tăng'.

## 5. Chạy benchmark và sử dụng mốc Git

### Kiểm tra nhanh không cần API/GPU

```cmd
cd /d D:\hochanh\hk252\plagiarism_runtime
venv\Scripts\python.exe -B -X utf8 performance\run_oracle_noise.py --self-test
tests\run_all_tests.cmd
```

Script benchmark hiện vẫn cần đọc evaluator, PAN và JSON shortlist cũ ở các
đường dẫn đã ghi trong script. Chuyển sang máy khác phải kiểm tra các phụ thuộc
này; không coi một Git clone là đã mang theo dataset, model hoặc venv.

### Bật API oracle chỉ cho phiên benchmark

Dừng API hiện có trước, sau đó ở CMD:

```cmd
cd /d D:\hochanh\hk252\plagiarism_runtime\runtime
..\venv\Scripts\python.exe -c "import api_service, uvicorn; api_service.API_CONFIG['allow_oracle_candidate_sources']=True; uvicorn.run(api_service.app, host='127.0.0.1', port=8000)"
```

Chờ `Application startup complete`. Ở CMD khác, dùng thư mục output CHƯA TỒN TẠI:

```cmd
cd /d D:\hochanh\hk252\plagiarism_runtime
venv\Scripts\python.exe -B -X utf8 performance\run_oracle_noise.py --run --output D:\hochanh\hk252\plagiarism_performance\oracle_noise_next_run
```

Đổi hậu tố output cho mỗi phép đo mới. Script hiện resume bằng response đã lưu;
chạy lại cùng thư mục không đồng nghĩa đã gọi API lại. Nó tái tạo danh sách từ
split và JSON shortlist cũ rồi kiểm tra manifest; phải đối chiếu với manifest
baseline đã commit, không tự bỏ kiểm tra hash khi input/config thay đổi.

Khi dùng API bình thường, dừng phiên oracle rồi khởi động theo lệnh chuẩn:

```cmd
cd /d D:\hochanh\hk252\plagiarism_runtime\runtime
..\venv\Scripts\python.exe api_service.py
```

### Tách nhánh thử nghiệm, giữ almost ready nguyên vẹn

Lấy hash mốc theo đúng message; kiểm tra worktree không có thay đổi cần giữ:

```cmd
git -C D:\hochanh\hk252\plagiarism_runtime log -1 --format="%h %s" --grep="^almost ready$"
git -C D:\hochanh\hk252\plagiarism_runtime status --short
```

Dùng hash vừa nhận thay `COMMIT_ALMOST_READY` trong lệnh sau, chỉ chạy khi branch
đích chưa tồn tại và các thay đổi của bạn đã được giữ an toàn:

```cmd
git -C D:\hochanh\hk252\plagiarism_runtime switch -c codex/precision-experiments COMMIT_ALMOST_READY
```

Đây là đổi branch trong cùng folder, không tạo bản sao corpus. Sau khi commit
các thử nghiệm trên branch mới, có thể quay về branch đang giữ mốc:

```cmd
git -C D:\hochanh\hk252\plagiarism_runtime switch codex/evidence-reranker-benchmark
```

Nếu Git báo có thay đổi chưa lưu, dừng và xử lý cùng người dùng; không dùng reset
--hard, checkout --, clean hoặc stash không được phép. Không đổi branch khi API
đang phục vụ request. Sau khi đổi code, khởi động lại API để nạp đúng phiên bản.

## 6. Mẫu bàn giao sau mỗi task

```text
Task đã làm: Nxx.
Commit nền và commit mới:
File sửa:
Lỗi tái hiện bằng fixture nào:
Nguyên nhân đã chứng minh và phần còn là giả thuyết:
Hành vi trước/sau:
Unit/regression test và lệnh:
Benchmark: manifest/config/scorer hash, số bài, lỗi, metric, thời gian:
Rủi ro, regression và giới hạn còn lại:
Kết luận: giữ baseline / đề xuất ứng viên; chưa tự làm task tiếp theo.
```

## 7. Thuật ngữ ngắn

- Oracle: cấp trước đúng các file nguồn từ XML để đo phần phát hiện bên trong.
- Noise/distractor: nguồn không được XML của bài ấy gán nhãn đạo văn; không chắc
  là hoàn toàn không có câu trùng thật ngoài phần được gán nhãn.
- Evidence: cặp đoạn submission/source cùng vị trí và thông tin cách phát hiện.
- Precision evidence-hit: bao nhiêu evidence hiển thị có ít nhất một cặp XML khớp.
- Recall theo đoạn: bao nhiêu đoạn XML phân biệt được bắt, không đếm lặp một đoạn.
- Character precision/recall: độ đúng/đủ tính theo ký tự, phải map về raw text.
- Consolidate: gộp những evidence liên quan thành vùng dễ đọc; khác chọn/loại vùng.
- Cap: giới hạn số evidence; cap theo nguồn có thể loại các vùng hợp lệ khác nhau.
- Reranker: chấm lại và sắp xếp candidate sau khi các phase trước đã tìm evidence.
- Hard negative: ví dụ dễ bị nhầm là đạo văn vì giống chủ đề/câu chữ nhưng không
  thuộc nhãn positive được đánh giá.
- Boilerplate: nội dung lặp mang tính khuôn mẫu như ghi công hoặc thông báo xuất bản.
- Cache invalidation: loại dữ liệu lưu tạm khi nguồn/index đổi phiên bản.
- Ablation: thay một thành phần mỗi lần để biết thành phần đó gây khác biệt gì.
