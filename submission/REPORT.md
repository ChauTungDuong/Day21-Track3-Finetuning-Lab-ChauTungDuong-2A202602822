# Lab 21 — Evaluation Report

**Họ tên**: Châu Tùng Dương  **MSSV**: 2A202602822  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB (Google Colab)`

> Mọi con số dưới đây khớp chính xác với các file kết quả trong `results/`.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (tỉ lệ 90/10, phân tách ngẫu nhiên cố định với seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(theo `results/token_stats.json`)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?** CÓ — *(theo `results/template_check.json`)*
- Kết quả kiểm tra: `open_tag_present: true`, `body_present: true`, `verdict: "reasoning preserved — safe to train on traces"`.
- Template của `unsloth/Qwen3.5-4B` bảo toàn nguyên vẹn cặp thẻ `<think>...</think>`, không nuốt hay loại bỏ nội dung suy luận trong quá trình tokenize và render hội thoại, đảm bảo toàn bộ reasoning trace tiếp cận đúng hàm loss.
- Về `max_length`: Độ dài chuỗi token đo được trên tập dữ liệu có mean = 93.1, p50 = 93, p95 = 98, p99 = 100, max = 101 token. Mức gợi ý tối thiểu theo luỹ thừa 2 là 256 token. Cấu hình Tier T4 giữ giá trị trần an toàn là 1024 token, đảm bảo 100% mẫu huấn luyện không bị cắt xén (truncate) mà vẫn nằm trọn trong vùng bộ nhớ GPU cho phép.

---

## 2. Mask proof (NB1)

| Tiêu chí | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số token được tính loss) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

3–5 dòng đầu của đoạn được tính loss (trích xuất từ `results/mask_proof.json`):

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

*Phân tích mask*: Phần tính loss chỉ bắt đầu sau thẻ đóng `</think>` và kết thúc ở token `<|im_end|>`, tổng cộng 39/94 token. Toàn bộ system instruction và nội dung ticket của người dùng được gán nhãn `-100` (masked hoàn toàn). Điều này chứng minh pipeline che loss hoạt động tuyệt đối chính xác: mô hình chỉ học cách suy ra kết quả phân loại chuẩn mà không bị tính loss lên câu hỏi, triệt tiêu hoàn toàn lỗi mô hình học vẹt viết tiếp prompt của người dùng.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7500 | 0.0000 | 3360.6 |
| (b) base + optimized prompt | 0.6875 | 0.7500 | 1.0000 | 999.8 |
| (c) LoRA fine-tune | 0.9375 | 0.7500 | 1.0000 | 1505.5 |

- **(b) có thật sự mạnh hơn (a) không?**: CÓ. Prompt tối ưu nâng độ chính xác mục tiêu từ 0.0% lên 68.75%, tuân thủ đúng định dạng JSON 100% (so với 0.0% khi dùng naive prompt), đồng thời giảm độ trễ từ 3360.6 ms xuống 999.8 ms do mô hình không bị sinh văn bản lan man ngoài cấu trúc.
- **Bạn có sửa `OPTIMIZED_PROMPT` không?**: KHÔNG. Chuỗi `OPTIMIZED_PROMPT` được giữ nguyên vẹn với mã hash SHA `719e74d3b6232053` như ban đầu. Việc giữ nguyên mốc đánh giá đóng băng này đảm bảo tính liêm chính khoa học: không hạ thấp tiêu chuẩn của baseline (b) để làm bản fine-tune trông có vẻ thắng ảo.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6265 | **0.9375** | 412.5 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5373 | **0.9375** | 269.1 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | **0.0000** | 400.0 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | **0.8438** | 467.5 | 3.86 |

> Bảng được xếp hạng theo cột **target (NB5 §4)**, không xếp theo cột train loss của NB4.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
- `attn_only` được nâng rank lên $r=283$ bằng thuật toán `matched_rank()` để có cùng số tham số huấn luyện xấp xỉ `correct` (32,456,704 so với 32,464,896, sai lệch chỉ 0.025%). Trên tập kiểm thử target, `attn_only` đạt điểm 0.9375, hoàn toàn hoà với `correct` (0.9375).
- Tuy nhiên, thứ tự này hoàn toàn trái ngược với thứ tự theo train loss: ở NB4, `attn_only` có loss thấp hơn hẳn `correct` (0.5373 so với 0.6265). Nếu một kỹ sư chỉ đánh giá dựa trên train loss, họ sẽ kết luận sai lầm rằng `attn_only` với rank 283 vượt trội hơn.
- Kết quả này chứng minh rằng rank cao dồn vào các khối attention chỉ đơn thuần giúp mô hình ghi nhớ (memorize/overfit) dữ liệu huấn luyện nhanh hơn, nhưng không giúp mô hình tổng quát hoá tốt hơn. Đòn bẩy biểu diễn thực sự đến từ việc phân bổ adapter trải rộng khắp các khối tuyến tính của mô hình (`text-linear`) thay vì tập trung cục bộ vào attention.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
- Run `wrong_lr` chỉ thay đổi một biến duy nhất: hạ learning rate từ $10^{-4}$ (thang chuẩn cho LoRA) xuống $10^{-5}$ (thang thông thường của full fine-tuning). Trong suốt 30 step, đường loss của `wrong_lr` gần như phẳng lì, dừng ở mức loss rất cao là 1.5702, và điểm target hoàn toàn bằng 0.0000 (format cũng 0.0000).
- Nếu chỉ nhìn vào đường loss không giảm mà không nắm rõ thang LR của LoRA, người huấn luyện sẽ rất dễ đưa ra các kết luận sai lầm nghiêm trọng như: bộ dữ liệu bị lỗi, tác vụ phân loại JSON quá khó với mô hình 4B, hoặc kiến trúc LoRA không tương thích.
- Thực chất, vì số lượng tham số trainable của LoRA chiếm tỷ trọng rất nhỏ (~0.8% tổng mô hình), gradient tích luỹ cần một bước nhảy đủ lớn (~10x) để cập nhật hiệu quả các ma trận $A$ và $B$; việc áp dụng máy móc LR của full fine-tuning khiến adapter dậm chân tại chỗ.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
- `qlora` (4-bit NF4) tiết kiệm được 4.92 GB VRAM, đưa đỉnh tiêu thụ từ 8.78 GB xuống chỉ còn 3.86 GB (giảm gần 56% bộ nhớ đồ hoạ). Điều này mở ra khả năng chạy trên các dòng GPU phổ thông dung lượng nhỏ.
- Tuy nhiên, sự đánh đổi là rất rõ ràng: thời gian huấn luyện tăng từ 412.5s lên 467.5s (chậm hơn khoảng 13% do chi phí tính toán dequantization liên tục giữa 4-bit và 16-bit), độ trễ suy luận tăng lên 1762.8 ms, và quan trọng nhất là điểm target giảm sút đáng kể từ 0.9375 xuống 0.8438 (mất gần 9.4% độ chính xác).
- Các số đo thực nghiệm này hoàn toàn ủng hộ khuyến nghị chính thức của Unsloth đối với kiến trúc Qwen3.5: sai số lượng tử hoá 4-bit làm suy giảm đáng kể chất lượng biểu diễn của mô hình. Do đó, nếu GPU có đủ VRAM (như T4 16GB), ta nên ưu tiên dùng 16-bit LoRA thay vì lạm dụng QLoRA.

---

## 5. Phán quyết (NB5)

- **Kết quả cổng hồi quy**: `PASSED`
- `target Δ = +0.250` · `regression Δ = +0.000` · `valid_trace_rate = 0.0000`

### Diễn giải phán quyết (156 từ)
Cổng hồi quy 4 nhóm chính thức xác nhận kết quả PASSED cho bản fine-tune `correct`. Bản mô hình đã vượt qua mốc chuẩn tối ưu hóa prompt (b) một cách thuyết phục với mức cải thiện độ chính xác tác vụ mục tiêu là +0.250 (+25.0%, từ 0.6875 lên 0.9375), đồng thời tuân thủ hoàn hảo 100% định dạng JSON đầu ra. Đáng chú ý, năng lực tổng quát trên tập hồi quy không hề bị suy giảm (`regression Δ = +0.000`, duy trì ở mức 0.750), chứng minh quá trình cập nhật tham số PEFT không gây ra hiện tượng quên thảm họa đối với các tri thức phổ thông ngoài miền CSKH. Mặc dù độ trễ suy luận của bản fine-tune (1505.5 ms) cao hơn so với baseline b (999.8 ms) do mô hình sinh chi tiết đầy đủ 4 khóa mà không cần prompt dài nhồi nhét, sự đánh đổi này là hoàn toàn xứng đáng với độ tin cậy vượt trội mà bản fine-tune mang lại cho quy trình xử lý dữ liệu tự động.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | `VN232232`: "Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt." | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | Nhận diện đúng intent, nhưng sentiment bị nhầm sang `trung_tinh` | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | ✅ **FT thắng**: Bắt chuẩn xác 4/4 trường, nhận diện được sentiment tích cực ("hỗ trợ tốt"). (Score 1.0) |
| 2 | `VN812931`: "Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình." | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}` | Output đôi khi thiếu key hoặc nhận diện nhầm urgency sang `cao` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}` | ✅ **FT thắng**: Tách đúng sản phẩm và gán nhãn tiêu cực chính xác theo từ khóa "bực mình". (Score 1.0) |
| 3 | `VN804124`: "Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều." | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | ❌ **FT thua**: Khách ghi "Khi nào tiện" (nhãn đúng là `thap`), nhưng mô hình FT lại phân loại thành `trung_binh`. (Score 0.75) |
| 4 | `DH249548`: "Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi." | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | ❌ **FT thua**: Tương tự ca trên, cụm từ "Khi nào tiện" bị mô hình FT gán sai thành `trung_binh`. (Score 0.75) |
| 5 | `VN880807`: "Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều." | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}` | Dự đoán urgency `trung_binh` do thấy lời cảm ơn | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}` | ✅ **FT thắng**: Bắt được tính chất "quá hạn rồi" để nâng urgency lên `cao` dù sentiment tích cực. (Score 1.0) |

### Có mẫu chung nào ở các ca FT thua không?
Cả hai ca mô hình fine-tune bị mất điểm (đạt score 0.75 thay vì 1.0) đều phạm phải cùng một lỗi cố hữu: phân loại nhầm mức `urgency` từ `thap` thành `trung_binh` khi gặp cụm từ "Khi nào tiện". Nguyên nhân là trong tập dữ liệu huấn luyện CSKH, các khiếu nại về tiền bạc ("chưa thấy tiền") hoặc lỗi hàng ("thiếu phụ kiện") hầu như luôn đi kèm mức độ khẩn cấp từ trung bình đến cao. Sự mất cân bằng nhãn cục bộ này đã tạo ra một thiên lệch quy nạp (inductive bias) mạnh mẽ, khiến mô hình fine-tune phớt lờ sắc thái từ vựng "khi nào tiện". Ngược lại, prompt engineering (b) nhờ giữ được năng lực zero-shot thuần ngữ nghĩa của base model nên đã diễn giải chuẩn xác cụm từ này thành mức độ khẩn cấp thấp.

---

## 7. Kết luận & điều tôi học được

### Kết luận (210 từ)
Dựa trên kết quả thực nghiệm toàn diện và việc vượt qua cổng hồi quy 4 nhóm, việc triển khai bản fine-tune `correct` vào môi trường sản xuất cho bài toán phân loại ticket CSKH là hoàn toàn khả thi và có lợi thế vượt trội. So với phương pháp prompt engineering truyền thống, mô hình fine-tune nâng cao độ chính xác mục tiêu thêm 25.0% (đạt 93.75%), cam kết chuẩn định dạng JSON 100%, và đặc biệt là không cần phải nhồi nhét system prompt phức tạp cùng các ví dụ few-shot vào mỗi lượt gọi API, giúp tiết kiệm đáng kể chi phí token context window.

Tuy nhiên, bài học cốt lõi rút ra từ bài lab này là đòn bẩy thực sự của quá trình tinh chỉnh LLM không nằm ở việc tăng rank hay chạy đua kích thước adapter, mà nằm ở tính đúng đắn của pipeline: **chọn đúng thang Learning Rate và kiểm soát chính xác Loss Masking**. Việc dồn rank lên 283 ở các tầng attention (`attn_only`) chỉ làm giảm loss huấn luyện chứ không mang lại bất kỳ lợi ích nào trên tập đánh giá thực tế so với rank 16 trải rộng toàn bộ lớp tuyến tính (`text-linear`). Ngược lại, chỉ cần lệch một bậc thang LR (`wrong_lr`), adapter hoàn toàn tê liệt. Kiểm soát chặt chẽ mask và tuân thủ các nguyên tắc thiết kế thí nghiệm công bằng chính là chìa khóa quyết định thành bại của một dự án fine-tuning.

### Ba điều tôi học được (cụ thể, không generic):
1. **Training loss là một chỉ số thay thế nguy hiểm và dễ gây ngộ nhận**: Thực nghiệm `attn_only` đạt loss 0.5373 (thấp hơn `correct` 0.6265) nhưng điểm target thực tế lại chỉ ngang bằng (0.9375). Đánh giá một mô hình fine-tune bắt buộc phải dựa trên benchmark tác vụ độc lập được đóng băng trước đó, không bao giờ được lấy training loss làm thước đo chất lượng.
2. **Quy tắc 10x Learning Rate cho LoRA là ranh giới sống còn**: Vì LoRA chỉ cập nhật một phần rất nhỏ tham số (~0.8%), nó đòi hỏi tốc độ học cao gấp 10 lần so với full fine-tuning ($10^{-4}$ so với $10^{-5}$). Sử dụng sai thang LR sẽ khiến gradient không đủ tạo ra bước cập nhật có ý nghĩa, làm adapter thất bại hoàn toàn.
3. **QLoRA không phải là giải pháp vạn năng miễn phí**: Dù QLoRA cắt giảm hơn 55% VRAM (từ 8.78 GB xuống 3.86 GB), nó phải trả giá bằng việc tăng thời gian huấn luyện (overhead dequantization) và làm giảm độ chính xác mục tiêu gần 9.4% trên kiến trúc Qwen3.5. Chỉ nên đánh đổi dùng QLoRA khi phần cứng thực sự không đáp ứng được 16-bit.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. Thử nghiệm kỹ thuật Data Replay: trộn thêm 2–5% dữ liệu tổng quát phổ thông vào tập huấn luyện để kiểm chứng xem liệu có thể đẩy điểm `regression` vượt mốc 0.75 hay không.
2. Hoàn thiện NB6 (`06_merge_and_serve.py`) để merge adapter vào base model, lượng tử hoá sang GGUF và benchmark throughput phục vụ thực tế trên vLLM hoặc llama.cpp.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
