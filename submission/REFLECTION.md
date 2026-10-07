# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là hiện tượng ở run `attn_only`: khi nâng rank lên $r=283$ dồn vào chỉ hai ma trận $q, v$ (để khớp cùng 32.4 triệu tham số trainable như `correct`), training loss giảm sâu hơn hẳn so với `correct` ($0.5373$ so với $0.6265$). Nhưng khi đưa lên tập kiểm thử độc lập NB5, điểm số target của hai mô hình lại hoàn toàn bằng nhau ($0.9375$). Nếu chỉ nhìn vào loss log trên màn hình terminal, tôi chắc chắn đã tin rằng `attn_only` là mô hình chiến thắng áp đảo. Nó cho thấy việc người làm AI bị đánh lừa bởi training loss dễ dàng đến mức nào nếu không có một tập benchmark kiểm tra chéo được đóng băng từ đầu.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở công đoạn chạy chuỗi đối chứng 3 cấu hình sai ở NB4 trên Google Colab T4 (~35–45 phút) và kiểm tra cẩn thận tính khớp tham số trainable giữa các run. Đây không phải chỗ ban đầu tôi dự đoán; ban đầu tôi tưởng việc cài đặt môi trường, sửa lỗi tokenizer và chat template ở NB1 sẽ chiếm phần lớn thời gian. Thực tế, việc chạy các bài đo thực nghiệm một cách công bằng (cùng step, cùng ngân sách tham số, chỉ đổi một biến duy nhất) đòi hỏi kỷ luật cao và tốn thời gian tính toán nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng có hai niềm tin sai lầm phổ biến:
1. "Rank LoRA càng cao thì chất lượng mô hình càng tốt" — Lab đã chứng minh rank cao trong attention chỉ tăng năng lực ghi nhớ (overfitting) tập train chứ không tăng tính khái quát hoá so với việc đặt rank nhỏ ($r=16$) trên toàn bộ các lớp tuyến tính (`text-linear`).
2. "QLoRA 4-bit luôn là mặc định tối ưu trên mọi GPU nhỏ" — Thực tế đo đạc trên Qwen3.5 cho thấy 4-bit làm tăng 13% thời gian huấn luyện (do overhead dequantization) và làm tụt gần 9.4% độ chính xác target do sai số lượng tử hoá, chứng minh khuyến nghị của vendor là hoàn toàn có cơ sở khoa học.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để giải thích cơ chế che loss của TRL/transformers, tự động hoá quy trình chạy liên hoàn qua script `colab_run.py`, và hỗ trợ rà soát các tiêu chí của rubric chấm điểm. Chỗ AI thường mắc sai lầm là: nếu không được định hướng chặt chẽ về tính liêm chính khoa học, AI có xu hướng phán đoán thứ tự các run dựa trên `final_loss` của NB4 (chính là lỗi kinh điển số 3 mà bài lab cảnh báo), hoặc tự ý đề xuất nới lỏng các ngưỡng kiểm tra thay vì phân tích bản chất thất bại của mô hình.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Tôi sẽ tuyệt đối không bật GPU hay nạp dữ liệu vào train ngay lập tức. Bước đầu tiên tôi làm sẽ là: thu thập một tập kiểm thử độc lập phản ánh đúng phân phối thực tế của khách hàng, đóng băng nó, và xây dựng một baseline prompt engineering thật mạnh (baseline b) kèm cổng hồi quy đa nhóm (độ chính xác tác vụ, khả năng giữ tri thức tổng quát, định dạng đầu ra, và độ trễ). Chỉ khi baseline b chứng minh không thể đáp ứng được yêu cầu nghiệp vụ (về độ chính xác, tuân thủ schema, hoặc chi phí context token), tôi mới tiến hành fine-tuning và dùng chính bộ cổng này để đưa ra phán quyết có nên đưa lên production hay không.
