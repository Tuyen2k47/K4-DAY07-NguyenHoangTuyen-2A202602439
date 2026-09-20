# BÁO CÁO CÁ NHÂN: HỆ THỐNG TRUY XUẤT VĂN BẢN VÀ ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG NGỮ NGHĨA

* **Họ và tên:** Nguyễn Hoàng Tuyên
* **Mã sinh viên:** 2A202602439
* **Nhóm:** L3B (Quy định & Chính sách Trả hàng - Hoàn tiền Shopee)
* **Kho lưu trữ:** `K4-DAY07-NguyenHoangTuyen-2A202602439`

---

## 1. Mục tiêu và Phạm vi

Báo cáo này trình bày kết quả nghiên cứu và thực nghiệm xây dựng hệ sinh thái truy xuất ngữ nghĩa (Semantic Retrieval Pipeline) phục vụ tra cứu các điều khoản, quy định chính sách Trả hàng và Hoàn tiền của sàn TMĐT Shopee.

Các mục tiêu kỹ thuật chính:
1. So sánh và đánh giá hai chiến lược phân mảnh văn bản: Cố định độ dài (`FixedWindowChunker`) và Theo cấu trúc ranh giới câu ngữ pháp (`SentenceChunker`).
2. Hiện thực hóa giải pháp lưu trữ véc-tơ (`VectorStore`) kết hợp cơ chế lọc thuộc tính siêu dữ liệu (`Metadata Filtering`).
3. Khảo sát năng lực biểu diễn ngữ nghĩa của mô hình nhúng đa ngôn ngữ thực tế `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
4. Đánh giá chất lượng truy xuất trên bộ benchmark gồm 5 câu hỏi chuẩn do nhóm L3B xây dựng.

---

## 2. Chiến lược Phân mảnh Văn bản (Chunking Strategy)

### 2.1. So sánh phương pháp luận

| Tiêu chí | FixedWindowChunker (Cửa sổ cố định) | SentenceChunker (Theo ranh giới câu) |
| :--- | :--- | :--- |
| **Nguyên lý** | Cắt văn bản theo số ký tự/từ cố định, dịch chuyển theo bước trượt có phần gối đầu (overlap). | Tách văn bản theo các dấu kết thúc câu (`.`, `!`, `?`), sau đó gom tối đa $N$ câu liên tiếp thành một chunk hoàn chỉnh. |
| **Bảo toàn ngữ nghĩa** | Kém. Dễ làm đứt gãy giữa câu, tách rời điều kiện ngoại lệ khỏi mệnh đề chính. | Tốt. Giữ nguyên vẹn cấu trúc logic của từng câu văn bản pháp lý/quy chế. |
| **Độ dài chunk** | Đồng đều tuyệt đối về số lượng ký tự hoặc từ. | Biến thiên tùy thuộc vào độ dài các câu trong đoạn văn. |
| **Mục đích phù hợp** | Dữ liệu phi cấu trúc, nhật ký log hệ thống hoặc tài liệu kỹ thuật dài. | Tài liệu chính sách quy định, điều khoản dịch vụ, văn bản hỏi đáp nghiệp vụ. |

### 2.2. Kết quả phân mảnh trên tập dữ liệu Shopee

Áp dụng `SentenceChunker(max_sentences_per_chunk=3)` trên 6 tài liệu chính sách Shopee, hệ thống tạo ra **46 chunks**:
- `buyer-refund-timeline`: 3 chunks
- `buyer-return-conditions`: 9 chunks
- `buyer-return-processing`: 11 chunks
- `buyer-return-request-guide`: 7 chunks
- `seller-mall-return-obligations`: 7 chunks
- `seller-rights-and-duties`: 9 chunks

---

## 3. Kiến trúc VectorStore và Cơ chế Metadata Filtering

### 3.1. Tính toán độ tương đồng véc-tơ
Mỗi chunk văn bản được biểu diễn dưới dạng một véc-tơ đặc trưng nhiều chiều $\vec{v} \in \mathbb{R}^d$. Độ tương đồng giữa câu truy vấn $\vec{q}$ và tài liệu $\vec{d}$ được tính bằng công thức Cosine Similarity:
$$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2}$$

### 3.2. Tác dụng của Metadata Filtering
Bộ dữ liệu có sự phân hóa rõ ràng về đối tượng:
- `audience = 'buyer'`: Hướng dẫn và quyền lợi dành cho Người Mua.
- `audience = 'seller'`: Quy trình, nghĩa vụ và chế tài dành cho Người Bán / Shopee Mall.

Việc tiền lọc (`filter_dict={'audience': ...}`) trước khi xếp hạng véc-tơ mang lại hai ưu điểm:
1. **Loại bỏ nhiễu chéo:** Tránh trường hợp người mua hỏi thời hạn hoàn tiền nhưng hệ thống lại trả về quy định hạn phản hồi của người bán.
2. **Tối ưu tốc độ:** Giảm số lượng phép tính tích vô hướng véc-tơ từ 46 chunks xuống chỉ còn các chunk thuộc đối tượng cần tra cứu.

---

## 4. Phân tích Độ tương đồng Ngữ nghĩa (Semantic Similarity)

Sử dụng mô hình nhúng thực tế `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, kết quả đo độ tương đồng Cosine giữa các cặp câu tiếng Việt như sau:

| Cặp câu | Nội dung câu A và câu B | Cosine Similarity | Phân tích bản chất ngữ nghĩa |
| :---: | :--- | :---: | :--- |
| **Cặp 1** | **A:** "Thời gian hoàn tiền qua thẻ tín dụng mất bao lâu?"<br>**B:** "Người mua nhận lại tiền thẻ tín dụng trong mấy ngày?" | **0.892** | Hai câu dùng từ ngữ hoàn toàn khác nhau ("bao lâu" vs "mấy ngày", "hoàn tiền" vs "nhận lại tiền") nhưng mô hình nhúng nhận diện độ tương đồng ngữ nghĩa rất cao. |
| **Cặp 2** | **A:** "Thời gian hoàn tiền qua thẻ tín dụng mất bao lâu?"<br>**B:** "Người bán cần chuẩn bị hàng trong thời hạn quy định." | **0.174** | Hai câu đề cập đến hai đối tượng và quy trình khác biệt (nhận tiền vs chuẩn bị hàng), độ tương đồng thấp phản ánh chính xác sự phân tách chủ đề. |
| **Cặp 3** | **A:** "Tôi muốn đổi ý không nhận hàng đã đặt."<br>**B:** "Chính sách trả hàng khi người mua thay đổi nhu cầu." | **0.835** | Nắm bắt tốt mối liên hệ giữa khẩu ngữ đời thường ("đổi ý không nhận") và thuật ngữ chính sách ("thay đổi nhu cầu"). |
| **Cặp 4** | **A:** "Thực phẩm đông lạnh được hoàn tiền trong bao lâu?"<br>**B:** "Cách đăng ký tài khoản Shopee Mall cho doanh nghiệp." | **0.048** | Hai ngữ cảnh hoàn toàn độc lập; điểm số tiệm cận 0 chứng minh không gian biểu diễn véc-tơ phân tách rất rõ ràng. |
| **Cặp 5** | **A:** "Bưu tá đến tận nhà thu hồi hàng hoàn."<br>**B:** "Phương thức lấy hàng trả tận nơi cho người mua." | **0.926** | Nhận biết xuất sắc tính tương đương ngữ nghĩa của các cụm từ đồng nghĩa ("thu hồi hàng hoàn" tương ứng với "lấy hàng trả tận nơi"). |

---

## 5. Kết quả Đánh giá Benchmark Truy xuất (Shopee Retrieval Benchmark)

Hệ thống được kiểm thử tự động trên 5 câu truy vấn chuẩn của nhóm với cấu hình:
- **Chiến lược phân mảnh:** `SentenceChunker(max_sentences_per_chunk=3)`
- **Mô hình nhúng:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Số lượng chunks lưu trữ:** 46

### 5.1. Bảng chi tiết kết quả truy xuất

| Query ID | Câu hỏi truy vấn | Filter | Văn bản chuẩn (Gold Doc) | Chunk Top-1 tìm được | Doc Hit (Top-3) | Evidence Hit (Top-3) | Điểm |
| :---: | :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **Q1** | Với thực phẩm tươi sống và đông lạnh, người mua phải gửi yêu cầu trong bao lâu? | `buyer` | `buyer-return-conditions` | `buyer-return-conditions` #1 (0.6161) | **YES (Rank 1)** | **YES (Rank 1)** | **2/2** |
| **Q2** | Trường hợp Người Mua đổi ý không muốn nhận hàng, Shopee xử lý theo hướng nào? | `buyer` | `buyer-return-conditions` | `buyer-return-processing` #7 (0.6562) | **NO** | **NO** | **0/2** |
| **Q3** | Có những phương thức trả hàng nào khả dụng cho Người Mua khi gửi hàng hoàn về? | `buyer` | `buyer-return-request-guide` | `buyer-return-request-guide` #6 (0.6797) | **YES (Rank 1)** | **NO** | **0/2** |
| **Q4** | Thời gian hoàn tiền cho Người Mua khi thanh toán qua Thẻ tín dụng mất bao lâu? | `buyer` | `buyer-refund-timeline` | `buyer-refund-timeline` #1 (0.7478) | **YES (Rank 1)** | **YES (Rank 2)** | **1/2** |
| **Q5** | Người Bán Shopee Mall có nghĩa vụ gì khi nhận được khiếu nại từ Người Mua? | `seller` | `seller-mall-return-obligations` | `seller-rights-and-duties` #3 (0.5726) *(Gold doc xếp Rank 2: 0.5529)* | **YES (Rank 2)** | **NO** | **0/2** |

### 5.2. Tổng kết hiệu năng

```text
================================================================================
SUMMARY
Retrieval doc hit rate : 4/5 (80.0%)
Answer evidence hit rate: 2/5 (40.0%)
Content benchmark score : 3/10 (30.0%)
Total queries           : 5
================================================================================
```

### 5.3. Nhận xét và Phân tích chuyên sâu

1. **Hiệu năng định vị tài liệu (Doc Hit Rate đạt 80%):**
   - Mô hình `sentence-transformers` thể hiện khả năng định vị tài liệu chứa câu trả lời rất tốt (4 trên 5 câu hỏi xuất hiện đúng tài liệu trong Top-3).
   - Ở Q1, Q3, Q4, tài liệu chuẩn đều đứng ở vị trí Rank 1 với độ tương đồng từ 0.61 đến 0.75.
2. **Nguyên nhân chênh lệch giữa Doc Hit (80%) và Evidence Hit (40%):**
   - Tài liệu quy định thường có cấu trúc dài; khi chia thành nhiều chunk độc lập không có phần gối đầu (no overlap), câu trả lời chứa thông tin chi tiết (ví dụ danh sách 3 phương thức trả hàng ở Q3) bị đẩy lùi về các chunk phía sau, nhường chỗ cho chunk mở đầu quy trình có chứa các từ khóa trùng lặp cao hơn.
   - Ở Q2, văn bản `buyer-return-processing` bị xếp trước `buyer-return-conditions` vì câu hỏi chứa cụm từ "xử lý yêu cầu Trả hàng/Hoàn tiền" dẫn tới việc mô hình ưu tiên văn bản chuyên về quy trình xử lý.
3. **Giải pháp khắc phục:**
   - Bổ sung tham số trượt chồng lấn (overlap sentences) khi phân mảnh câu để giữ liền mạch ngữ cảnh danh mục liệt kê.
   - Áp dụng mô hình xếp hạng lại (Cross-Encoder Re-ranker) để chấm điểm trực tiếp cặp `(Query, Chunk)` trước khi trích xuất câu trả lời.

---

## 6. Kết luận

Dự án đã triển khai thành công pipeline xử lý và truy xuất dữ liệu chính sách Shopee:
- Vượt qua toàn bộ 42/42 bài kiểm thử tự động (`pytest tests/ -v`).
- Tích hợp thành công mô hình ngôn ngữ véc-tơ thực nghiệm đa ngữ `sentence-transformers`, đem lại kết quả truy xuất sát với thực tế vận hành của hệ thống RAG (Retrieval-Augmented Generation).
