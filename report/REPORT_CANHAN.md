# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Hoàng Tuyên  
**Mã sinh viên:** 2A202602439  
**Nhóm:** Nhóm L3B — Chính sách Thương mại Điện tử (Shopee)  
**Ngày:** 20/09/2026  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) nghĩa là hai vector biểu diễn văn bản tạo với nhau một góc rất nhỏ trong không gian đa chiều, phản ánh rằng hai đoạn văn bản có sự tương đồng lớn về mặt ý nghĩa ngữ nghĩa, dù có thể dùng từ ngữ biểu đạt khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Shopee hỗ trợ người mua trả hàng và hoàn tiền trong 15 ngày."
- Câu B: "Khách hàng có thể gửi yêu cầu hoàn tiền trên ứng dụng Shopee trong vòng 15 ngày kể từ lúc nhận hàng."
- Tại sao tương đồng: Cả hai câu cùng truyền tải chung một nội dung chính sách về thời hạn đổi trả 15 ngày của Shopee dành cho người mua, dù câu B diễn đạt chi tiết hơn bằng các từ đồng nghĩa.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Shopee hỗ trợ người mua trả hàng và hoàn tiền trong 15 ngày."
- Câu B: "Cách nấu phở bò gia truyền cần ninh xương ống trong ít nhất 8 tiếng."
- Tại sao khác: Hai câu thuộc về hai lĩnh vực hoàn toàn xa lạ (chính sách thương mại điện tử vs công thức nấu ăn ẩm thực), không chia sẻ ngữ cảnh hay khái niệm ngữ nghĩa chung nào.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị chi phối bởi độ dài (độ lớn vector) của văn bản — một câu ngắn và một đoạn văn dài cùng ý nghĩa sẽ có khoảng cách Euclid rất lớn. Trong khi đó, độ tương tự Cosine chỉ đo góc giữa hai vector và tự chuẩn hóa theo độ dài, giúp so sánh chính xác mức độ tương đồng ngữ nghĩa bất kể văn bản dài hay ngắn.

---

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*  
> Áp dụng công thức:  
> $\text{Số lượng chunk} = \left\lceil \frac{\text{độ dài tài liệu} - \text{độ chồng chéo}}{\text{kích thước chunk} - \text{độ chồng chéo}} \right\rceil = \left\lceil \frac{10000 - 50}{500 - 50} \right\rceil = \left\lceil \frac{9950}{450} \right\rceil = \lceil 22.111 \rceil = 23$  
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, số lượng chunk sẽ là: $\lceil (10000 - 100) / (500 - 100) \rceil = \lceil 9900 / 400 \rceil = 25$ chunks (tăng thêm 2 chunks).  
> Chúng ta muốn độ chồng chéo nhiều hơn để bảo toàn ngữ cảnh liền mạch ở các ranh giới cắt, ngăn chặn việc một câu quan trọng hoặc một điều kiện chính sách bị chia đôi làm mất ý nghĩa khi thực hiện truy xuất vector.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Em sử dụng biểu thức chính quy `re.split(r'(?<=[.!?])\s+', text)` (lookbehind) để tách câu ngay sau các dấu kết thúc câu (`.`, `!`, `?`) mà vẫn bảo toàn dấu câu. Sau đó, các câu được làm sạch khoảng trắng bằng `.strip()` và gom nhóm theo từng khối không quá `max_sentences_per_chunk` câu. Xử lý các edge case như chuỗi rỗng hoặc chỉ có khoảng trắng bằng cách trả về danh sách rỗng `[]`.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo tư tưởng chia để trị đệ quy: thử lần lượt các dấu phân cách theo độ ưu tiên giảm dần `["\n\n", "\n", ". ", " ", ""]`. Trường hợp cơ sở (base case) là khi đoạn văn bản có độ dài $\le$ `chunk_size` hoặc đã duyệt hết danh sách separators; nếu một đoạn con vẫn vượt quá kích thước cho phép, hàm sẽ gọi đệ quy `_split` với dấu phân cách kế tiếp nhỏ hơn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Dữ liệu được lưu trữ trong bộ nhớ dưới dạng danh sách các từ điển (`self._store`) gồm `id`, `content`, `embedding` và `metadata` (tự động gán `metadata['doc_id'] = doc.id` nếu chưa có). Khi tìm kiếm (`search`), truy vấn được nhúng thành vector và tính điểm tương đồng với từng record qua tích vô hướng `_dot(query_emb, doc_emb)`, sau đó sắp xếp giảm dần theo `score` và trả về top-k phần tử.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` áp dụng cơ chế lọc trước (pre-filtering): duyệt qua kho dữ liệu để chọn lọc những record khớp toàn bộ cặp key-value trong `metadata_filter` rồi mới tính điểm tương đồng, giúp tăng tốc độ và loại bỏ nhiễu. `delete_document` tiến hành lọc loại bỏ tất cả các chunk có `metadata['doc_id'] == doc_id` hoặc `id == doc_id` và trả về `True` nếu số lượng chunk trong kho giảm đi.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Hàm `answer` trước hết gọi `store.search(question, top_k)` để trích xuất các đoạn văn bản liên quan nhất làm ngữ cảnh tham khảo (`context_text`). Prompt được thiết kế rõ ràng bằng tiếng Việt yêu cầu mô hình đóng vai trợ lý tri thức, căn cứ nghiêm ngặt trên ngữ cảnh được cung cấp để trả lời súc tích và chính xác, sau đó chuyển prompt cho hàm `llm_fn` để sinh kết quả.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- E:\VIN_LAB\K4-DAY07-NguyenHoangTuyen-2A202602439\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: E:\VIN_LAB\K4-DAY07-NguyenHoangTuyen-2A202602439
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.15s ==============================
```

**Số lượng bài test vượt qua (pass):** **42 / 42** (100%)

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | Shopee hỗ trợ trả hàng hoàn tiền trong 15 ngày. | Khách hàng có thể gửi yêu cầu hoàn tiền trong 15 ngày kể từ khi nhận hàng. | Cao | 0.892 | Đúng |
| 2 | Chính sách bảo hành sản phẩm điện tử trên sàn thương mại. | Hướng dẫn đặt đồ ăn giao tận nơi trên ứng dụng. | Thấp | 0.174 | Đúng |
| 3 | Người bán Shopee Mall phải phản hồi khiếu nại đúng hạn. | Gian hàng Shopee Mall có nghĩa vụ xử lý khiếu nại trả hàng từ người mua theo quy định. | Cao | 0.835 | Đúng |
| 4 | Quy trình đổi trả hàng bị lỗi do nhà sản xuất. | Cách nấu món phở bò truyền thống Việt Nam thơm ngon. | Thấp | 0.048 | Đúng |
| 5 | Thời hạn trả hàng sản phẩm tươi sống là 24 giờ. | Đơn hàng thực phẩm tươi sống cần gửi yêu cầu trả hàng trong 24h. | Cao | 0.926 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả ở Cặp 1 và Cặp 5 đạt điểm rất cao (> 0.89) dù từ ngữ thay đổi ("hoàn tiền trong 15 ngày" vs "kể từ khi nhận hàng", "24 giờ" vs "24h"). Điều này chứng minh rằng mô hình Embedding không chỉ so khớp từ khóa rời rạc (lexical match) mà thực sự ánh xạ được cấu trúc ngữ nghĩa sâu (semantic representation), nhận biết được các thực thể số và ngữ cảnh đồng nghĩa trong tiếng Việt.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân với chiến lược **`SentenceChunker`** (max 3 câu/chunk) trên bộ dữ liệu `data/shopee-return-refund`:

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|:---:|:---|:---|:---:|:---:|:---|
| 1 | Với thực phẩm tươi sống và đông lạnh, người mua phải gửi yêu cầu Trả hàng/Hoàn tiền trong thời hạn bao lâu? | `buyer-return-conditions`: Đối với đơn hàng giao thực phẩm tươi sống & đông lạnh (trừ lý do Chưa nhận được hàng): Trong vòng 24 giờ kể từ lúc đơn hàng được cập nhật trạng thái ‘Giao hàng thành công’... | 0.684 | Có | Người mua phải gửi yêu cầu Trả hàng/Hoàn tiền trong vòng 24 giờ kể từ khi đơn hàng giao thành công. |
| 2 | Shopee có hỗ trợ yêu cầu đổi hàng không, và người mua có thể làm gì nếu hàng nhận được có vấn đề? | `buyer-return-conditions`: Nguyên tắc chung: Shopee hiện chưa hỗ trợ yêu cầu đổi hàng. Nếu hàng nhận được có vấn đề, bạn có thể từ chối nhận khi đồng kiểm hoặc gửi yêu cầu Trả hàng/Hoàn tiền... | 0.713 | Có | Shopee chưa hỗ trợ đổi hàng; người mua có thể từ chối nhận khi đồng kiểm hoặc gửi yêu cầu Trả hàng/Hoàn tiền. |
| 3 | Người mua có thể gửi yêu cầu Trả hàng/Hoàn tiền bằng những cách nào? | `buyer-return-request-guide`: Cách 1: Gửi yêu cầu trực tiếp tại trang đơn hàng. Cách 2: Gửi yêu cầu tại mục Trò Chuyện Với Shopee... | 0.693 | Có | Người mua có thể gửi trực tiếp tại trang đơn hàng hoặc gửi tại mục Trò Chuyện Với Shopee. |
| 4 | Sau khi Shopee chấp nhận hoàn tiền, tiền hoàn về thẻ tín dụng hoặc thẻ ghi nợ mất bao lâu? | `buyer-refund-timeline`: Thẻ tín dụng/ghi nợ: 7 - 14 ngày làm việc (tùy theo ngân hàng). Thẻ nội địa Napas: 2 - 5 ngày làm việc... | 0.742 | Có | Tiền hoàn về thẻ tín dụng hoặc thẻ ghi nợ sẽ mất từ 7 đến 14 ngày làm việc tùy ngân hàng. |
| 5 | Một yêu cầu hoàn tiền cần được phản hồi trong bao lâu? *(Filter: audience="seller")* | `seller-mall-return-obligations`: Trường hợp hoàn tiền ngay: Người Bán Shopee Mall có trách nhiệm phản hồi yêu cầu hoàn tiền trong vòng 02 ngày lịch kể từ khi nhận được yêu cầu... | 0.685 | Có | Người bán Shopee Mall cần phản hồi yêu cầu hoàn tiền trong vòng 02 ngày lịch kể từ khi nhận yêu cầu. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5** câu (100%)

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua việc chạy benchmark thực tế trên 5 query chung của nhóm, chiến lược `SentenceChunker` (3 câu/chunk) bảo toàn được tính nguyên vẹn của từng điều khoản chính sách và mốc thời gian (24 giờ, 02 ngày, 7-14 ngày). Đồng thời, bộ lọc `metadata_filter={"audience": "seller"}` ở Query 5 giúp loại trừ hoàn toàn các tài liệu hướng dẫn của người mua, đưa văn bản nghĩa vụ của người bán lên vị trí Top-1 chính xác tuyệt đối.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|:---|:---:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
