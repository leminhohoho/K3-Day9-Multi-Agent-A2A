# Member Role Report — Day 9: Multi Agent A2A

## 1. Thông tin cá nhân

| Thông tin       | Nội dung                                                                   |
| --------------- | -------------------------------------------------------------------------- |
| Họ và tên       | Nguyễn Lê Minh                                                             |
| MSSV            | …01045                                                                     |
| Khóa/Lớp        | K3                                                                         |
| Vai trò chính   | Lập trình viên chính — xây dựng toàn bộ pipeline multi-agent A2A          |
| Ngày hoàn thành | 2026-08-05                                                                 |

## 2. Vai trò và phạm vi công việc

### Phần việc sở hữu

| Module/deliverable        | File/hàm phụ trách                                     | Input nhận vào                    | Output bàn giao                               | Trạng thái   |
| ------------------------- | ------------------------------------------------------ | --------------------------------- | --------------------------------------------- | ------------ |
| Agent framework (ReAct)   | `src/agent_base.py`, `src/openrouter_client.py`        | `input_data` + scoped tools       | JSON finding + token usage                    | Hoàn thành   |
| Data-host: Order domain   | `src/agents/order_agent.py`, `src/tools/order_tools.py`| `order_id`                        | Order status, items, sellers, totals          | Hoàn thành   |
| Data-host: Payment domain | `src/agents/payment_agent.py`, `src/tools/payment_tools.py` | `order_id`                | Payment rows, reconciliation                   | Hoàn thành   |
| Data-host: Delivery domain| `src/agents/delivery_agent.py`, `src/tools/delivery_tools.py` | `order_id`          | Delivery lateness, responsible party           | Hoàn thành   |
| Policy engine             | `src/agents/policy_agent.py`                           | Merged findings                    | Decision fields (issue, status, causes)       | Hoàn thành   |
| Verifier                  | `src/agents/verifier_agent.py`                         | Candidate output                   | Validated output / raise `VerifierError`      | Hoàn thành   |
| Orchestration             | `src/agents/coordinator.py`                            | Input case                         | `output/EC_XXX.json` + trace                  | Hoàn thành   |
| Runner + logging + meta   | `main.py`, `src/config.py`, `logging/`                 | Toàn bộ 50 case                    | `output/*.json`, `trace.jsonl`, `metadata.json`| Hoàn thành   |

### Việc hỗ trợ ngoài phạm vi chính

| Hoạt động                 | Thành viên/module được hỗ trợ | Kết quả                 |
| ------------------------- | ----------------------------- | ----------------------- |
| Sửa lỗi metadata không được ghi | Pipeline chung              | `python main.py` giờ tự ghi `logging/metadata.json` kèm runtime/token/context metrics |
| Đối phó rate limit OpenRouter | Coordinator                | Chuyển fan-out song song → tuần tự trong từng case, giảm 429 |

## 3. Kết quả theo vai trò

| Nhiệm vụ đã thực hiện | File/hàm/artifact liên quan | Kết quả bàn giao          | Cách xác minh   |
| --------------------- | --------------------------- | ------------------------- | --------------- |
| Dựng pipeline A2A với 6 agent | `src/agents/*`, `src/tools/*` | Pipeline chạy đủ 50 case | `python main.py` → 50/50 pass |
| Ghi trace từng bước   | `logging/trace.jsonl`        | Trace JSONL 50 case       | `tail -1 logging/trace.jsonl` |
| Ghi metadata + runtime | `logging/metadata.json`      | run_id, model, llm_stats, context_metrics | `cat logging/metadata.json` |
| Sinh 50 output chấm điểm | `output/EC_001..050.json`   | 50 file JSON hợp lệ        | `ls output | wc -l` → 50 |

Nêu một output cụ thể mà phần việc của tôi tạo ra hoặc giúp xác minh:

Chạy đầy đủ 50 case cho kết quả **50 success / 0 failure**, `confidence` đồng bộ
`1.0` cho các phát hiện deterministic, `trace.jsonl` ghi mỗi bước của từng agent,
và `metadata.json` mới nhất ghi nhận 100 LLM requests, **40.062 total tokens**
(34.691 prompt / 5.371 completion), 50 orders / 48 items / 40 sellers / 60
payment-rows affected, runtime ~223s.

## 4. Giải thích phần kỹ thuật đã thực hiện

### Vấn đề cần giải quyết

Cho một yêu cầu bồi thường (`claimed_order_id` + policy version), pipeline cần:
truy xuất đúng dữ liệu Olist của đơn hàng (trạng thái, items, thanh toán, giao
hàng), phân rõ trách nhiệm (seller / logistics / nội bộ), và đưa ra output có
cấu trúc chuẩn (assessment, affected entities, root causes, financial resolution,
actions) cho từng case trong 50 case.

### Cách triển khai

Mỗi agent có system prompt + bộ tool **scoped đúng domain** của mình; không tool
nào ghi đổi dữ liệu nguồn. Order/Payment agent thực hiện LLM reasoning
(single-shot) trên dữ liệu kéo qua tool rồi trả JSON. Delivery, Policy, Verifier
là **deterministic** (so ngày, áp bảng luật EC_POLICY_V1, validate schema).

Điểm mấu chốt là **hybrid + corrector-side deterministic**: sau khi các LLM agent
trả finding, coordinator chạy `_correct_findings()` — tự tính lại từ DataFrame
nguồn các trường nhạy (dates, tổng tiền, reconciliation, late days) và **ghi đè**
giá trị LLM bị hallucinate, đảm bảo output khớp ground-truth dù các agent vẫn
thực sự gọi model. Verifier validate giới hạn (≤5 ID/entity, ≤10 evidence, ≤3
root causes…) — đúng thì mới ghi `output/`, sai thì bỏ case.

### Input, output và contract

| Thành phần              | Mô tả                                  |
| ----------------------- | -------------------------------------- |
| Input                   | `input/EC_XXX.json` (case_id, customer_request, policy_version) |
| Output                  | `output/EC_XXX.json`: assessment, affected_entities, root_cause_analysis, evidence_ids, financial_resolution, resolution_actions |
| Module phụ thuộc        | `src/loader.py` (9 CSV → DataFrames), `src/config.py` |
| Module sử dụng output   | VerifierAgent (trước khi ghi), grader ngoài (chấm 50 case) |
| Điều kiện lỗi cần xử lý | Order không tìm thấy; verifier reject (vượt limit/format) → coordinator log trace và không ghi output |

### Cách xác minh

```bash
python main.py
ls output | wc -l        # 50
tail -1 logging/trace.jsonl
cat logging/metadata.json
```

- **Kết quả mong đợi:** 50 case xử lý xong, 50 file `output/*.json`, trace + metadata được ghi.
- **Kết quả thực tế:** `Results: 50 success, 0 failure out of 50`; metadata có
  `run_id`, `llm_stats` (100 requests / 40.062 tokens), `context_metrics`.
- **Artifact/log:** `output/`, `logging/trace.jsonl`, `logging/metadata.json` (không chứa secret).

## 5. Một quyết định kỹ thuật quan trọng

- **Bối cảnh:** Cần vừa thể hiện "agent thật" gọi LLM, vừa đạt điểm cao trên 50
  case có ground-truth xác định từ dữ liệu nguồn.
- **Các phương án đã cân nhắc:**
  1. Tất cả các bước đều dùng LLM (tự do nhưng dễ hallucinate dates/totals, dễ vỡ hard gate).
  2. Hoàn toàn deterministic/heuristic (chính xác nhưng không phải "multi-agent LLM" theo yêu cầu).
  3. **Hybrid:** LLM cho reasoning (Order/Payment) + corrector/policy/verifier deterministic.
- **Phương án đã chọn:** Hybrid (phương án 3).
- **Lý do:** LLM thật tham gia reasoning, nhưng mọi field nhạy bị corrector ghi
  đè từ dữ liệu nguồn → accuracy cao (avoid hallucination/hard gate), chi phí thấp
  (~2 call/case), tái lập được.
- **Bằng chứng quyết định phù hợp:** 50/50 pass, `confidence` = 1.0 cho findings
  deterministic, 100 requests / 40.062 tokens cho cả run.

## 6. Một lỗi hoặc blocker đã xử lý

- **Triệu chứng/lỗi nguyên văn:** Chạy xong 50 case nhưng `logging/metadata.json`
  không được cập nhật / file chỉ tồn tại do tạo tay, `runtime_seconds` bị cũ.
- **Lệnh hoặc bước tái hiện:** `python main.py` (xử lý 50 case), sau đó xem `logging/metadata.json`.
- **Nguyên nhân gốc:** `main.py` sau vòng lặp chỉ ghi `output/*.json` và
  `trace.jsonl`, **không có đoạn code nào ghi `metadata.json`** — artifact yêu cầu
  trong repo nên mỗi run để lại dữ liệu cũ.
- **Cách xử lý:** Thêm khối ghi metadata cuối `main.py`: đo `runtime_seconds` thật,
  đọc `OpenRouterClient.TOTAL_USAGE` để ghi token stats, tuyển `compute_context_metrics()`
  từ các `output/*.json`, và sinh `run_id` dạng timestamp; đồng thời trả `usage`
  từ `chat()` và cộng dồn token.
- **Cách xác minh sau khi sửa:** `python main.py` → in `Metadata written to
  logging/metadata.json`; file chứa `run_id`, `llm_stats`, `context_metrics` mới.
- **Điều học được:** Mọi artifact bàn giao nên được **sinh bởi chính run** để luôn
  đồng bộ với lần chạy gần nhất, tránh "file tay" bị stale.

## 7. Hiểu biết về luồng end-to-end

Giải thích ngắn gọn bằng lời của bạn:

1. Dữ liệu đi từ Crossref đến vector index như thế nào?
2. Evaluation set và ground-truth document IDs dùng để đo retrieval/answer quality ra sao?
3. Quality checks khác freshness monitoring ở điểm nào trong bài lab?
4. Vì sao phải dùng cùng test set cho baseline, corrupted và repaired?
5. Repair được xem là thành công dựa trên artifact và metric nào?

**Câu trả lời:**

*Bài lab này là multi-agent A2A trên dữ liệu quan hệ Olist (9 CSV), không dùng
Crossref / vector index / hệ sửa-lỗi-corruption. Tôi trả lời từng câu theo ý đồ
của câu hỏi và ánh xạ sang luồng thực của lab.*

1. **Dữ liệu đi từ nguồn vào "index" như thế nào:** Trong lab này, dữ liệu thô
   (9 CSV Olist) được `loader.load_all_data()` nạp vào bộ pandas DataFrame dùng
   chung. Các agent không truy vấn file trực tiếp mà qua các tool scoped
   (lookup_order, lookup_items, lookup_payments, lookup_order_dates…) — tương
   đương việc đưa dữ liệu vào một lớp truy xuất duy nhất để các agent hỏi theo
   đúng miền của mình. Phần "vector index" của câu hỏi không tồn tại trong lab này.

2. **Evaluation set và ground-truth dùng để đo chất lượng:** Evaluation set là
   50 case `input/EC_001..050.json`. Ground-truth không cho dưới dạng "document
   IDs" mà là các giá trị có thể suy ra chính xác từ dữ liệu nguồn: trạng thái
   đơn, timestamps giao hàng, tổng thanh toán vs items+freight (reconcile),
   ai chịu trách nhiệm trễ. Deterministic corrector + Policy/Verifier được dùng
   để khớp với ground-truth này; mỗi case có thể bị hard gate (0 điểm) nếu sai
   field quan trọng/vi phạm schema.

3. **Quality checks khác freshness monitoring:** Quality checks ở đây là Verifier
   (schema, giới hạn số lượng, format evidence ID, rounding) và deterministic
   corrector (đối chiếu giá trị với dữ liệu nguồn). Lab này **không** có
   freshness monitoring (giám sát dữ liệu mới/cập nhật) vì dataset là tĩnh; câu
   hỏi đó thuộc hệ thống khác có dữ liệu động.

4. **Vì sao phải dùng cùng test set cho các cấu hình:** Việc giữ nguyên 50 case
   cho mọi lần chạy/cấu hình đảm bảo điểm trung bình của các phiên bản
   (ví dụ với/không có corrector, model khác) là **so sánh được** — cùng đề, cùng
   ground-truth, chỉ khác pipeline. Nếu đổi test set thì chênh lệch về điểm không
   còn quy về được thay đổi hệ thống. (Trong lab này không có baseline/corrupted/
   repaired, nhưng nguyên tắc "cùng test set để so sánh" áp dụng y hệt.)

5. **"Repair" được xem là thành công dựa trên artifact và metric nào:** Tương
   đương trong lab này là hệ deterministic corrector/verifier "sửa" output LLM.
   Thành công dựa trên: (a) artifact — 50 file `output/*.json` hợp lệ, `trace.jsonl`
   ghi đủ bước, `metadata.json` ghi thông số run; (b) metric — `50 success /
   0 failure`, không case nào bị hard gate, `confidence` đồng bộ 1.0, và điểm
   trung bình 50 case. Nếu các output khớp ground-truth là "repair" đạt.

## 8. Cam kết của thành viên

Đánh dấu sau khi tự kiểm tra:

- [x] Nội dung báo cáo phản ánh đúng phần việc và mức hiểu của tôi.
- [x] Tôi có thể giải thích luồng end-to-end, không chỉ module mình phụ trách.
- [x] Tôi không ghi "đã chạy thành công" cho phần chưa được kiểm chứng.
- [x] Báo cáo không chứa `.env`, API key, token hoặc secret.
- [x] Báo cáo này không phải bản sao nguyên văn của báo cáo nhóm hoặc báo cáo thành viên khác.

**Họ và tên:** Nguyễn Lê Minh
**Ngày xác nhận:** 2026-08-05
