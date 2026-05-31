# Các cải tiến so với MonoGS

*Tài liệu tổng hợp những khác biệt phương pháp giữa hệ thống SLAM trong repo này và MonoGS (Matsuki et al., CVPR 2024 — "Gaussian Splatting SLAM"). Văn phong báo cáo: mỗi cải tiến được trình bày theo trình tự **Động lực → Phương pháp → Đóng góp**, kèm tham chiếu mã nguồn để truy vết.*

---

## 0. Bối cảnh

MonoGS là hệ SLAM dùng 3D Gaussian Splatting làm biểu diễn bản đồ duy nhất, với front-end tracking thuần quang học (photometric) tối ưu trực tiếp pose camera trên ảnh render, và back-end mapping tối ưu chung pose + Gaussian trong một cửa sổ keyframe. Điểm yếu cốt lõi của MonoGS ở chế độ **monocular** là: (i) khởi tạo bản đồ phụ thuộc vào back-projection độ sâu render — vốn không đáng tin ở những frame đầu khi bản đồ còn rỗng; (ii) nhập nhằng tỉ lệ (scale ambiguity) đặc trưng của monocular; (iii) tracking quang học dễ trôi (drift) khi chuyển động nhanh hoặc xoay mạnh.

Hệ thống trong repo này (nền tảng OpenGS-SLAM, mở rộng thêm) giữ lại khung 3DGS-SLAM của MonoGS nhưng thay thế lõi khởi tạo/định vị bằng tiên nghiệm hình học học sâu từ **DUSt3R**, đồng thời bổ sung một loạt cơ chế thích nghi (adaptive) và một chiến lược mở rộng bản đồ có dẫn hướng. Toàn bộ pipeline dưới đây được trình bày như một hệ thống thống nhất, đối chiếu với MonoGS làm mốc.

---

## 1. Khởi tạo bản đồ bằng pointmap DUSt3R thay vì back-projection độ sâu

**Động lực.** MonoGS khởi tạo Gaussian bằng cách render độ sâu từ bản đồ hiện có rồi back-project qua nội tham số camera. Ở monocular, frame đầu chưa có bản đồ nên độ sâu khởi tạo phải dựa vào noise/giá trị ngẫu nhiên — tạo ra phụ thuộc vòng (circular dependency) và khởi tạo kém ổn định.

**Phương pháp.** Mỗi keyframe được khởi tạo trực tiếp từ **pointmap 3D metric** do DUSt3R dự đoán cho cặp ảnh (frame hiện tại ↔ frame tham chiếu):
- Suy luận DUSt3R + global alignment trên cặp ảnh ở độ phân giải 512 (`get_result`, [utils/dust3r_utils.py:86-128](utils/dust3r_utils.py#L86-L128)).
- Lọc điểm hợp lệ: `valid = isfinite(p) ∧ confidence_mask`, trong đó `confidence_mask = scene.get_masks()` của DUSt3R ([gaussian_splatting/scene/gaussian_model.py:244-246](gaussian_splatting/scene/gaussian_model.py#L244-L246)).
- Hiệu chỉnh tỉ lệ (mục 3), biến đổi về hệ thế giới qua pose tham chiếu `T`, random downsample, rồi khởi tạo tham số Gaussian: scale từ khoảng cách k-NN (`distCUDA2`), rotation = quaternion đơn vị, opacity = 0.5 ([gaussian_model.py:212-319](gaussian_splatting/scene/gaussian_model.py#L212-L319)).

**Đóng góp.** Loại bỏ phụ thuộc vòng vào độ sâu render lúc khởi tạo; bản đồ ban đầu là đám mây điểm metric có cấu trúc hình học thực, giúp hội tụ nhanh và ổn định hơn ở chế độ RGB-only.

---

## 2. Định vị keyframe dựa trên global alignment của DUSt3R

**Động lực.** Tracking quang học của MonoGS dễ trôi khi chuyển động lớn giữa hai keyframe, vì nó tối ưu cục bộ quanh pose khởi tạo.

**Phương pháp.** Hệ thống dùng kiến trúc **hybrid**:
- **Tracking frame-to-frame** vẫn là tối ưu quang học trên ảnh render (giữ như MonoGS), với loss L1 có trọng số opacity ([utils/slam_frontend.py:267-334](utils/slam_frontend.py#L267-L334), `get_loss_tracking` tại [utils/slam_utils.py](utils/slam_utils.py)).
- **Pose giữa keyframe và frame tham chiếu** lấy trực tiếp từ **global alignment của DUSt3R** (`scene.get_im_poses()`), không qua tối ưu quang học. Frame tham chiếu được chọn bằng độ lệch khỏi ma trận đơn vị nhỏ nhất; pose tương đối suy ra từ đó ([dust3r_utils.py:95-110](utils/dust3r_utils.py#L95-L110)).
- Điểm tương ứng 3D giữa hai frame được tìm bằng `find_reciprocal_matches` ([dust3r_utils.py:113-126](utils/dust3r_utils.py#L113-L126)), phục vụ ước lượng tỉ lệ (mục 3).

**Tùy chọn khởi tạo pose.** `pose_init` hỗ trợ `'previous_pose'` (mặc định hiện tại) và `'constant_velocity'` (ngoại suy chuyển động SE(3) hai frame) ([slam_frontend.py:187-208](utils/slam_frontend.py#L187-L208)).

**Đóng góp.** Pose keyframe có tiên nghiệm hình học mạnh, giảm trôi tích lũy trên các đoạn chuyển động khó so với tracking quang học thuần.

---

## 3. Adaptive Scale Mapper — giải nhập nhằng tỉ lệ monocular

**Động lực.** DUSt3R xuất pointmap "metric" nhưng tỉ lệ của nó không nhất quán với tỉ lệ bản đồ SLAM đang duy trì. Nếu chèn thẳng, bản đồ sẽ co/giãn sai.

**Phương pháp.** Ước lượng một **scale divisor** mỗi keyframe bằng tỉ số quãng đường dịch chuyển:
```
scale_divisor = ‖t_dust3r‖ / ‖t_map‖
scale_divisor = clip(scale_divisor, dust3r_scale_min, dust3r_scale_max)
```
trong đó `t_dust3r` là tịnh tiến tương đối từ DUSt3R, `t_map` là tịnh tiến giữa hai keyframe theo pose bản đồ SLAM (`estimate_keyframe_dust3r_scale`, [slam_frontend.py:210-231](utils/slam_frontend.py#L210-L231)). Pointmap sau đó được chia cho scale trước khi chèn (`pts = pts * (1/scale)`, [gaussian_model.py:258-261](gaussian_splatting/scene/gaussian_model.py#L258-L261)). Khoảng kẹp mặc định `dust3r_scale_min=0.05`, `dust3r_scale_max=20.0` chặn các ước lượng phân kỳ.

**Đóng góp.** Hợp nhất tỉ lệ pointmap DUSt3R với tỉ lệ bản đồ toàn cục, duy trì nhất quán hình học khi liên tục chèn Gaussian từ nhiều keyframe — điều MonoGS không cần xử lý vì không dùng tiên nghiệm ngoài.

---

## 4. Điều chỉnh tốc độ học thích nghi theo góc xoay (θ-based Adaptive LR)

**Động lực.** Khi camera xoay mạnh giữa hai keyframe, vùng nhìn thay đổi lớn; giữ nguyên lịch học có thể gây cập nhật Gaussian không ổn định.

**Phương pháp.** Tính góc xoay θ giữa hai keyframe liên tiếp từ vết của ma trận xoay tương đối:
```
cosθ = clamp((trace(R_lastᵀ R_now) − 1) / 2, −1, 1);  θ = deg(acos(cosθ))
```
([slam_frontend.py:94-102](utils/slam_frontend.py#L94-L102)). Ở back-end, nếu θ ≥ 2°, giảm `iteration_count` tích lũy theo:
```
iteration_count ← iteration_count · (1 − √(θ/90))
```
rồi cập nhật lại lịch learning rate ([slam_backend.py:559-567](utils/slam_backend.py#L559-L567)). Ở θ=90° hệ số về 0 (reset), ở θ nhỏ chỉ giảm nhẹ.

**Đóng góp.** Bộ điều tiết LR theo tín hiệu chuyển động, ổn định tối ưu khi viewpoint thay đổi nhanh — không có trong MonoGS.

---

## 5. Kích thước điểm thích nghi theo độ sâu (Adaptive Point Size)

**Động lực.** Một kích thước Gaussian khởi tạo cố định gây dấu chân thị giác không đều giữa cảnh gần và xa.

**Phương pháp.** Khi `adaptive_pointsize=True`, kích thước điểm được co giãn theo độ sâu trung vị của frame:
```
point_size = min(0.05, point_size · median(depth))
```
([gaussian_model.py:141-143](gaussian_splatting/scene/gaussian_model.py#L141-L143)). Tham số nền `point_size=0.01`.

**Đóng góp.** Giữ footprint thị giác của Gaussian khởi tạo nhất quán theo độ sâu cảnh.

---

## 6. Coverage-Residual Guided Map Expansion (CGE)

> **Trạng thái:** Cải tiến mới nhất, có cờ bật/tắt trong config (`coverage_guided_expansion.enabled`). Tắt → fallback chính xác về hành vi chèn pointmap toàn phần.

**Động lực.** Cơ chế gốc chèn **toàn bộ** pointmap DUSt3R (sau downsample) mỗi keyframe, bất kể vùng đó bản đồ đã render tốt hay chưa. Trên các sequence khó (chuyển động nhanh, motion blur, méo ống kính như TUM `fr1_desk`), điều này vừa gây dư thừa Gaussian ở vùng đã hội tụ, vừa không tập trung lấp các vùng tái dựng kém (under-reconstructed) — biểu hiện qua PSNR render sụp đổ trong khi ATE vẫn tốt (chứng tỏ lỗi nằm ở bản đồ, không phải pose).

**Phương pháp.** Trước khi chèn pointmap của keyframe mới (bỏ qua lần init khi bản đồ còn rỗng), render bản đồ hiện tại từ chính pose của keyframe đó và xây một **insertion mask 2D** trên lưới pointmap:
```
need_coverage   = (opacity_render  < τ_cov)        # bản đồ chưa phủ đủ
need_appearance = (rgb_residual    > τ_rgb)        # render sai màu
need_depth      = (depth_residual  > τ_depth)      # (tùy chọn) sai hình học
insertion = (need_coverage ∨ need_appearance ∨ need_depth) ∨ (opacity < floor)
final_mask = confidence_mask_DUSt3R ∧ insertion
```
Chỉ các điểm pointmap có pixel nằm trong `final_mask` mới được chèn. Sàn `min_opacity_floor` đảm bảo vùng gần như rỗng luôn được lấp bất kể residual. Các map render được resize song tuyến về độ phân giải pointmap để căn pixel.

- Hàm tính mask: `compute_coverage_residual_mask` ([utils/slam_utils.py](utils/slam_utils.py)).
- Tích hợp ở back-end: `add_next_kf_dust3r` / `_apply_coverage_residual_guidance`, AND insertion mask vào confidence mask của DUSt3R rồi log tỉ lệ điểm giữ lại ([utils/slam_backend.py:78-185](utils/slam_backend.py#L78-L185)).
- Cấu hình + ngưỡng: khối `coverage_guided_expansion` trong [configs/mono/tum/base_config.yaml](configs/mono/tum/base_config.yaml).

**Quan sát thực nghiệm (kiểm chứng chạy thử trên `fr1_desk`).** Mỗi keyframe chỉ giữ lại ~16–40% số điểm ứng viên (phần còn lại bị loại vì vùng đó đã render đủ tốt); tỉ lệ giữ tăng dần theo keyframe sau — hợp lý với việc camera liên tục quan sát vùng mới chưa được phủ.

**Đóng góp.** Chuyển từ "chèn mù theo keyframe" sang **chèn có dẫn hướng theo phản hồi coverage–residual**: tập trung Gaussian vào vùng tái dựng kém, giảm dư thừa ở vùng đã tốt. Nhắm trực tiếp vào điểm yếu render trên sequence khó mà không phình bản đồ trên sequence dễ.

---

## 7. Mở rộng hỗ trợ dataset & quy ước cấu hình

**Động lực.** Repo nền tập trung vào cảnh ngoài trời Waymo; cần đánh giá trên benchmark TUM RGB-D trong nhà (chuẩn so sánh SLAM phổ biến).

**Phương pháp.**
- `TUMParser` + `TUMDataset` với liên kết (association) ảnh–độ sâu–pose theo timestamp ([utils/dataset.py](utils/dataset.py)), cùng các config `configs/mono/tum/{fr1_desk,fr2_xyz,fr3_office}.yaml` kế thừa `base_config.yaml` qua `inherit_from` (merge đệ quy).
- Tham số `num_frames` (đọc trong `MonocularDataset.__init__`, [utils/dataset.py:239-247](utils/dataset.py#L239-L247)) và cờ CLI `--num_frames` ([slam.py:224,234-235](slam.py#L224)) cho phép giới hạn số frame chạy — phục vụ thử nghiệm/ablation nhanh; `num_frames=-1` chạy toàn bộ.
- Script tải dữ liệu `scripts/download_tum.sh`, script chạy hàng loạt `run_tum.sh`.

**Đóng góp.** Đưa pipeline (vốn cho cảnh ngoài trời) sang đánh giá trên TUM trong nhà, mở đường cho các ablation về tracking/mapping.

---

## Bảng tổng hợp đối chiếu

| Khía cạnh | MonoGS | Hệ thống trong repo này |
|---|---|---|
| Khởi tạo bản đồ | Back-project độ sâu render | **Pointmap metric DUSt3R** + hiệu chỉnh tỉ lệ |
| Tracking frame-to-frame | Quang học (render loss) | Quang học (giữ nguyên) |
| Pose keyframe | Từ tối ưu quang học | **Global alignment DUSt3R** |
| Nhập nhằng tỉ lệ monocular | Trung vị độ sâu render | **Adaptive Scale Mapper** (tỉ số quãng đường, kẹp [0.05, 20]) |
| Learning rate | Lịch cố định | **Thích nghi theo góc xoay θ** (θ≥2° → giảm) |
| Kích thước Gaussian khởi tạo | Cố định | **Thích nghi theo độ sâu trung vị** |
| Chèn Gaussian mỗi keyframe | (N/A — không dùng tiên nghiệm ngoài) | Mặc định: toàn bộ pointmap; **CGE: chèn có dẫn hướng coverage–residual** |
| Chọn keyframe | Covisibility (overlap) + dịch chuyển | Tương tự: overlap Jaccard/Szymkiewicz + ngưỡng dịch chuyển theo độ sâu trung vị |
| Tỉa Gaussian (prune) | Theo covisibility | Chế độ `'slam'`: chỉ tỉa Gaussian từ keyframe mới với `n_obs ≤ 3` |
| Chính quy hóa hình dạng | — | **Isotropic loss** (trọng số 10) đẩy Gaussian về dạng cầu |
| Dataset | (gốc) | **Bổ sung TUM RGB-D** + giới hạn `num_frames` |

---

## Tham chiếu mã nguồn nhanh

| Cơ chế | Vị trí |
|---|---|
| Suy luận DUSt3R + global alignment | [utils/dust3r_utils.py:86-128](utils/dust3r_utils.py#L86-L128) |
| Khởi tạo Gaussian từ pointmap | [gaussian_splatting/scene/gaussian_model.py:212-319](gaussian_splatting/scene/gaussian_model.py#L212-L319) |
| Adaptive Scale Mapper | [utils/slam_frontend.py:210-231](utils/slam_frontend.py#L210-L231) |
| Tính góc xoay θ | [utils/slam_frontend.py:94-102](utils/slam_frontend.py#L94-L102) |
| Adaptive LR theo θ | [utils/slam_backend.py:559-567](utils/slam_backend.py#L559-L567) |
| Adaptive point size | [gaussian_splatting/scene/gaussian_model.py:141-143](gaussian_splatting/scene/gaussian_model.py#L141-L143) |
| CGE — tính mask | [utils/slam_utils.py](utils/slam_utils.py) (`compute_coverage_residual_mask`) |
| CGE — tích hợp back-end | [utils/slam_backend.py:78-185](utils/slam_backend.py#L78-L185) |
| Loss tracking/mapping (RGB & RGBD) | [utils/slam_utils.py](utils/slam_utils.py) |
| Chọn keyframe & cửa sổ covisibility | [utils/slam_frontend.py:336-425](utils/slam_frontend.py#L336-L425) |
| Vòng tối ưu mapping & tỉa | [utils/slam_backend.py](utils/slam_backend.py) (`map`) |
| Hỗ trợ TUM + giới hạn frame | [utils/dataset.py](utils/dataset.py) |

---

*Lưu ý: các con số ngưỡng nêu trên lấy từ `configs/mono/tum/base_config.yaml` tại thời điểm viết. Các quan sát thực nghiệm về CGE là từ một lần chạy kiểm chứng (`fr1_desk`, số frame giới hạn) nhằm xác nhận pipeline hoạt động, chưa phải kết quả ablation đầy đủ.*
