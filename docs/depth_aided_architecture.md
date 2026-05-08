# Tracking Hỗ Trợ Bằng Depth Và Mapping Keyframe Chỉ Dùng DUSt3R

Tài liệu này tổng hợp kiến trúc SLAM mới trong fork này. Thay đổi chính là DUSt3R không còn được dùng cho mọi cặp frame liên tiếp trong giai đoạn tracking. Thay vào đó, depth dense của Waymo được dùng để hỗ trợ tracking cục bộ, còn DUSt3R chỉ được dùng khi cần chèn keyframe vào map.

## Mục Tiêu

- Giảm chi phí tracking theo frame bằng cách tránh chạy DUSt3R trên mọi cặp RGB.
- Giữ global Gaussian map xoay quanh keyframe và vẫn được hỗ trợ bởi DUSt3R.
- Dùng depth map của Waymo làm hỗ trợ tracking cục bộ và làm fallback khi chèn keyframe bằng DUSt3R thất bại.
- Tái dùng các code path sẵn có theo kiểu OpenGS-SLAM và MonoGS, không viết lại logic back-projection hoặc optimization đã có.

## Pipeline Mới

### Khởi tạo frame 0

1. Frontend load RGB, depth và GT pose thông qua `Camera.init_from_dataset()`.
2. Pose đầu tiên được khởi tạo từ GT, giữ nguyên giả định khởi động ban đầu của hệ thống.
3. Frontend gửi message `init` kèm depth map đã được filter từ dataset.
4. Backend khởi tạo global Gaussian map bằng đường RGB-D đã có sẵn:
   - `BackEnd.add_next_kf()`
   - `GaussianModel.extend_from_pcd_seq()`
   - `GaussianModel.create_pcd_from_image(..., depthmap=...)`

Không còn dùng cặp DUSt3R self-pair để khởi tạo.

### Tracking frame thường

1. Frontend khởi tạo pose hiện tại bằng prior constant velocity.
   - Frame 1 dùng lại pose của frame trước.
   - Từ frame 2 trở đi, hệ thống dùng hai pose gần nhất để dự đoán transform world-to-camera hiện tại.
2. Frontend tạo tracking map chỉ dùng để render bằng cách nối:
   - global Gaussian map hiện tại từ backend;
   - một local buffer nhỏ gồm các Gaussian được back-project từ depth của những frame vừa tracking xong.
3. Pose hiện tại được refine bằng photometric tracking loss sẵn có:
   - `get_loss_tracking()`
   - `update_pose()`
4. Frontend render lại global map để tính visibility và kiểm tra overlap keyframe, nhờ đó `occ_aware_visibility` vẫn tương thích với số lượng Gaussian của global map.
5. Sau khi tracking thành công, depth của frame hiện tại được back-project và lưu vào local buffer.

Tracking frame thường không gọi DUSt3R.

### Tạo keyframe

1. Việc chọn keyframe vẫn dùng logic overlap và motion hiện có:
   - `FrontEnd.is_keyframe()`
   - `FrontEnd.add_to_window()`
2. Khi một frame trở thành keyframe, frontend chạy DUSt3R cho cặp keyframe đó.
3. Frontend cập nhật DUSt3R pointmaps, màu ảnh, reciprocal matches và trạng thái adaptive scale nếu có matches hợp lệ.
4. Backend chèn keyframe vào global map bằng:
   - `BackEnd.add_next_kf_dust3r()`
   - `GaussianModel.create_pcd_from_dust3r()`
5. Nếu thiếu dữ liệu DUSt3R hoặc chèn pointmap thất bại, backend fallback về:
   - `BackEnd.add_next_kf(..., depth_map=...)`

Global Gaussian map vẫn chỉ được cập nhật thông qua keyframe.

## Local Depth Gaussian Buffer

Local buffer chỉ dùng để render trong tracking và không mutate backend map hoặc optimizer state.

- Depth map được filter bằng `Tracking.depth_min` và `Tracking.depth_max`.
- Giá trị depth không hợp lệ được set về `0`, để Open3D bỏ qua khi tạo point cloud.
- Back-projection tái dùng `GaussianModel.create_pcd_from_image(..., depthmap=...)`.
- Các tensor trả về được detach và lưu theo từng frame.
- Kích thước buffer được giới hạn bởi `Tracking.depth_tracking_buffer`.
- Một `GaussianModel` tạm thời được lắp ráp cho tracking render bằng cách concat tensor global với các tensor local trong buffer.

Thiết kế này cố tình tránh dùng `extend_from_pcd()` cho non-keyframe, vì hàm đó mutate global Gaussian model có optimizer backing.

## Cấu Hình

Behavior mới được cấu hình trong `configs/mono/waymo/base_config.yaml`:

```yaml
Tracking:
  pose_init: "constant_velocity"
  use_dust3r_every_frame: False
  wait_for_keyframe_backend: True
  depth_tracking_buffer: 3
  use_depth_local_map: True
  depth_min: 0.1
  depth_max: 80.0
```

Giá trị mặc định hiện tại:

- `pose_init: "constant_velocity"`: dùng motion prior thay cho relative pose từ DUSt3R đối với frame thường.
- `use_dust3r_every_frame: False`: DUSt3R chỉ dành cho keyframe.
- `wait_for_keyframe_backend: True`: frontend chờ backend chèn xong keyframe trước khi xử lý frame tiếp theo, tránh bỏ lỡ keyframe khi tracking nhanh hơn mapping.
- `depth_tracking_buffer: 3`: giữ ba entry local depth Gaussian gần nhất.
- `use_depth_local_map: True`: bật local depth support chỉ dùng để render.
- `depth_min` và `depth_max`: filter các giá trị Waymo depth không hợp lệ hoặc quá cực đoan trước khi back-project.

## Điểm Triển Khai Quan Trọng

- Các thay đổi frontend nằm trong `utils/slam_frontend.py`.
  - Helper local depth buffer tạo và kết hợp các Gaussian chỉ dùng để render.
  - `tracking()` hiện dùng constant-velocity initialization và photometric refinement.
  - DUSt3R chỉ được gọi bởi helper dành cho keyframe.
- Các thay đổi backend nằm trong `utils/slam_backend.py`.
  - `init` dùng RGB-D/depth back-projection.
  - `keyframe` dùng pointmap DUSt3R khi có dữ liệu hợp lệ, nếu không thì fallback về depth.
- Các hàm sẵn có cho loss, pose update, keyframe window, tạo Gaussian và mapping optimization vẫn được giữ lại.

## Kiểm Chứng

Chạy syntax check:

```bash
python -m py_compile slam.py utils/slam_frontend.py utils/slam_backend.py gaussian_splatting/scene/gaussian_model.py
```

Chạy sanity check ngắn cho local depth back-projection:

```bash
python -c "from utils.config_utils import load_config; from utils.dataset import load_dataset; from utils.camera_utils import Camera; from utils.slam_frontend import FrontEnd; from gaussian_splatting.scene.gaussian_model import GaussianModel; from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2; from munch import munchify; cfg=load_config('configs/mono/waymo/405841.yaml'); cfg['Training']['monocular']=True; ds=load_dataset(munchify(cfg['model_params']), cfg['model_params']['source_path'], cfg); proj=getProjectionMatrix2(0.01,100.0,ds.fx,ds.fy,ds.cx,ds.cy,ds.width,ds.height).transpose(0,1).to(ds.device); cam=Camera.init_from_dataset(ds,0,proj); cam.compute_grad_mask(cfg); fe=FrontEnd(cfg,None); fe.set_hyperparams(); fe.gaussians=GaussianModel(3,config=cfg); t=fe._build_depth_gaussian_tensors(cam); print('local depth points:', t['xyz'].shape[0])"
```

Chạy pipeline SLAM:

```bash
CUDA_VISIBLE_DEVICES=0 python slam.py --config configs/mono/waymo/405841.yaml
```

Để smoke run nhanh hơn, tạm thời giảm các giá trị config sau:

```yaml
Results:
  eval_rendering: False
  color_refinement: False

Training:
  init_itr_num: 50
  tracking_itr_num: 20
  mapping_itr_num: 20
```

Kỳ vọng khi chạy:

- Frame 0 khởi tạo từ depth của dataset.
- Frame thường tracking mà không chạy DUSt3R inference.
- DUSt3R chỉ được kích hoạt khi có keyframe mới.
- Nếu chèn bằng DUSt3R thất bại, backend log fallback về depth và tiếp tục chạy.
