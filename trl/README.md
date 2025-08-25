# TRL

GitHub link: [trl](https://github.com/huggingface/trl)

## Env
``` bash
# https://huggingface.co/docs/trl/installation
git clone https://github.com/huggingface/trl.git
cd trl
conda create -n trl python=3.10
conda activate trl
pip install -e .

# custom
```

## Data
``` json
// 3D medical images
[
    {
        "id": "report_generation_0",
        "messages": [
            {
                "role": "user",
                "content": "<video>\nGenerate..."
            },
            {
                "role": "assistant",
                "content": "Findings: ..."
            }
        ],
        "videos": ["/.../train_1_a_1.npy"]
    }
]
```

## Code

``` python
# Read video data through `.npy` files.
# Replace `~/anaconda3/envs/swift/lib/python3.10/site-packages/qwen_vl_utils/vision_process.py` with `swift/src/vision_process.py`.
...
def temporal_resize(
    video: torch.Tensor,
    target_T: int,
    method: str,          # 'sample' | 'linear'
) -> torch.Tensor:
    """
    将 (T, C, H, W) 的张量在时间维调整到 target_T。

    参数
    ----
    video     : Tensor  (T, C, H, W)
    target_T  : int     目标帧数
    method    : str
        - 'sample'      均匀抽帧，只支持下采样
        - 'linear'      仅对时间维做 1D 线性插值
    """
    T, C, H, W = video.shape
    if T <= target_T:
        return video  # 不做任何处理

    # ---------- sample ----------
    if method == "sample":
        # 均匀抽帧
        idx = torch.linspace(0, T - 1, target_T, device=video.device).round().long()
        return video.index_select(0, idx)

    # ---------- 插值 ----------
    elif method == "linear":
        x = video.permute(1, 2, 3, 0).contiguous()      # (C, H, W, T)
        x = x.view(-1, 1, T)                            # (C*H*W, 1, T)
        x = torch.nn.functional.interpolate(x, size=target_T, mode="linear", align_corners=False)  # (C*H*W, 1, target_T)
        x = x.view(C, H, W, target_T).permute(3, 0, 1, 2).contiguous()  # (target_T, C, H, W)
        return x


import numpy as np
def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video = np.load(ele["video"])  # [1, T, H, W]
        if video.ndim == 3:  # [T, H, W]
            video = video[np.newaxis, ...]
        video = torch.tensor(video).float()
        video = video.permute(1, 0, 2, 3)  # [T, 1, H, W]
        video = video.expand(-1, 3, -1, -1)  # [T, 3, H, W], [0, 255]
        # print("before:", video.shape, video.dtype, video.min(), video.max())

        # normalize to [0, 255]
        # HU_MIN, HU_MAX = -1000, 1000
        # video = torch.clamp(video, HU_MIN, HU_MAX)
        # video = (video - HU_MIN) / (HU_MAX - HU_MIN) * 255
        # print("after normalize:", video.shape, video.dtype, video.min(), video.max())

        # resize to [resized_depth, C, resized_height, resized_width]
        resized_depth, resized_height, resized_width = 32, 364, 364
        video = temporal_resize(video, target_T=resized_depth, method="linear")  # [T, 3, H, W]
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        # print("after resize:", video.shape, video.dtype, video.min(), video.max())
        return video
...
```
