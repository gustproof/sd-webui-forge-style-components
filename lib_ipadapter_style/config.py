from dataclasses import dataclass
from torchvision.transforms.v2 import (
    Resize,
    Compose,
    InterpolationMode,
    ToTensor,
    Normalize,
    CenterCrop,
)


@dataclass
class StyleComponentsConfig:
    style_extractor_path: str
    pca_path: str
    adapter_path: str
    image_preprocessor: str


def MyResize(area, d):
    def f(im):
        w, h = im.size
        s = (area / w / h) ** 0.5
        wd, hd = int(s * w / d), int(s * h / d)
        e = lambda a, b: 1 - min(a, b) / max(a, b)
        wd, hd = min(
            (
                (ww * d, hh * d)
                for ww, hh in [(wd + i, hd + j) for i in (0, 1) for j in (0, 1)]
                if ww * d * hh * d <= area
            ),
            key=lambda wh: e(wh[0] / wh[1], w / h),
        )

        return Compose(
            [
                Resize(
                    (int(h * wd / w), wd) if wd / w > hd / h else (hd, int(w * hd / h)),
                    InterpolationMode.BICUBIC,
                ),
                CenterCrop((hd, wd)),
            ]
        )(im)

    return f


STYLE_COMPONENTS_PREPROCESSORS = dict(
    v0_2=Compose(
        [
            Resize(size=336, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ]
    ),
    v0_3=Compose(
        [
            MyResize((518 * 1.3) ** 2, 14),
            ToTensor(),
            Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ]
    ),
)

STYLE_COMPONENTS_CONFIGS = dict(
    v0_2=StyleComponentsConfig(
        style_extractor_path="https://huggingface.co/spaces/gustproof/style-similarity/resolve/main/style-extractor-v0.2.0.ckpt",
        pca_path="https://huggingface.co/gustproof/sdxl-style/resolve/main/pca-v0.2.1.bin",
        adapter_path="https://huggingface.co/gustproof/sdxl-style/resolve/main/ip-adapter-style_sdxl_amix_v0.1.0_fp16.safetensors",
        image_preprocessor="v0_2",
    ),
    v0_3=StyleComponentsConfig(
        style_extractor_path="https://huggingface.co/spaces/gustproof/style-similarity/resolve/main/style-extractor-v0.3.0.ckpt",
        pca_path="https://huggingface.co/gustproof/sdxl-style/resolve/main/pca-v0.3.0.bin",
        adapter_path="https://huggingface.co/gustproof/sdxl-style/resolve/main/ip-adapter-style_sdxl_amix_v0.3.0_fp16.safetensors",
        image_preprocessor="v0_3",
    ),
)
