# Use --show-controlnet-example to see this extension.

import os
import gradio as gr
import torch

from modules import scripts
from modules.modelloader import load_file_from_url
from lib_ipadapter_style.IPAdapterPlus import IPAdapterApply
from lib_ipadapter_style.config import (
    STYLE_COMPONENTS_CONFIGS,
    STYLE_COMPONENTS_PREPROCESSORS,
)
from modules.paths_internal import models_path
from modules.devices import device
import pickle
import random
import numpy as np
import timm
from safetensors import safe_open
from collections import defaultdict

SC_EXTENSION_VERSION = "0.3.0"

opIPAdapterApply = IPAdapterApply().apply_ipadapter

N_COMPONENTS_MAX = 200
style_models_path = os.path.join(models_path, "StyleAdapter")


def path_from_url_or_local(p: str):
    return (
        load_file_from_url(
            p,
            model_dir=style_models_path,
        )
        if p.startswith("http")
        else p
    )


def load_ip(adapter_path: str):
    so = safe_open(adapter_path, "pt")
    d = defaultdict(dict)
    for k in so.keys():
        a, _, b = k.partition(".")
        d[a][b] = so.get_tensor(k).to(device)
    return d


class ControlNetExampleForge(scripts.Script):
    model = None

    def load_config(self, config_name: str):
        self.ip_state_dict_cached = None
        self.style_extractor_cached = None
        self.pca_cached = None
        self.config = STYLE_COMPONENTS_CONFIGS[config_name]
        self.image_preprocessor = STYLE_COMPONENTS_PREPROCESSORS[
            self.config.image_preprocessor
        ]
        self.n_components = self.pca[0].n_components

    @property
    def pca(self):
        if self.pca_cached is None:
            pca_path = path_from_url_or_local(self.config.pca_path)
            with open(pca_path, "rb") as f:
                self.pca_cached = pickle.load(f)
            print("[STYLE COMPONENTS]: pca loaded")
        return self.pca_cached

    @property
    def ip_state_dict(self):
        if self.ip_state_dict_cached is None:
            p = path_from_url_or_local(self.config.adapter_path)
            self.ip_state_dict_cached = load_ip(p)
            print("[STYLE COMPONENTS]: adapter loaded")
        return self.ip_state_dict_cached

    @property
    def style_extractor(self):
        if self.style_extractor_cached is None:
            p = path_from_url_or_local(self.config.style_extractor_path)
            self.style_extractor_cached = torch.load(p).to(device).eval()
            print("[STYLE COMPONENTS]: extractor loaded")
        return self.style_extractor_cached

    def unload_models(self):
        if self.ip_state_dict_cached is not None:
            self.ip_state_dict_cached = None
            print("[STYLE COMPONENTS]: adapter unloaded")
        if self.style_extractor_cached is not None:
            self.style_extractor_cached = None
            print("[STYLE COMPONENTS]: extractor unloaded")

    def __init__(self) -> None:
        super().__init__()
        self.pca_cached = None
        self.ip_state_dict_cached = None
        self.style_extractor_cached = None

    def title(self):
        return "Style components XL"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        def load_config(config_name):
            self.load_config(config_name)
            return [
                config_name,
                *(
                    gr.update(visible=i < (1 + self.n_components))
                    for i in range(1 + N_COMPONENTS_MAX)
                ),
            ]

        with gr.Accordion(open=False, label=self.title()):
            configs = gr.Dropdown(
                [*STYLE_COMPONENTS_CONFIGS], label="Style components config"
            )
            with gr.Column(visible=False) as main:
                enable = gr.Checkbox(label="Enable")
                with gr.Accordion(open=True, label="Style import/export"):
                    with gr.Tabs():
                        with gr.Tab("Import from image"):
                            gr.Markdown("Extract the style from of 1-10 images")
                            with gr.Row(
                                elem_classes=["cnet-image-row"], equal_height=True
                            ):
                                self.images = [
                                    gr.Image(
                                        source="upload",
                                        type="pil",
                                    )
                                    for _ in range(10)
                                ]
                            with gr.Row():
                                button_import_one = gr.Button(
                                    "Load style components from images",
                                    variant="secondary",
                                )

                                def on_import(*imgs):
                                    if all(img is None for img in imgs):
                                        return [0] * N_COMPONENTS_MAX
                                    from statistics import mean

                                    return [
                                        round(mean(g), 2)
                                        for g in zip(
                                            *(
                                                self.get_component(img)
                                                for img in imgs
                                                if img is not None
                                            )
                                        )
                                    ] + [0] * (N_COMPONENTS_MAX - self.n_components)

                        with gr.Tab("Import/Export from text"):
                            value_area = gr.TextArea(
                                placeholder="[0.3, 0.5, -.7, ...] (Tip: copy and paste from generated image info)"
                            )
                            with gr.Row():
                                button_import_text = gr.Button(
                                    "Import from text", variant="secondary"
                                )
                                button_export_text = gr.Button(
                                    "Export to text", variant="secondary"
                                )
                with gr.Column():
                    with gr.Row():
                        btn_random_style = gr.Button(
                            "Randomize style", variant="secondary"
                        )
                        btn_zero_style = gr.Button("Zero style", variant="secondary")
                    with gr.Row():
                        lucky = gr.Checkbox(
                            False,
                            label="Use random style on each generation",
                        )
                        mult = gr.Slider(
                            -3,
                            3,
                            1,
                            label="Style multiplier",
                            info="Scales each component",
                        )
                        strength = gr.Slider(
                            -3,
                            3,
                            1,
                            label="Style strength",
                            info="Scales the output of the style network",
                        )
                    with gr.Accordion("Style component values", open=True):
                        sliders = self.sliders = [
                            gr.Slider(
                                value=0.0,
                                label=f"Style component {i}",
                                minimum=-10.0,
                                maximum=10.0,
                            )
                            for i in range(N_COMPONENTS_MAX)
                        ]

        configs.change(load_config, configs, [configs, main, *sliders])

        def randomize_style():
            return [round(random.gauss(0, 1), 3) for _ in range(N_COMPONENTS_MAX)]

        def zero_style():
            return [0 for _ in range(N_COMPONENTS_MAX)]

        def import_vals(s):
            vals = [0] * N_COMPONENTS_MAX
            for i, w in enumerate(s.replace(",", " ").strip().split()):
                w = "".join(c for c in w if c in "1234567890.-eE")
                vals[i] = float(w)
            return vals

        def export_vals(*vals):
            return ", ".join(f"{x:.3f}" for x in vals[: self.n_components])

        button_import_text.click(import_vals, inputs=value_area, outputs=sliders)
        button_export_text.click(export_vals, outputs=value_area, inputs=sliders)
        btn_random_style.click(randomize_style, outputs=self.sliders)
        btn_zero_style.click(zero_style, outputs=self.sliders)
        button_import_one.click(on_import, inputs=self.images, outputs=self.sliders)

        return enable, lucky, mult, strength, *sliders

    def get_component(self, img):
        data_config = timm.data.resolve_model_data_config(self.style_extractor)
        tf = timm.data.create_transform(**data_config, is_training=False)

        from torchvision.transforms import (
            CenterCrop,
            Compose,
            InterpolationMode,
            Resize,
        )

        def DResize(area, d):
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
                            (
                                (int(h * wd / w), wd)
                                if wd / w > hd / h
                                else (hd, int(w * hd / h))
                            ),
                            InterpolationMode.BICUBIC,
                        ),
                        CenterCrop((hd, wd)),
                    ]
                )(im)

            return f

        # tf.transforms[0].size = 336
        # tf.transforms[1:2] = []
        tf = Compose([DResize((518 * 1.3) ** 2, 14), *tf.transforms[2:]])
        with torch.no_grad():
            emb = self.style_extractor(tf(img).to(device).unsqueeze(0))
        comp = style_to_randn(self.pca, emb.cpu().numpy())
        return [float(x) for x in comp[0]]

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enable, lucky, mult, strength, *sliders = script_args
        if not enable:
            self.unload_models()
            return
        x0 = np.array(
            [round(x * mult, 2) for x in sliders[: self.n_components]], dtype=float
        )

        if lucky:
            for i in range(len(x0)):
                x0[i] = round(random.gauss(0, 1), 2) * mult

        x = randn_to_style(self.pca, x0).reshape((1, -1))
        embeds = torch.Tensor(x).requires_grad_(False)
        zeros = torch.zeros_like(embeds)
        unet = p.sd_model.forge_objects.unet
        unet = opIPAdapterApply(
            ipadapter=self.ip_state_dict,
            model=unet,
            weight=strength,
            embeds=torch.stack([embeds, zeros]),
        )[0]
        p.sd_model.forge_objects.unet = unet
        p.extra_generation_params.update(
            dict(
                style_components=[*x0],
                style_components_strength=strength,
                style_components_model=self.config.adapter_path,
                style_components_extension_version=SC_EXTENSION_VERSION,
            )
        )


def randn_to_style(pca, x):
    pca, std = pca
    return x * std[: pca.n_components] @ pca.components_ + pca.mean_


def style_to_randn(pca, y):
    pca, std = pca
    return pca.transform(y.reshape((1, -1))) / std[: pca.n_components]
