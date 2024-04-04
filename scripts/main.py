# Use --show-controlnet-example to see this extension.

import os
import gradio as gr
import torch

from modules import scripts
from modules.modelloader import load_file_from_url
from lib_ipadapter_style.IPAdapterPlus import IPAdapterApply
from modules.paths_internal import models_path
from modules.devices import device
import pickle
import random
import numpy as np
import timm
from safetensors import safe_open
from collections import defaultdict

SC_EXTENSION_VERSION = '0.2.1'

opIPAdapterApply = IPAdapterApply().apply_ipadapter

N_COMPONENTS = 100
style_models_path = os.path.join(models_path, "StyleAdapter")
EXTRACTOR_URL = "https://huggingface.co/spaces/gustproof/style-similarity/resolve/main/style-extractor-v0.2.0.ckpt"
ADAPTER_URL = "https://huggingface.co/gustproof/sdxl-style/resolve/main/ip-adapter-style_sdxl_amix_v0.1.0_fp16.safetensors"
PCA_URL = "https://huggingface.co/gustproof/sdxl-style/resolve/main/pca-v0.2.1.bin"


def load_ip():
    adapter_path = load_file_from_url(
        ADAPTER_URL,
        model_dir=style_models_path,
    )
    so = safe_open(adapter_path, "pt")
    d = defaultdict(dict)
    for k in so.keys():
        a, _, b = k.partition(".")
        d[a][b] = so.get_tensor(k).to(device)
    return d


class ControlNetExampleForge(scripts.Script):
    model = None

    @property
    def ip_state_dict(self):
        if self.ip_state_dict_cached is None:
            self.ip_state_dict_cached = load_ip()
            print("[STYLE COMPONENTS]: adapter loaded")
        return self.ip_state_dict_cached

    @property
    def style_extractor(self):
        if self.style_extractor_cached is None:
            extractor_path = load_file_from_url(
                EXTRACTOR_URL,
                model_dir=style_models_path,
            )
            self.style_extractor_cached = torch.load(extractor_path).to(device).eval()
            print("[STYLE COMPONENTS]: extractor loaded")
        return self.style_extractor_cached

    def unload_models(self):
        if self.ip_state_dict_cached is not None:
            del self.ip_state_dict_cached
            self.ip_state_dict_cached = None
            print("[STYLE COMPONENTS]: adapter unloaded")
        if self.style_extractor_cached is not None:
            del self.style_extractor_cached
            self.style_extractor_cached = None
            print("[STYLE COMPONENTS]: extractor unloaded")

    def __init__(self) -> None:
        super().__init__()
        pca_path = load_file_from_url(
            PCA_URL,
            model_dir=style_models_path,
        )
        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)
        print("[STYLE COMPONENTS]: pca loaded")
        self.ip_state_dict_cached = None
        self.style_extractor_cached = None

    def title(self):
        return "Style components XL"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        self.dummy_gradio_update_trigger = gr.Number(value=0, visible=False)
        with gr.Accordion(open=False, label=self.title()):
            enable = gr.Checkbox(label="Enable")
            with gr.Accordion(open=True, label="Style import/export"):
                with gr.Tabs():
                    with gr.Tab("Import from image"):
                        gr.Markdown("Extract the style from of 1-10 images")
                        with gr.Row(elem_classes=["cnet-image-row"], equal_height=True):
                            self.images = [
                                gr.Image(
                                    source="upload",
                                    type="pil",
                                )
                                for _ in range(10)
                            ]
                        with gr.Row():
                            button_import_one = gr.Button(
                                "Load style components from images", variant="secondary"
                            )

                            def on_import(*imgs):
                                if all(img is None for img in imgs):
                                    return [0] * N_COMPONENTS
                                from statistics import mean

                                return [
                                    round(mean(g), 3)
                                    for g in zip(
                                        *(
                                            self.get_component(img)
                                            for img in imgs
                                            if img is not None
                                        )
                                    )
                                ]

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
                    btn_random_style = gr.Button("Randomize style", variant="secondary")
                    btn_zero_style = gr.Button("Zero style", variant="secondary")
                with gr.Row():
                    lucky = gr.Checkbox(
                        False,
                        label="Use random style on each generation",
                    )
                    mult = gr.Slider(
                        -3, 3, 1, label="Style multiplier", info="Scales each component"
                    )
                    strength = gr.Slider(
                        -3,
                        3,
                        1,
                        label="Style strength",
                        info="Scales the output of the style network",
                    )
                sliders = self.sliders = [
                    gr.Slider(
                        value=0.0,
                        label=f"Style component {i}",
                        minimum=-10.0,
                        maximum=10.0,
                    )
                    for i in range(N_COMPONENTS)
                ]

        def randomize_style():
            return [round(random.gauss(0, 1), 3) for _ in range(N_COMPONENTS)]

        def zero_style():
            return [0 for _ in range(N_COMPONENTS)]

        def import_vals(s):
            vals = [0] * N_COMPONENTS
            for i, w in enumerate(s.replace(",", " ").strip().split()):
                w = "".join(c for c in w if c in "1234567890.-eE")
                vals[i] = float(w)
            return vals

        def export_vals(*vals):
            return ", ".join(f"{x:.3f}" for x in vals)

        button_import_text.click(import_vals, inputs=value_area, outputs=sliders)
        button_export_text.click(export_vals, outputs=value_area, inputs=sliders)
        btn_random_style.click(randomize_style, outputs=self.sliders)
        btn_zero_style.click(zero_style, outputs=self.sliders)
        button_import_one.click(on_import, inputs=self.images, outputs=self.sliders)

        return enable, lucky, mult, strength, *sliders

    def get_component(self, img):

        data_config = timm.data.resolve_model_data_config(self.style_extractor)
        tf = timm.data.create_transform(**data_config, is_training=False)
        tf.transforms[0].size = 336
        tf.transforms[1:2] = []
        with torch.no_grad():
            emb = self.style_extractor(tf(img).to(device).unsqueeze(0))
        comp = style_to_randn(self.pca, emb.cpu().numpy())
        return [float(x) for x in comp[0]]

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enable, lucky, mult, strength, *sliders = script_args
        if not enable:
            self.unload_models()
            return
        x0 = np.array([round(x * mult, 2) for x in sliders], dtype=float)

        if lucky:
            for i in range(len(sliders)):
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
                style_components_model=ADAPTER_URL,
                style_components_extension_version=SC_EXTENSION_VERSION,
            )
        )


def randn_to_style(pca, x):
    pca, std = pca
    return x * std[:N_COMPONENTS] @ pca.components_ + pca.mean_


def style_to_randn(pca, y):
    pca, std = pca
    return pca.transform(y.reshape((1, -1))) / std[:N_COMPONENTS]
