# Style Components for Stable Diffusion WebUI Forge

Style control for SDXL anime models. A Forge extension.

**Links**: [Forge extension](https://github.com/gustproof/sd-webui-forge-style-components) | [ComfyUI nodes](https://github.com/gustproof/ComfyUI_IPAdapter_plus_Style_Components) | [Civitai](https://civitai.com/models/339104)

![](assets/forge-screenshot.jpg)

## What is Style Components

Style Components is an IP-Adapter model conditioned on anime styles. The style embeddings can either be extracted from images or created manually. This repo currently only supports the SDXL model trained on AutismmixPony.

## Why?

Currently, the main means of style control is through artist tags. This method reasonably raises the concern of style plagiarism. By breaking down styles into interpretable components that are present in all artists, direct copying of styles can be avoided. Furthermore, new styles can be easily created by manipulating the magnitude of the style components, offering more controllability over stacking artist tags or LoRAs.

Additionally, this can be potentially useful for general purpose training, as training with style condition may weaken style leakage into concepts. This also serves as a demonstration that image models can be conditioned on arbitrary tensors other than text or images. Hopefully, more people can understand that it is not necessary to force conditions that are inherently numerical (aesthetic scores, dates, ...) into text form tags

## Usage

Clone this repo under `extensions/`, or use Forge's `Extensions/Install from URL` with the URL `https://github.com/gustproof/sd-webui-forge-style-components`. Restart and make sure to check the `Enable` checkbox.

The model is trained on AutismmixPony. Functionality on other Pony derivatives is purely coincidental and not guaranteed. The adapter is not trained with Pony tags (source_anime, score_9, ...), so these tags can be omitted when prompting.


## Technical details

A style embedding model is created by Supervised Contrastive Learning on an artists dataset. Then, a modified IP-Adapter is trained on using the same dataset with WD1.4 tags for 45k steps of batch size 1.

Due to how the model was trained, the style embeddings capture more of the local style rather than global composition. Also, no efforts were made to ensure the faces were included in the crops in training, so style embeddings may not capture well the face or eye style.