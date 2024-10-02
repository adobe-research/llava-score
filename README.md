# 🌋 LLaVA-score: Removing Distributional Discrepancies in Captions Improves Image-Text Alignment

Our code is bulit upon original LLaVA repo and `llava_score_evaluator.py` file is the main evaluation file

## Install

The following is from LLaVA

```
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```


## Inferece 

You can download the ckpt from [here](https://scenegan.s3.us-east-2.amazonaws.com/llava-score-ckpt.zip)



```
python llava_score_evaluator.py  --image_file PATH_TO_IMAGE  --caption CAPTION_OF_IMAGE
```


## Usage and License Notices
This project utilizes certain datasets and models that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the Azure Product Terms for Online Services (https://www.microsoft.com/licensing/terms/product/ForOnlineServices/all) for certain parts of the dataset (the negative captions), the COCO Terms of Use (https://cocodataset.org/#termsofuse) for other parts of the dataset (the images/captions from COCO), and the specific licenses for models and other code, including Apache 2.0 for the LLaVA codebase (https://github.com/haotian-liu/LLaVA/blob/main/LICENSE), the applicable terms for llava-v1.5-13b (https://huggingface.co/liuhaotian/llava-v1.5-13b), and the applicable terms for distilbert-base-uncased (https://huggingface.co/distilbert/distilbert-base-uncased).
