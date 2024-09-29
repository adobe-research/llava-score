"""
ADOBE CONFIDENTIAL
Copyright 2024 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
"""
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from PIL import Image


class LLaVAEvaluator:
    def __init__(self, model_path):

        self.model_path = model_path.rstrip('/')
        self.dtype = torch.float16

        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(self.model_path, None, model_name)
        self.conv_mode = "llava_v1"

    @torch.no_grad()
    def __call__(self, image, caption):
 

        user_question = "Does this image match the following caption: "+ caption + "?\nAnswer Yes/No directly."

        
        if self.model.config.mm_use_im_start_end:
            user_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_question
        else:
            user_prompt = DEFAULT_IMAGE_TOKEN + '\n' + user_question

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(image).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config).cuda()
        image_tensor = image_tensor.to(self.dtype)


        setattr(self.model, 'tokenizer', self.tokenizer) 

        output = self.model.forward(input_ids, images=image_tensor)

        last_word_logits = output['logits'][0,-1]


        yes_idx = 3869
        no_idx = 1939

        yes_exp_logit = torch.exp( last_word_logits[yes_idx].float() )
        no_exp_logit = torch.exp( last_word_logits[no_idx].float() )

        score = yes_exp_logit / (yes_exp_logit + no_exp_logit)

        return score.item()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--llava_model_path", type=str, default='../../LLaVA/checkpoints/fullft_llava-v1.5-13b-coco_original_merge_all2+seetrue_merge_all2_repeat1/checkpoint-1658')
    parser.add_argument("--image_file", type=str, required=True)
    parser.add_argument("--caption", type=str, required=True)
    args = parser.parse_args()


    evaluator = LLaVAEvaluator(model_path=args.llava_model_path)
    with torch.no_grad():
        score = evaluator(args.image_file, args.caption)


    print(score)