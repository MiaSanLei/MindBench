import torch
import os
import json
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


def model_gen( model, text, images, hd_num=25, need_bos=True):
    text = text.split('Please answer')[0].strip()
    text = f'[UNUSED_TOKEN_146]user\n{text}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
    pt1 = 0
    embeds = []
    im_mask = []
    images = [images]
    images_loc = [0]
    for i, pts in enumerate(images_loc + [len(text)]):
        subtext = text[pt1:pts]
        if need_bos or len(subtext) > 0:
            text_embeds = model.encode_text(subtext, add_special_tokens=need_bos)
            im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
            embeds.append(text_embeds)
            need_bos = False
        if i < len(images):
            # image = Image.open(images[i]).convert('RGB')
            # image = __resize_img__(image)
            # image = model.vis_processor(image).unsqueeze(0).cuda()
            image_embeds = model.encode_img(images[i], hd_num)
            # print(f"utils, image_embeds: {image_embeds.shape}")
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
            embeds.append(image_embeds)
        pt1 = pts
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    outputs = model.generate(inputs_embeds=embeds, im_mask=im_mask,
                        temperature=1.0, max_new_tokens=1536, num_beams=2,
                        do_sample=False, repetition_penalty=1.0)
    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0]
    return output_text


def inference(model_path, input_file, output_dir, root_dir="./data"):
    os.makedirs(output_dir, exist_ok=True)

    save_path = Path(output_dir) / f"{Path(input_file).stem}_results.jsonl"
    if os.path.exists(save_path):
        print(f"inference on {Path(input_file).name} is finished...")
        return

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    model.tokenizer = tokenizer

    # Load input data
    with open(input_file,'r',encoding="utf-8")as f:
        lines = [json.loads(line) for line in f]

    evaluate_output=[]
    for line in tqdm(lines):
        question = line['conversations'][1]['value']
        answer = line['conversations'][2]['value']
        img_path = os.path.join(root_dir, line['image'][0])
        assert os.path.exists(img_path), img_path

        with torch.cuda.amp.autocast():
            response = model_gen(model, question, img_path)

        print(question)
        print('gt:', answer)
        print('model:', response)
        print('-'*10)

        line['model_answer'] = response
        evaluate_output.append(line)

    with open(save_path, 'w', encoding='utf-8')as f:
        for line in evaluate_output:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with multimodal model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--root_dir", type=str, default="./data", help="Root directory containing data files")
    
    args = parser.parse_args()
    
    inference(args.model_path, args.input_file, args.output_dir, args.root_dir)