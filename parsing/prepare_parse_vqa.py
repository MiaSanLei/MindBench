import json
import random
from typing import Any, Dict, List, Tuple, Union
import os
import argparse

from prepare_structured_vqa import generate_random_node, clean_tree, save_jsonl


mmparse_prompts = {
    'en': [
        "Convert this mind map into a clear structured data format.",
        "Analyze the image and describe in detail the levels and relationships of information in the mind map.",
        "Identify and output all the nodes and their interconnections in this mind map image.",
        "Please list in detail the topics, subtopics, and their connections in the mind map.",
        "Organize the content of this mind map into structured textual information.",
        "Present the information from the mind map in the image in the form of tree-structured data.",
        "Parse this mind map and present its content in a structured way.",
        "Extract the key points from this mind map and construct their hierarchical relationship.",
        "Convert the mind map information in the image into a readable structured output.",
        "Identify the elements of the mind map in this image and arrange them into structured data according to their intrinsic logical relationships."
    ],
    'cn': [
        "将这张思维导图转化为清晰的结构化数据格式。",
        "分析图像并详细描述思维导图中的信息层级和关系。",
        "识别并输出此思维导图图像中所有节点及其相互连接的结构。",
        "请详细列出思维导图中的主题、子主题和它们之间的联系。",
        "把这幅思维导图的内容整理成结构化的文本信息。",
        "将图中的思维导图信息以树状结构数据的形式展现出来。",
        "解析这张思维导图，并将其内容以结构化的方式呈现。",
        "提炼出这幅思维导图中的关键点，并构建它们之间的层次关系。",
        "将图像中的思维导图信息转换为可读的结构化输出。",
        "识别这张图中的思维导图元素，并按照其内在逻辑关系排列成结构化数据。"
    ]
}

mmpart_parse_prompts = {
    'en': [
        "Parse a mind map subgraph with the theme '[content]' and present its content in a structured way.",
        "Extract the part of the mind map centered around '[content]' and present its information in a structured data format.",
        "Convert the mind map subgraph with '[content]' as the central node into clear structured data.",
        "List in detail the topics, subtopics, and their connections in the mind map subgraph centered around '[content]'.",
        "Organize the content of the mind map subgraph centered on '[content]' into structured textual information.",
        "Analyze the mind map subgraph with '[content]' as the root node and present its tree-structured data.",
        "Identify all nodes and their connections in the mind map subgraph with the theme '[content]'.",
        "Extract the key points from the mind map subgraph centered on '[content]' and construct their hierarchical relationship.",
        "Convert the information from the mind map subgraph with the theme '[content]' into a readable structured output.",
        "Identify the elements of the mind map subgraph with '[content]' as the core and arrange them into structured data according to their intrinsic logical relationships."
    ],
    'cn': [
        "解析以“[content]”为主题的思维导图子图，并将其内容以结构化的方式呈现。",
        "提取以“[content]”为中心的思维导图部分，并以结构化数据格式展示其信息。",
        "将以“[content]”为核心节点的思维导图子图转化为清晰的结构化数据。",
        "详细列出以“[content]”为中心的思维导图子图的主题、子主题及其相互关系。",
        "把以“[content]”为中心的思维导图内容整理成结构化的文本信息。",
        "分析以“[content]”为根节点的思维导图子图，呈现其树状结构数据。",
        "识别以“[content]”为主题的思维导图子图中的所有节点及其连接关系。",
        "提炼出以“[content]”为中心的思维导图子图的关键点，并构建层次关系。",
        "将以“[content]”为主题的思维导图子图信息转换为可读的结构化输出。",
        "识别以“[content]”为核心的思维导图子图元素，并按其逻辑关系排列成结构化数据。"
    ]
}


def json2token(obj: Any, update_special_tokens_for_json_key: bool = False, sort_json_key: bool = False):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                output += (
                    fr"<s_{k}>"
                    + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        obj = str(obj)
        return obj


def find_subtree(tree, target_text):
    """Parse a subgraph with the "target_text" theme"""
    if isinstance(tree, dict):
        for key, value in tree.items():
            if key == 'text' and value == target_text:
                return tree
            elif isinstance(value, list) or isinstance(value, dict):
                found = find_subtree(value, target_text)
                if found:
                    return found
    elif isinstance(tree, list):
        for item in tree:
            found = find_subtree(item, target_text)
            if found:
                return found
    return None


def update_subtree_node_key(data, depth=0):
    """Update hierarchical sequence numbers of the subtree"""
    if isinstance(data, dict):
        keys_to_update = [key for key in data if key.startswith('node-')]
        for key in keys_to_update:
            new_key = f'node-{depth}'
            data[new_key] = data.pop(key)
            update_subtree_node_key(data[new_key], depth+1)
        for key in data:
            if isinstance(data[key], list):
                for item in data[key]:
                    update_subtree_node_key(item, depth+1)
    elif isinstance(data, list):
        for item in data:
            update_subtree_node_key(item, depth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse_type", type=str, choices=['part', 'full'], help="Specify the parse type (part or full)")
    args = parser.parse_args()

    ori_anno_json = "./annotations/synth_test.json"

    with open(ori_anno_json, "r", encoding='utf-8') as f:
        json_info = json.load(f)
    annotations = json_info["annotations"]
    print(len(annotations))

    parse_type = args.parse_type
    new_annotations = []
    for anno in annotations:
        gt_parse = anno["ground_truth"]["gt_parse"]
        gt_parse = clean_tree(gt_parse)

        if parse_type == 'part':
            node = generate_random_node(gt_parse)
            gt_parse = find_subtree(gt_parse, node)

            tree = {"node-0": gt_parse}
            update_subtree_node_key(tree)
            gt_parse = {"map": tree}

        gt_token_sequence = json2token(gt_parse)
        new_anno = {}
        new_anno["image"] = [anno["image"]]
        new_anno["prompt"] = ""
        new_anno["text"] = ""
        new_anno["system_instruction"] = ""
        new_anno["conversations"] = []
        new_anno["conversations"].append({'from': 'user', 'value': '<image>'})

        lang = 'en' if 'synth_v2/en_' in anno["image"] else 'cn'
        if parse_type == 'full':
            prompt = random.choice(mmparse_prompts[lang])
        else:
            prompt = random.choice(mmpart_parse_prompts[lang])
            prompt = prompt.replace("[content]", node)
        new_anno["conversations"].append({'from': 'user', 'value': prompt})
        new_anno["conversations"].append({'from': 'assistant', 'value': gt_token_sequence})
        new_anno["task_type"] = "qa_sft"
        new_annotations.append(new_anno)
    print(len(new_annotations))
    save_jsonl(new_annotations, os.path.join(os.path.dirname(ori_anno_json), f"synth_test_{parse_type}_parse.jsonl"))
