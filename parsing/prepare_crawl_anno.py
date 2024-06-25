# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import json
import os
import random
import re

from prepare_parse_vqa import mmparse_prompts, json2token
from prepare_structured_vqa import clean_tree, save_jsonl


def filter_special_symbols(text):
    # regex = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\uff01\u2018\u2019\u2026\u002e\u002c\u003a\u0022\u0028\u0029\u003f\u002b\u002d\u002a\u002f\u003d\u005e\u003c\u003e\u0021\u0040\u0023\u0024\u007e\u00b7\u0020\u0021\u003f\u0028\u0029\u0040\u0023\u0024\u007e\u2022\u00b7\u0020\r\n\u300c\u300d\u3010\u3011\u007b\u007d\u005b\u005d\uff5c\u007c\uffe5\u2014\u005f\u0025\u0026\uFF1B\u003B\u0027]')
    # filtered_text = regex.sub('', text)
    filtered_text = re.sub(' +', ' ', text)
    return filtered_text

def parse_html(filename, add_relation=True):
    """
    Convert the mind map into a nested JSON format
    """
    with open(filename, 'r') as f:
        contents = f.read()

    soup = BeautifulSoup(contents, 'lxml')
    root_text = soup.h1.a.string if soup.h1.a.string is not None else ''

    root = {'text': root_text, 'node': []}
    if add_relation:
        root['relation'] = []
    stack = [root]
    for tag in soup.h1.find_next_siblings(['h2', 'h3', 'p']):
        if tag.name == 'p':
            # The same handling way for summary and relationship
            if add_relation:
                # Filter duplicate relation based on href
                a_tags = [a for a in tag.find_all('a') if a.string is not None]
                unique_a_tags = set()
                for a in a_tags:
                    unique_a_tags.add((a['href'], a.string))
                relations = [filter_special_symbols(a_string) for _, a_string in unique_a_tags]

                relations = [relation for relation in relations if relation != "" and relation != " "]
                stack[-1]['relation'].extend(relations)
        else:
            level = 2 if tag.name == 'h2' else 2 + tag.a.string.count('\xa0')
            node_text = tag.a.string.replace('\xa0', '') if tag.a.string is not None else ''
            node_text = filter_special_symbols(node_text)
            node_text = '' if node_text.isspace() else node_text
            node = {'text': node_text, 'node': []}
            if add_relation:
                node['relation'] = []
            while len(stack) >= level:
                stack.pop()
            stack[-1]['node'].append(node)
            stack.append(node)
    return root

def remove_empty_nodes(data):
    """
    Remove empty nodes and empty relations
    """
    if isinstance(data, dict):
        if 'node' in data and not data['node']:
            del data['node']
        if 'relation' in data and not data['relation']:
            del data['relation']
        for key in data:
            if isinstance(data[key], (dict, list)):
                remove_empty_nodes(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_empty_nodes(item)

def remove_empty_subtrees(tree):
    """
    Remove empty subtree with empty text
    """
    if isinstance(tree, dict):
        if 'node' in tree:
            tree['node'] = [remove_empty_subtrees(subtree) for subtree in tree['node']]
            tree['node'] = [subtree for subtree in tree['node'] if subtree is not None]
        if ('node' not in tree or not tree['node']) and not tree['text']:
            return None
    return tree

def write_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def update_node_key(data, depth=0):
    """Add hierarchical sequence numbers to nested nodes"""
    if isinstance(data, dict):
        if 'node' in data:
            data[f'node-{depth}'] = data.pop('node')
            update_node_key(data[f'node-{depth}'], depth+1)
        for key in data:
            if isinstance(data[key], list):
                for item in data[key]:
                    update_node_key(item, depth)
    elif isinstance(data, list):
        for item in data:
            update_node_key(item, depth)

def count_nodes(tree):
    """Count the number of nodes in a tree"""
    count = 0
    for key, value in tree.items():
        if "text" in key:
            count += 1
        if isinstance(value, dict):
            count += count_nodes(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    count += count_nodes(item)
    return count


def parse2json(input_dir):
    for item in os.listdir(input_dir):
        anno_p = os.path.join(input_dir, item)
        if not (os.path.isdir(anno_p) and anno_p.endswith('_anno')):
            continue
        html_files = [os.path.join(anno_p, file) for file in os.listdir(anno_p) if file.endswith('.html')]
        print(len(html_files))
        for hfile in html_files:
            data = parse_html(hfile, add_relation=True)
            remove_empty_nodes(data)
            remove_empty_subtrees(data)
            write_json(data, hfile[:-5]+".json")


def prepare_anno(input_dir):
    for item in os.listdir(input_dir):
        anno_p = os.path.join(input_dir, item)
        if not (os.path.isdir(anno_p) and anno_p.endswith('_anno')):
            continue
        annotations = []
        for jfile in os.listdir(anno_p):
            if not jfile.endswith('.json'):
                continue
            with open(os.path.join(anno_p, jfile), 'r') as f:
                json_info = json.load(f)
            json_info = {'node': json_info}
            update_node_key(json_info)
            res_dict = dict()
            res_dict['ground_truth'] = {'gt_parse': {'map': json_info}}
            res_dict['image'] = os.path.join(os.path.basename(anno_p).replace("_anno", "_png"), jfile.replace('.json', '.png'))
            annotations.append(res_dict)
        print(len(annotations))
        write_json({"annotations": annotations}, anno_p+".json")


def split_data(input_dir):
    json_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('_anno.json')]
    for jfile in json_files:
        with open(jfile, 'r') as f:
            json_info = json.load(f)
        annotations = json_info["annotations"]
        print(len(annotations))
        random.shuffle(annotations)
        train_ratio = 0.95 if ('xmind_en' in jfile or 'zhixi' in jfile) else 0.9
        train_size = int(len(annotations) * train_ratio)
        train_annos = annotations[:train_size]
        test_annos = annotations[train_size:]
        print(len(train_annos))
        print(len(test_annos))
        write_json({"annotations": train_annos}, jfile.replace(".json", "_train.json"))
        write_json({"annotations": test_annos}, jfile.replace(".json", "_test.json"))


def filter_difficult_mindmap(input_dir):
    json_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('_test.json')]
    for jfile in json_files:
        with open(jfile, 'r') as f:
            json_info = json.load(f)
        annotations = json_info["annotations"]
        print(len(annotations))
        easy_annotations = []
        for anno in annotations:
            count = count_nodes(anno['ground_truth'])
            if count <= 60:
                easy_annotations.append(anno)
        easy_counts = len(easy_annotations)
        diff_counts = len(annotations) - easy_counts
        print(f"easy: {easy_counts} diff: {diff_counts}")
        write_json({"annotations": easy_annotations}, jfile.replace(".json", "_easy.json"))


def prepare_ureader_train_anno(input_dir):
    json_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('_train.json')]
    new_annotations = []
    for jfile in json_files:
        with open(jfile, 'r') as f:
            json_info = json.load(f)
        annotations = json_info["annotations"]
        print(len(annotations))

        for anno in annotations:
            gt_parse = anno["ground_truth"]["gt_parse"]
            gt_token_sequence = json2token(clean_tree(gt_parse))

            new_anno = {}
            new_anno["image"] = [anno["image"]]
            new_anno["prompt"] = ""
            new_anno["text"] = ""
            new_anno["system_instruction"] = ""
            new_anno["conversations"] = []
            new_anno["conversations"].append({'from': 'user', 'value': '<image>'})
            if 'xmind_en' in anno["image"] or 'bxmind' in anno["image"] or 'bmmanger' in anno["image"]:
                lang = "en"
            else:
                lang = "cn"
            new_anno["conversations"].append({'from': 'user', 'value': random.choice(mmparse_prompts[lang])})
            new_anno["conversations"].append({'from': 'assistant', 'value': gt_token_sequence})
            new_anno["task_type"] = "qa_sft"
            new_annotations.append(new_anno)
    save_jsonl(new_annotations, os.path.join(input_dir, 'train.jsonl'))


def prepare_ureader_val_anno(input_dir):
    json_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('_test_easy.json')]
    for jfile in json_files:
        new_annotations = []
        with open(jfile, 'r') as f:
            json_info = json.load(f)
        annotations = json_info["annotations"]
        print(len(annotations))

        for anno in annotations:
            gt_parse = anno["ground_truth"]["gt_parse"]
            gt_token_sequence = json2token(clean_tree(gt_parse))

            new_anno = {}
            new_anno["image"] = [anno["image"]]
            new_anno["prompt"] = ""
            new_anno["text"] = ""
            new_anno["system_instruction"] = ""
            new_anno["conversations"] = []
            new_anno["conversations"].append({'from': 'user', 'value': '<image>'})
            if 'xmind_en' in anno["image"] or 'bxmind' in anno["image"] or 'bmmanger' in anno["image"]:
                lang = "en"
            else:
                lang = "cn"
            new_anno["conversations"].append({'from': 'user', 'value': random.choice(mmparse_prompts[lang])})
            new_anno["conversations"].append({'from': 'assistant', 'value': gt_token_sequence})
            new_anno["task_type"] = "qa_sft"
            new_annotations.append(new_anno)
        save_jsonl(new_annotations, os.path.join(input_dir, os.path.basename(jfile).replace('_anno', '').replace('.json', '.jsonl')))


if __name__ == "__main__":
    input_dir = './crawl_annotations/'

    # 1 Parsing html files into json format
    parse2json(input_dir)
    
    # 2 Prepare training and test label files
    prepare_anno(input_dir)    
    split_data(input_dir)
    filter_difficult_mindmap(input_dir)

    # 3 Prepare token sequence label files for training
    prepare_ureader_train_anno(input_dir)
    prepare_ureader_val_anno(input_dir)