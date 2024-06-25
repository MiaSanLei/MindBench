import json
import os
import random
import argparse

from prepare_structured_vqa import generate_random_node, find_node_level, find_parent_node, find_children_nodes, is_parent_child, are_siblings, clean_text, clean_tree, save_jsonl
from prepare_parse_vqa import find_subtree, update_subtree_node_key, json2token


mmpos_prompts = {
    'en': [
        # recognition
        "Identify and return the text of the node within the bounding box <bbox>[content]</bbox>.",
        "Determine and inform which level the node within the bounding box <bbox>[content]</bbox> is located on.",
        "Identify and return the parent node of the node within the bounding box <bbox>[content]</bbox>.",
        "Identify and return all child nodes of the node within the bounding box <bbox>[content]</bbox>.",
        "Determine and inform whether the nodes within the bounding boxes <bbox>[content1]</bbox> and <bbox>[content2]</bbox> have a parent-child relationship.",
        "Determine and inform whether the nodes within the bounding boxes <bbox>[content1]</bbox> and <bbox>[content2]</bbox> have a sibling relationship.",
        "Determine and inform how many nodes are contained within the bounding box <bbox>[content]</bbox>.",
        # grounding
        "Identify and return the bounding box of the node labeled '[content]' in the mind map.",
        "Identify and return the parent node and its bounding box of the node labeled '[content]' in the mind map.",
        "Identify and return all child nodes and their bounding boxes of the node labeled '[content]' in the mind map.",
        "Identify and return the bounding box of the subgraph in the mind map with the theme '[content]'."
    ],
    'cn': [
        # recognition
        "请识别并返回边界框<bbox>[content]</bbox>内的节点的文本。",
        "请确定并告知边界框<bbox>[content]</bbox>内的节点位于哪一层级。",
        '请识别并返回边界框<bbox>[content]</bbox>内的节点的父节点。',
        '请识别并返回边界框<bbox>[content]</bbox>内的节点的所有子节点。',
        '请判断并告知边界框<bbox>[content1]</bbox>内的节点和边界框<bbox>[content2]</bbox>内的节点是否构成父子关系。',
        '请判断并告知边界框<bbox>[content1]</bbox>内的节点和边界框<bbox>[content2]</bbox>内的节点是否构成兄弟关系。',
        '请确定并告知边界框<bbox>[content]</bbox>内共有几个节点。',
        # grounding
        "请识别并返回在思维导图中标记为“[content]”的节点的边界框。",
        "请识别并返回在思维导图中标记为“[content]”的节点的父节点及其边界框。",
        "请识别并返回在思维导图中标记为“[content]”的节点的所有子节点及其边界框。",
        "请识别并返回以“[content]”为主题的思维导图子图的边界框。"
    ]
}

mmpos_parse_prompts = {
    'en': [
        "Parse the mind map subgraph within the bounding box <bbox>[content]</bbox> and present its content in a structured way.",
        "Extract the section of the mind map within the bounding box <bbox>[content]</bbox> and present its information in a structured data format.",
        "Convert the mind map subgraph inside the bounding box <bbox>[content]</bbox> into clear structured data.",
        "List in detail the topics, subtopics, and their connections in the mind map subgraph within the bounding box <bbox>[content]</bbox>.",
        "Organize the content of the mind map subgraph within the bounding box <bbox>[content]</bbox> into structured textual information.",
        "Analyze the mind map subgraph within the bounding box <bbox>[content]</bbox> and present its tree-structured data.",
        "Identify all nodes and their connections in the mind map subgraph within the bounding box <bbox>[content]</bbox>.",
        "Extract the key points from the mind map subgraph inside the bounding box <bbox>[content]</bbox> and construct their hierarchical relationship.",
        "Convert the information from the mind map subgraph within the bounding box <bbox>[content]</bbox> into a readable structured output.",
        "Identify the elements of the mind map subgraph within the bounding box <bbox>[content]</bbox> and arrange them into structured data according to their intrinsic logical relationships."
    ],
    'cn': [
        "解析边界框<bbox>[content]</bbox>内的思维导图子图，并将其内容以结构化的方式呈现。",
        "提取边界框<bbox>[content]</bbox>内的思维导图部分，并以结构化数据格式展示其信息。",
        "将边界框<bbox>[content]</bbox>内的思维导图子图转化为清晰的结构化数据。",
        "详细列出边界框<bbox>[content]</bbox>内的思维导图子图的主题、子主题及其相互关系。",
        "把边界框<bbox>[content]</bbox>内的思维导图内容整理成结构化的文本信息。",
        "分析边界框<bbox>[content]</bbox>内的思维导图子图，呈现其树状结构数据。",
        "识别边界框<bbox>[content]</bbox>内的思维导图子图中的所有节点及其连接关系。",
        "提炼出边界框<bbox>[content]</bbox>内的思维导图子图的关键点，并构建层次关系。",
        "将边界框<bbox>[content]</bbox>内的思维导图子图信息转换为可读的结构化输出。",
        "识别边界框<bbox>[content]</bbox>内的思维导图子图元素，并按其逻辑关系排列成结构化数据。"
    ]
}


def collect_subtree_nodes(node, target_text, found=False):
    """
    Return a list of all nodes with the theme of target_text
    """
    nodes_list = []

    if node['text'] == target_text:
        found = True
    if found:
        nodes_list.append(node['text'])
    for key in node:
        if key.startswith('node-'):
            for child in node[key]:
                nodes_list.extend(collect_subtree_nodes(child, target_text, found))
    
    return nodes_list

def cal_bbox_union(bounding_boxes):
    union_x1 = float('inf')
    union_y1 = float('inf')
    union_x2 = float('-inf')
    union_y2 = float('-inf')

    for box in bounding_boxes:
        union_x1 = min(union_x1, box[0])
        union_y1 = min(union_y1, box[1])
        union_x2 = max(union_x2, box[2])
        union_y2 = max(union_y2, box[3])

    return [union_x1, union_y1, union_x2, union_y2]

def is_overlap(boxA, boxB, threshold=0.5):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])

    ratio = interArea / float(boxAArea)

    return ratio >= threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_type", type=str, choices=['parse', 'vqa'], help="Specify the pos task type (parse or vqa)")
    parser.add_argument('--input_folder', type=str, default='../synthesis', help='Path to dataset input folder')
    args = parser.parse_args()

    ori_anno_json = "./annotations/synth_test.json"

    with open(ori_anno_json, "r", encoding='utf-8') as f:
        json_info = json.load(f)
    annotations = json_info["annotations"]
    print(len(annotations))

    pos_type = args.pos_type
    new_annotations = []
    for anno in annotations:
        gt_parse = anno["ground_truth"]["gt_parse"]
        lang = 'en' if 'synth_v2/en_' in anno["image"] else 'cn'

        # Position information of all nodes in the current sample
        path_seg = "/".join(os.path.dirname(anno["image"]).split("/")[:2])
        file_name = os.path.splitext(os.path.basename(anno["image"]))[0]
        pos_file = os.path.join(args.input_folder, path_seg, 'graph', file_name+'.json')
        with open(pos_file, "r", encoding='utf-8') as f:
            pos_info = json.load(f)

        for _ in range(1):  # You can ask multiple questions about a sample
            gt_parse = clean_tree(gt_parse)

            if pos_type == 'parse':
                # 1 Randomly select a node
                node = generate_random_node(gt_parse)
                # 2 Recursively traverse the subtree and return a list of nodes
                subtree_nodes = collect_subtree_nodes(gt_parse['map']['node-0'], node)
                # 3 Find the bounding boxes of all child nodes
                bboxes = []
                for item in pos_info:
                    label = item["label"]
                    if clean_text(label) in subtree_nodes:
                        x1, y1, x2, y2 = item["xyxy"].split(",")
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        bboxes.append(bbox)
                # 4 Take the union to obtain the bounding box, forming the prompt
                union_bbox = cal_bbox_union(bboxes)
                union_bbox_str = f"{union_bbox[0]},{union_bbox[1]},{union_bbox[2]},{union_bbox[3]}"
                prompt = random.choice(mmpos_parse_prompts[lang])
                prompt = prompt.replace("[content]", union_bbox_str)

                # 5 The answer is the subtree structure with the selected node as the central theme
                gt_parse = find_subtree(gt_parse, node)

                tree = {"node-0": gt_parse}
                update_subtree_node_key(tree)
                gt_parse = {"map": tree}

                gt_token_sequence = json2token(gt_parse)
            else:
                qa_choice = random.randint(1, 11)
                prompt = mmpos_prompts[lang][qa_choice-1]                    
                # 1 Randomly select a node
                node = generate_random_node(gt_parse)
                # 2 Obtain the corresponding bounding box
                if qa_choice in [1, 2, 3, 4, 5, 6, 7, 8]:
                    bbox_str = ""
                    for item in pos_info:
                        label = item["label"]
                        if clean_text(label) == node:
                            bbox_str = item["xyxy"]
                            break
                # (Optional) randomly select another node
                if qa_choice in [5, 6, 7]:
                    node2 = generate_random_node(gt_parse)
                    bbox_str2 = ""
                    for item in pos_info:
                        label = item["label"]
                        if clean_text(label) == node2:
                            bbox_str2 = item["xyxy"]
                            break
                # 3 Generate prompts, except for Q7
                if qa_choice in [1, 2, 3, 4]:
                    prompt = prompt.replace("[content]", bbox_str)
                elif qa_choice in [5, 6]:
                    prompt = prompt.replace("[content1]", bbox_str).replace("[content2]", bbox_str2)
                elif qa_choice in [8, 9, 10, 11]:
                    prompt = prompt.replace("[content]", node)

                # 4 Create different types of QA
                if qa_choice == 1:
                    # Q1: Return the node content within the bbox
                    gt_token_sequence = node
                elif qa_choice == 2:
                    # Q2: Provide the level of the node within the bbox
                    level = find_node_level(gt_parse, node)
                    if level is not None:
                        gt_token_sequence = level + 1
                    else:
                        gt_token_sequence = "None"
                elif qa_choice == 3:
                    # Q3: Find the parent of the node within the bbox
                    gt_token_sequence = find_parent_node(gt_parse, node)
                    if gt_token_sequence is None:
                        gt_token_sequence = "None"
                elif qa_choice == 4:
                    # Q4: Find all children of the node within the bbox
                    gt_token_sequence = find_children_nodes(gt_parse, node)
                    if len(gt_token_sequence) == 0:
                        gt_token_sequence = "None"
                elif qa_choice == 5:
                    # Q5: Determine whether the nodes inside bbox1 and bbox2 form a parent-child relationship
                    is_parent = is_parent_child(gt_parse, node, node2) or is_parent_child(gt_parse, node2, node)
                    if is_parent:
                        gt_token_sequence = "Yes" if lang == 'en' else "是"
                    else:
                        gt_token_sequence = "No" if lang == 'en' else "否"
                elif qa_choice == 6:
                    # Q6: Determine whether the nodes inside bbox1 and bbox2 are siblings
                    is_sibling = are_siblings(gt_parse, node, node2)
                    if is_sibling:
                        gt_token_sequence = "Yes" if lang == 'en' else "是"
                    else:
                        gt_token_sequence = "No" if lang == 'en' else "否"
                elif qa_choice == 7:
                    # Q7: Count the number of nodes within the bbox                    
                    # 1 Calculate the union of the bounding boxes of two random nodes
                    x11, y11, x12, y12 = bbox_str.split(",")
                    bbox1 = [int(x11), int(y11), int(x12), int(y12)]
                    x21, y21, x22, y22 = bbox_str2.split(",")
                    bbox2 = [int(x21), int(y21), int(x22), int(y22)]
                    bboxes = [bbox1, bbox2]
                    union_bbox = cal_bbox_union(bboxes)
                    union_bbox_str = f"{union_bbox[0]},{union_bbox[1]},{union_bbox[2]},{union_bbox[3]}"
                    # print(union_bbox)
                    prompt = prompt.replace("[content]", union_bbox_str)
                    # 2 Count the number of nodes that appear within the union region
                    overlap_count = 0
                    for item in pos_info:
                        x1, y1, x2, y2 = item["xyxy"].split(",")
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        # print(bbox)
                        if is_overlap(bbox, union_bbox):
                            overlap_count += 1
                    gt_token_sequence = overlap_count
                elif qa_choice == 8:
                    # Q8: Return the bounding box of the random node
                    gt_token_sequence = "<bbox>" + bbox_str + "</bbox>"
                elif qa_choice == 9:
                    # Q9: Return the parent node and its bounding box of the random node
                    parent_node = find_parent_node(gt_parse, node)
                    if parent_node is None:
                        gt_token_sequence = "None"
                    else:
                        bbox_str = ""
                        for item in pos_info:
                            label = item["label"]
                            if clean_text(label) == parent_node:
                                bbox_str = item["xyxy"]
                                break
                        gt_token_sequence = parent_node + "<bbox>" + bbox_str + "</bbox>"
                elif qa_choice == 10:
                    # Q10: Return all children and their bounding bboxes of the random node
                    children_nodes = find_children_nodes(gt_parse, node)
                    if len(children_nodes) == 0:
                        gt_token_sequence = "None"
                    else:
                        new_children_nodes = []
                        for ch_node in children_nodes:
                            bbox_str = ""
                            for item in pos_info:
                                label = item["label"]
                                if clean_text(label) == ch_node:
                                    bbox_str = item["xyxy"]
                                    ch_node_bbox = ch_node + " <bbox>" + bbox_str + "</bbox>"
                                    new_children_nodes.append(ch_node_bbox)
                                    break
                        assert len(new_children_nodes)
                        gt_token_sequence = new_children_nodes
                elif qa_choice == 11:
                    # Q11: Return the bounding box of the subgraph with the random node as the theme
                    subtree_nodes = collect_subtree_nodes(gt_parse['map']['node-0'], node)
                    bboxes = []
                    for item in pos_info:
                        label = item["label"]
                        if clean_text(label) in subtree_nodes:
                            x1, y1, x2, y2 = item["xyxy"].split(",")
                            bbox = [int(x1), int(y1), int(x2), int(y2)]
                            bboxes.append(bbox)
                    union_bbox = cal_bbox_union(bboxes)
                    union_bbox_str = f"{union_bbox[0]},{union_bbox[1]},{union_bbox[2]},{union_bbox[3]}"
                    gt_token_sequence = "<bbox>" + union_bbox_str + "</bbox>"

            new_anno = {}
            new_anno["image"] = [anno["image"]]
            new_anno["prompt"] = ""
            new_anno["text"] = ""
            new_anno["system_instruction"] = ""
            new_anno["conversations"] = []
            new_anno["conversations"].append({'from': 'user', 'value': '<image>'})
            new_anno["conversations"].append({'from': 'user', 'value': prompt})
            new_anno["conversations"].append({'from': 'assistant', 'value': str(gt_token_sequence)})
            new_anno["task_type"] = "qa_sft"
            new_annotations.append(new_anno)
    print(len(new_annotations))
    save_jsonl(new_annotations, os.path.join(os.path.dirname(ori_anno_json), f"synth_test_pos_{pos_type}.jsonl"))
