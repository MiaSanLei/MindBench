import json
import random
import os


mmvqa_prompts = {
    'en': [
        'Please describe the central theme of the mind map depicted in the provided image.',
        'Please identify and return the parent node of the node labeled "[content]" in the mind map.',
        'Please identify and return all child nodes of the node labeled "[content]" in the mind map.',
        'Please determine and inform which level the node labeled "[content]" is located on in the mind map.',
        'Please list all the nodes contained on the [content]-th level of the mind map.',
        'Please determine and inform whether the nodes labeled "[content1]" and "[content2]" constitute a parent-child relationship in the mind map.',
        'Please determine and inform whether the nodes labeled "[content1]" and "[content2]" constitute a sibling relationship in the mind map.'
    ],
    'cn': [
        '请描述在提供的图片中思维导图的核心主题。',
        '请识别并返回在思维导图中标记为“[content]”的节点的父节点。',
        '请识别并返回在思维导图中标记为“[content]”的节点的所有子节点。',
        '请确定并告知在思维导图中标记为“[content]”的节点位于哪一层级。',
        '请列出在思维导图中第[content]层包含的所有节点。',
        '请判断并告知在思维导图中标记为“[content1]”和“[content2]”的节点是否构成父子关系。',
        '请判断并告知在思维导图中标记为“[content1]”和“[content2]”的节点是否构成兄弟关系。'
    ]
}

def clean_text(text):
    return text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()

def clean_tree(node):
    if isinstance(node, dict):
        for key, value in node.items():
            node[key] = clean_tree(value)
    elif isinstance(node, list):
        node = [clean_tree(elem) for elem in node]
    elif isinstance(node, str):
        node = clean_text(node)
    return node

def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e, ensure_ascii=False) for e in data]))
    print('save %d samples to %s' % (len(data), filename))


def generate_random_node(tree, level=None):
    """
    Generate a random node at the specified level.
    If level is None, it generates a random node from any level.
    """
    nodes_at_level = []
    if level is not None:
        level_key = f'node-{level}'

    def recurse(subtree, current_level):
        if level is None or current_level == level:
            if 'text' in subtree:
                nodes_at_level.append(subtree['text'])
        for key, value in subtree.items():
            if key.startswith('node-') and isinstance(value, list):
                for child in value:
                    recurse(child, current_level + 1)

    recurse(tree['map']['node-0'], 0)
    
    if not nodes_at_level:
        return None
    return random.choice(nodes_at_level)


def get_root_node(tree):
    """
    Q1: Get the root node of the tree.
    """
    if 'map' in tree and 'node-0' in tree['map'] and 'text' in tree['map']['node-0']:
        return tree['map']['node-0']['text']
    return None

def find_parent_node(tree, target_text):
    """
    Q2: Find the parent of the specified node.
    """
    parent_node = None

    def recurse(subtree, parent):
        nonlocal parent_node
        if 'text' in subtree and subtree['text'] == target_text:
            parent_node = parent
            return
        for key, value in subtree.items():
            if key.startswith('node-') and isinstance(value, list):
                for child in value:
                    recurse(child, subtree)

    recurse(tree['map']['node-0'], None)
    if parent_node and 'text' in parent_node:
        return parent_node['text']
    else:
        return None

def find_children_nodes(tree, target_text):
    """
    Q3: Find all children of the specified node.
    """
    children_nodes = []

    def recurse(subtree):
        if 'text' in subtree and subtree['text'] == target_text:
            for key, value in subtree.items():
                if key.startswith('node-') and isinstance(value, list):
                    for child in value:
                        if 'text' in child:
                            children_nodes.append(child['text'])
            return
        for key, value in subtree.items():
            if key.startswith('node-') and isinstance(value, list):
                for child in value:
                    recurse(child)

    recurse(tree['map']['node-0'])
    return children_nodes

def find_node_level(tree, target_text):
    """
    Q4: Provide the level of the specified node.
    """
    level_of_node = None

    def recurse(subtree, current_level):
        nonlocal level_of_node
        if 'text' in subtree and subtree['text'] == target_text:
            level_of_node = current_level
            return
        for key, value in subtree.items():
            if key.startswith('node-') and isinstance(value, list):
                for child in value:
                    recurse(child, current_level + 1)

    recurse(tree['map']['node-0'], 0)
    return level_of_node

def print_nodes_at_level(tree, level):
    """
    Q5: Print the list of nodes located at the i-th level.
    """
    level_key = f'node-{level}'
    nodes_at_level = []

    def recurse(subtree, current_level):
        if current_level == level:
            if 'text' in subtree:
                nodes_at_level.append(subtree['text'])
            return
        for key, value in subtree.items():
            if key.startswith('node-') and isinstance(value, list):
                for child in value:
                    recurse(child, current_level + 1)

    recurse(tree['map']['node-0'], 0)
    return nodes_at_level

def is_parent_child(tree, parent, child):
    """
    Q6: Determine if the specified parent and child nodes have a parent-child relationship.
    """
    def search_node(subtree, node_to_find):
        if 'text' in subtree and subtree['text'] == node_to_find:
            return subtree
        for key, value in subtree.items():
            if key.startswith('node-') and isinstance(value, list):
                for child in value:
                    result = search_node(child, node_to_find)
                    if result:
                        return result
        return None

    parent_node = search_node(tree['map']['node-0'], parent)
    
    if parent_node:
        for key, value in parent_node.items():
            if key.startswith('node-') and isinstance(value, list):
                for child_node in value:
                    if 'text' in child_node and child_node['text'] == child:
                        return True
    return False

def are_siblings(tree, node1, node2):
    """
    Q7: Determine if the specified nodes are siblings.
    """
    def find_parent(subtree, node_to_find):
        for key, value in subtree.items():
            if key.startswith('node-') and isinstance(value, list):
                for child in value:
                    if 'text' in child and child['text'] == node_to_find:
                        return subtree
                    result = find_parent(child, node_to_find)
                    if result:
                        return result
        return None

    parent_node1 = find_parent(tree['map']['node-0'], node1)
    parent_node2 = find_parent(tree['map']['node-0'], node2)

    if parent_node1 and parent_node2 and parent_node1 == parent_node2:
        return True
    return False


if __name__ == '__main__':
    ori_anno_json = "./annotations/synth_test.json"

    with open(ori_anno_json, "r", encoding='utf-8') as f:
        json_info = json.load(f)
    annotations = json_info["annotations"]
    print(len(annotations))

    new_annotations = []
    for anno in annotations:
        gt_parse = anno["ground_truth"]["gt_parse"]
        tree = clean_tree(gt_parse)
        for _ in range(1):  # You can ask multiple questions about a sample
            node1 = generate_random_node(tree)
            node2 = generate_random_node(tree)

            qa_choice = random.randint(1, 7)
            lang = 'en' if 'synth_v2/en_' in anno["image"] else 'cn'
            prompt = mmvqa_prompts[lang][qa_choice-1]
            if qa_choice == 1:
                answer = get_root_node(tree)
            elif qa_choice == 2:
                prompt = prompt.replace("[content]", node1)
                answer = find_parent_node(tree, node1)
            elif qa_choice == 3:
                prompt = prompt.replace("[content]", node1)
                children_nodes = find_children_nodes(tree, node1)
                if len(children_nodes) == 0:
                    answer = "None"
                else:
                    answer = children_nodes
            elif qa_choice == 4:
                prompt = prompt.replace("[content]", node1)
                level = find_node_level(tree, node1)
                if level is not None:
                    answer = level + 1
                else:
                    answer = "None"
            elif qa_choice == 5:
                level = random.randint(0, 7)  # Synthetic data has a maximum of 8 levels
                prompt = prompt.replace("[content]", str(level+1))
                nodes_at_level = print_nodes_at_level(tree, level)
                if len(nodes_at_level) == 0:
                    answer = "None"
                else:
                    answer = nodes_at_level
            elif qa_choice == 6:
                prompt = prompt.replace("[content1]", node1).replace("[content2]", node2)
                is_parent = is_parent_child(tree, node1, node2) or is_parent_child(tree, node2, node1)
                if is_parent:
                    answer = "Yes" if lang == 'en' else "是"
                else:
                    answer = "No" if lang == 'en' else "否"
            else:
                prompt = prompt.replace("[content1]", node1).replace("[content2]", node2)
                is_sibling = are_siblings(tree, node1, node2)
                if is_sibling:
                    answer = "Yes" if lang == 'en' else "是"
                else:
                    answer = "No" if lang == 'en' else "否"

            new_anno = {}
            new_anno["image"] = [anno["image"]]
            new_anno["prompt"] = ""
            new_anno["text"] = ""
            new_anno["system_instruction"] = ""
            new_anno["conversations"] = []
            new_anno["conversations"].append({'from': 'user', 'value': '<image>'})
            new_anno["conversations"].append({'from': 'user', 'value': prompt})
            if answer is None:
                answer = "None"
            new_anno["conversations"].append({'from': 'assistant', 'value': str(answer)})
            new_anno["task_type"] = "qa_sft"
            new_annotations.append(new_anno)
    print(len(new_annotations))
    save_jsonl(new_annotations, os.path.join(os.path.dirname(ori_anno_json), f"synth_test_structured_vqa.jsonl"))
