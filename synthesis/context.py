from collections import OrderedDict
import numpy as np
import random
import json
import os
from datetime import datetime
import uuid


class TextReader:
    def __init__(self, path, cache_size=2 ** 28, block_size=2 ** 20):
        self.fp = open(path, "r", encoding="utf-8")
        self.length = 0
        self.offsets = [0]
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.block_size = block_size
        self.bucket_size = cache_size // block_size
        self.idx = 0

        while True:
            text = self.fp.read(self.block_size)
            if not text:
                break
            self.length += len(text)
            self.offsets.append(self.fp.tell())

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        char = self.get()
        self.next()
        return char

    def move(self, idx):
        self.idx = idx

    def next(self):
        self.idx = (self.idx + 1) % self.length

    def prev(self):
        self.idx = (self.idx - 1) % self.length

    def get(self):
        key = self.idx // self.block_size

        if key in self.cache:
            text = self.cache[key]
        else:
            if len(self.cache) >= self.bucket_size:
                self.cache.popitem(last=False)

            offset = self.offsets[key]
            self.fp.seek(offset, 0)
            text = self.fp.read(self.block_size)
            self.cache[key] = text

        self.cache.move_to_end(key)
        char = text[self.idx % self.block_size]
        return char
    

class TreeNode:
    def __init__(self, text):
        self.text = text
        self.node = []


def generate_random_tree(content, max_depth=8, max_children=8, max_nodes=60, max_length=20):
    if not content or max_nodes < 3:
        return None

    used_contents = set()
    def sample_content(max_length=20):
        while True:
            node_c = np.random.randint(5, 50)
            s_idx = np.random.randint(len(content) - node_c)
            
            text = content[s_idx:s_idx + node_c]
            text = text.replace("\\", "\\\\\\\\'),")
            if text.lstrip().startswith('%'):  # Starting with a '%' in pygraphviz will result in failed escaping
                text = text.lstrip().lstrip('%')
                if len(text) == 0:
                    text = ' %'
            if len(text) > max_length:
                # Sample the length of the text for each node using a normal distribution
                values = np.arange(0, int(len(text) // max_length * 1.5) + 1)
                weights = np.exp(-0.5 * ((values - 5) / 2.5) ** 2)
                weights /= weights.sum()
                newline_count = np.random.choice(values, p=weights)
                # Sample the insertion positions for newline character
                if newline_count > 0:
                    possible_positions = list(range(1, len(text)))
                    newline_positions = random.sample(possible_positions, min(newline_count, len(possible_positions)))
                    newline_positions.sort(reverse=True)
                    for pos in newline_positions:
                        text = text[:pos] + '\n' + text[pos:]
            if text not in used_contents:
                used_contents.add(text)
                return text

    root = TreeNode(sample_content(max_length))
    nodes = [root]
    node_count = 1
    current_max_depth = 1
    current_max_children = 0

    def add_children(node, depth):
        nonlocal node_count, current_max_depth, current_max_children
        if depth > current_max_depth:
            current_max_depth = depth
        if depth >= max_depth or node_count >= max_nodes:
            return
        num_children = np.random.randint(0, max_children)
        if num_children > current_max_children:
            current_max_children = num_children
        for _ in range(num_children):
            if node_count >= max_nodes:
                break
            child = TreeNode(sample_content(max_length))
            node.node.append(child)
            nodes.append(child)
            node_count += 1
            add_children(child, depth + 1)

    add_children(root, 1)

    return {
        'root': root,
        'nodes': nodes,
        'max_depth': current_max_depth,
        'max_children': current_max_children,
        'total_nodes': node_count
    }


def tree_to_dict(node):
    node_dict = {"text": node.text}
    if node.node:
        node_dict["node"] = [tree_to_dict(child) for child in node.node]
    return node_dict


def write_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == "__main__":

    for dname in ['en_test', 'zh_test']:
        lang = dname.split('_')[0]
        reader = TextReader('./resources/corpus/{}wiki.txt'.format(lang))
        counts = {'en': 1000, 'zh': 500}
        output_dir = './synth_v2/{}/anno'.format(dname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        total_num = 0
        while True:
            if total_num == 10:  # The quantity to generate at once
                break
            # Randomly sample the corpus
            reader.move(np.random.randint(len(reader)))
            content_len = counts[lang]
            chars = []
            for char in reader:
                if char in "\r\n":
                    continue
                if len(chars) >= content_len:
                    break
                chars.append(char)
            content = "".join(chars).strip()

            # Randomly generate a tree with nodes between [10, 60]
            max_depth = np.random.randint(2, 8)
            max_children = np.random.randint(2, 8)
            max_nodes = np.random.randint(10, 60)
            max_length = 20 if lang == 'en' else 10  # newline character may be inserted if exceeding the max length
            tree_info = generate_random_tree(content, max_depth=max_depth, max_children=max_children, max_nodes=max_nodes, max_length=max_length)
            root = tree_info['root']
            root_json = tree_to_dict(root)
            # print(f"Maximum depth: {tree_info['max_depth']}")
            # print(f"Maximum number of children for a node: {tree_info['max_children']}")
            # print(f"Total number of nodes: {tree_info['total_nodes']}")

            if tree_info['total_nodes'] >= 5:  # Discard generated trees with fewer than 5 nodes, as they are too simple.
                total_num += 1
                write_json(root_json, os.path.join(output_dir, "{}_{}.json".format(timestamp, uuid.uuid4())))
