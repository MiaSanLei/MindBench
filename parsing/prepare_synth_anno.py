import json
import os
import random


def update_node_key(data, depth=0):
    """Add hierarchical sequence numbers to nested nodes"""
    if isinstance(data, dict):
        if 'node' in data:
            # Rename the 'node' key to 'node-{depth}'
            data[f'node-{depth}'] = data.pop('node')
            update_node_key(data[f'node-{depth}'], depth+1)
        for key in data:
            if isinstance(data[key], list):
                for item in data[key]:
                    update_node_key(item, depth)
    elif isinstance(data, list):
        for item in data:
            update_node_key(item, depth)

def write_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def prepare_synthmm_anno(data_dir):
    annotations = []
    for dname in os.listdir(data_dir):
        anno_p = os.path.join(data_dir, dname, 'anno')
        if not os.path.isdir(anno_p):
            continue
        json_files = [os.path.join(anno_p, file) for file in os.listdir(anno_p) if file.endswith('.json')]
        print("Total JSON files in {}: {}".format(anno_p, len(json_files)))

        for jfile in json_files:        
            with open(jfile, 'r') as f:
                json_info = json.load(f)
            json_info = {'node': json_info}
            update_node_key(json_info)
            res_dict = dict()
            res_dict['ground_truth'] = {'gt_parse': {'map': json_info}}
            res_dict['image'] = os.path.join(os.path.basename(data_dir), dname, "img", os.path.basename(jfile).replace('.json', '.jpg'))
            annotations.append(res_dict)
    print(len(annotations))
    random.shuffle(annotations)
    save_path = './annotations/synth_test.json'
    write_json({"annotations": annotations}, save_path)


if __name__ == "__main__":
    data_dir = '../synthesis/synth_v2'
    prepare_synthmm_anno(data_dir)