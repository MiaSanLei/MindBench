import os
import argparse
import json
import numpy as np
from sklearn.metrics import f1_score
from donut_util import JSONParseEvaluator, token2json, save_json


def clean_text(text):
    # Replace line breaks, tabs, and extra spaces with a single space
    return text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()

def is_empty(tree):
    # Check if the 'text' key exists in the tree and if it's not empty
    if 'text' in tree and tree['text'] != '':
        return False
    # Iterate through all the keys in the tree
    for key in tree.keys():
        if key.startswith('node-'):
            if isinstance(tree[key], list):
                for sub_tree in tree[key]:
                    if not is_empty(sub_tree):
                        return False
            else:
                if not is_empty(tree[key]):
                    return False
    return True

def remove_empty_subtrees(tree):
    # Recursively removes empty subtrees from the tree
    if not isinstance(tree, dict):
        return tree
    # Create a list to store the keys that need to be deleted
    keys_to_delete = []
    for key in tree.keys():
        if key.startswith('node-'):
            if isinstance(tree[key], list):
                # Create a new list to store non-empty subtrees
                new_nodes = []
                # Iterate through each subtree in the list
                for sub_tree in tree[key]:
                    if not is_empty(sub_tree):
                        new_nodes.append(remove_empty_subtrees(sub_tree))
                if new_nodes:
                    tree[key] = new_nodes
                else:
                    keys_to_delete.append(key)
            elif isinstance(tree[key], str):
                return tree
            else:
                if is_empty(tree[key]):
                    keys_to_delete.append(key)
                else:  # If the subtree is not empty, recursively call the function on the subtree
                    tree[key] = remove_empty_subtrees(tree[key])
    # Delete the keys that need to be deleted
    for key in keys_to_delete:
        del tree[key]
    # Return the updated tree
    return tree


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, help="Path to evaluation results")
    args = parser.parse_args()

    result_path = args.result_path
    # e.g., {'image': ['synth_v2/en_test/img/20240521200137_09f31cee-f9ea-4ec9-821c-e2e5d4238315.jpg'], 'prompt': '', 'text': '', 'system_instruction': '', 'conversations': [{'from': 'user', 'value': '<image>'}, {'from': 'user', 'value': 'Please describe the central theme of the mind map depicted in the provided image.'}, {'from': 'assistant', 'value': 'BCE in both the digestive and re'}], 'task_type': 'qa_sft', 'model_answer': ' :CE in both the digestive and re'}
    with open(result_path, 'r', encoding="utf-8") as f:
        model_preds = [json.loads(line) for line in f]
    print(len(model_preds))

    task = 'vqa' if '_vqa' in result_path else 'ie'
    if task == 'ie':
        predictions = []
        ground_truths = []
        accs = []

        evaluator = JSONParseEvaluator()

        for idx, pred in enumerate(model_preds):
            img_path = pred['image'][0]
            model_answer = str(pred['model_answer'])
            model_answer = clean_text(model_answer)
            if not model_answer.endswith('</s_node-0></s_map>'):
                model_answer = model_answer + '</s_node-1></s_node-0></s_map>'

            # convert the predicted token sequence to JSON format
            pred_tree = token2json(model_answer)
            if 'map' in pred_tree and 'node-0' in pred_tree['map']:
                pred_tree['map']['node-0'] = remove_empty_subtrees(pred_tree['map']['node-0'])

            # convert the ground truth token sequence to JSON format
            gt = pred['conversations'][2]['value']
            gt = clean_text(gt)
            gt_tree = token2json(gt)

            # Evaluate by field-level F1 score and Tree Edit Distance (TED) based accuracy
            score = evaluator.cal_acc(pred_tree, gt_tree)
            accs.append(score)
            predictions.append(pred_tree)
            ground_truths.append(gt_tree)

        scores = {
        "ted_accuracies": accs,
        "ted_accuracy": np.mean(accs),
        "f1_accuracy": evaluator.cal_f1(predictions, ground_truths),
        }
        print(
        f"Total number of samples: {len(accs)}, Tree Edit Distance (TED) based accuracy score: {scores['ted_accuracy']}, F1 accuracy score: {scores['f1_accuracy']}"
        )

        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        # Save the evaluation scores and predictions to a JSON file
        save_json(os.path.join(os.path.dirname(result_path), 'output.json'), scores)

    elif task == 'vqa':
        groundtruth = [pred['conversations'][2]['value'] for pred in model_preds]
        prediction = [clean_text(pred['model_answer']) for pred in model_preds]

        matches = [int(pred == gt) for pred, gt in zip(prediction, groundtruth)]
        print(f"matches: {matches}")
        # Calculate the F1 score for VQA task
        f1 = f1_score(matches, [1] * len(matches), average='binary')
        print(f1)
