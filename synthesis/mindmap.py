import os
import random
import json
import time
import threading
from queue import Queue
from PIL import Image, ImageColor, ImageFilter
Image.MAX_IMAGE_PIXELS = None

from graphviz import gen_mindmap_by_pygraphviz, find_edges

# TODO: change to your image folder
bg_path = './resources/background'
BACKGROUNDS = [os.path.join(bg_path, img) for img in os.listdir(bg_path) if img.endswith('.jpg')]

def gen_low_saturation_bg_color():
    hue = random.randint(0, 359)
    saturation = random.randint(10, 60)
    lightness = random.randint(20, 80)
    return ImageColor.getrgb(f"hsl({hue}, {saturation}%, {lightness}%)")


def syn_mindmap_with_bg(foreground_path, output_path):
    # Random parameter settings
    short_size = 3840
    aspect_ratio = random.uniform(1, 3)
    long_size = int(short_size * aspect_ratio)
    use_bg_color = True if random.random() > 0.5 else False
    use_multi_bg = True if random.random() > 0.5 else False
    use_blurred_bg = True if random.random() > 0.5 else False

    # Open the foreground mind map image
    graph_image = Image.open(foreground_path)
    graph_width, graph_height = graph_image.size

    # Modify canvas size
    if graph_width > graph_height:
        size = (long_size, int(long_size*graph_height/graph_width))
    else:
        size = (int(long_size*graph_width/graph_height), long_size)

    if use_bg_color:
        bg_color = gen_low_saturation_bg_color()
    else:
        bg_color = (255, 255, 255)
    canvas = Image.new('RGB', size, bg_color)
    canvas_width, canvas_height = size

    # Resize the mind map image
    scale = min(canvas_width / float(graph_width), canvas_height / float(graph_height))
    graph_width = int(graph_width * scale)
    graph_height = int(graph_height * scale)
    graph_image = graph_image.resize((graph_width, graph_height), Image.BILINEAR)
    pos_x = random.randint(0, canvas_width - graph_width)
    pos_y = random.randint(0, canvas_height - graph_height)

    if use_multi_bg:
        bg_num = random.randint(0, 5)
    else:
        bg_num = random.randint(0, 1)
    if bg_num == 0:
        canvas.paste(graph_image, (pos_x, pos_y), graph_image)
    else:  # Use 1 to multiple background images
        if bg_num == 1:
            bg = random.choice(BACKGROUNDS)
            bg_img = Image.open(bg)
            bg_width, bg_height = bg_img.size
            if random.random() > 0.5:  # Scale the background image with its aspect ratio
                scale = min(canvas_width / float(bg_width), canvas_height / float(bg_height))
                bg_width = int(bg_width * scale)
                bg_height = int(bg_height * scale)
                bg_resized = bg_img.resize((bg_width, bg_height), Image.BILINEAR)
                pos_x_tmp = random.randint(0, canvas_width - bg_width)
                pos_y_tmp = random.randint(0, canvas_height - bg_height)
                mask = Image.new('L', (bg_width, bg_height), color=255)
                canvas.paste(bg_resized, (pos_x_tmp, pos_y_tmp), mask)
            else:  # Resize the background image to the size of the canvas
                bg_resized = bg_img.resize((canvas_width, canvas_height), Image.BILINEAR)
                mask = Image.new('L', (canvas_width, canvas_height), color=255)
                canvas.paste(bg_resized, (0, 0), mask)
        else:  # Randomly place multiple background images of different scales
            bg_list = random.sample(BACKGROUNDS, bg_num)
            for bg in bg_list:
                bg_img = Image.open(bg)
                bg_width, bg_height = random.randint(50, canvas_width // 2), random.randint(50, canvas_height // 2)
                bg_resized = bg_img.resize((bg_width, bg_height), Image.BILINEAR)
                pos_x_tmp = random.randint(0, canvas_width - bg_width)
                pos_y_tmp = random.randint(0, canvas_height - bg_height)
                mask = Image.new('L', (bg_width, bg_height), color=255)
                canvas.paste(bg_resized, (pos_x_tmp, pos_y_tmp), mask)

        if 'A' in graph_image.getbands():
            mask = graph_image.split()[-1]
        else:
            mask = Image.new('L', (graph_width, graph_height), color=255)
        if use_blurred_bg:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=5))
        canvas.paste(graph_image, (pos_x, pos_y), mask)
    canvas.save(output_path)
    

def process_file(anno, data_dir):
    anno_dir = os.path.join(data_dir, "anno")
    graphviz_dir = os.path.join(data_dir, "graph")
    synthetic_dir = os.path.join(data_dir, "img")
    lang = os.path.basename(data_dir).split('_')[0]

    input_path = os.path.join(graphviz_dir, anno[:-5]+'.png')
    if os.path.exists(input_path):
        return
    with open(os.path.join(anno_dir, anno), 'r', encoding='utf-8') as f:
        tree_data = json.load(f)

    # Convert the tree structure of mind map into edges for rendering mind map images
    edges = []
    find_edges(tree_data, tree_data["text"], edges)
    # Use pygraphviz to draw mind maps without background
    gen_mindmap_by_pygraphviz(input_data=edges, output_dir=graphviz_dir, filename=anno[:-5], lang=lang)

    # Merge graphviz foreground images and background images
    output_path = os.path.join(synthetic_dir, anno[:-5]+'.jpg')
    if os.path.exists(output_path):
        return
    syn_mindmap_with_bg(input_path, output_path)


def worker():
    while True:
        item = q.get()
        if item is None:
            break
        anno, data_dir = item
        process_file(anno, data_dir)
        q.task_done()


if __name__ == "__main__":
    # Multi-threaded processing
    q = Queue()
    num_worker_threads = 4

    # Start worker thread
    start_time = time.time()
    threads = []
    for i in range(num_worker_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    # Add the generation task to the queue
    for dname in ['en_test', 'zh_test']:
        data_dir = f'./synth_v2/{dname}'
        anno_dir = os.path.join(data_dir, "anno")
        graphviz_dir = os.path.join(data_dir, "graph")
        synthetic_dir = os.path.join(data_dir, "img")
        os.makedirs(graphviz_dir, exist_ok=True)
        os.makedirs(synthetic_dir, exist_ok=True)

        for anno in os.listdir(anno_dir):
            q.put((anno, data_dir))

    # Block until all tasks are processed
    q.join()
    print(f"Runtime: {time.time() - start_time} seconds...")

    # Stop the worker threads
    for i in range(num_worker_threads):
        q.put(None)
    for t in threads:
        t.join()

    print("All tasks have been finished...")
