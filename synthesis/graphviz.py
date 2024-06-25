import os
import json
import pygraphviz as pgv
import random
import os


# Set a wide range of properties for nodes and edges
# engine style
ENGINES = ['dot', 'neato', 'twopi', 'circo', 'sfdp',]  # 'fdp', 'patchwork', 'osage']

# node style
DEFAULT_FONTS = ['Standard Symbols PS', 'URW Gothic', 'Nimbus Roman', 'Nimbus Sans', 'Nimbus Mono PS', 'DejaVu Sans,DejaVu Sans Light', 'Nimbus Sans Narrow', 'URW Bookman', 'DejaVu Sans', 'DejaVu Sans,DejaVu Sans Condensed', 'C059', 'Z003', 'P052']
# TODO: please install the fonts in the resources/fonts folder manually
FONTS = ["Amorria", "Amorria Brush", "Standard Symbols PS", "Anitha Rounded", "URW Gothic", "Nimbus Roman", "Bartino Stripes", "Hand Stylus", "Grus", "Kryshna", "Glimp", "Glimp Thin Cond", "Metadannye", "The Hungry", "Normál", "Rocks__G", "Astonpoliz", "Nihonium113", "Italique", "Kursiv", "Nimbus Sans", "Copilme", "Vezla_2.0", "Cigalir", "Rocks Serif", "Anitha", "Nimbus Mono PS", "Normal", "ZT Bros Oskon 90s", "ZT Bros Oskon 90s ExtLt", "Murah", "DejaVu Sans", "DejaVu Sans Light", "SELINCAH", "Nimbus Sans Narrow", "URW Bookman", "Beinancia", "Sundae Plush", "DejaVu Sans", "Skeletons", "InColhua", "Huelic", "gaucherand", "Nihonium113 Console", "DejaVu Sans", "DejaVu Sans Condensed", "Caracas", "C059", "Glimp", "Glimp Thin Cond Ita", "Artecallya", "Artecallya Script", "Dogtective", "Fotales", "CaracasFina2.0", "Bartino Outline", "Damesplay", "Damesplay Script", "Ango", "Meticulous", "Lummoxie", "Lummoxie Script", "Z003", "Wortlaut AH", "Dedicool", "AwA", "Novice Writer Regular", "Copilme", "Copilme Light", "Mikoena", "Mikoena Demo", "Comic Wolf 01", "P052", "Colieplay", "Poetirey", "PixelArmy", "Snorkad", "S.A.O 16", "Wendy Neue", "Crosseline", "Crosseline Demo", "Starzoom Shavian", "Bartino"]
ZH_DEFAULT_FONTS = ZH_FONTS = ["Noto Serif HK", "Noto Serif TC", "Noto Serif TC", "Noto Sans SC", "Noto Sans TC", "Noto Sans HK", "ZCOOL XiaoWei", "ZCOOL KuaiLe", "ZCOOL QingKe HuangYou", "Ma Shan Zheng", "Long Cang", "Zhi Mang Xing", "Liu Jian Mao Cao"]

COLORSCHEMES = ['accent', 'blues', 'brbg', 'bugn', 'bupu', 'dark2', 'gnbu', 'greens', 'greys', 'oranges', 'orrd', 'paired', 'pastel1', 'pastel2', 'piyg', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purples', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'reds', 'set1', 'set2', 'set3', 'spectral', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd']
DARK_COLORS = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro']
LIGHT_COLORS = ['white', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow']
FONT_COLORS = ["#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080", "#2F4F4F", "#00008B", "#008B8B", "#B8860B", "#A9A9A9", "#006400", "#BDB76B", "#8B008B", "#556B2F", "#FF8C00", "#9932CC", "#8B0000", "#E9967A"]
COLORS = DARK_COLORS + LIGHT_COLORS
IMAGEPOS = ['tl', 'tc', 'tr', 'ml', 'mc', 'mr', 'bl', 'bc', 'br']
USUAL_SHAPES = ['box', 'ellipse']
SHAPES = ['box', 'ellipse', 'polygon', 'triangle', 'diamond', 'trapezium', 'parallelogram', 'invtriangle', 'invtrapezium', 'invhouse']

# edge style
ARROWHEADS = ['box', 'curve', 'diamond', 'dot', 'inv', 'none', 'normal', 'tee', 'vee', 'invdot', 'invodot', 'ediamond', 'open', 'halfopen', 'empty', 'invempty']
ARROWSIZES = ['1', '1.25', '1.5', '1.75', '2']
STYLES = ['solid', 'dashed', 'dotted', 'bold', 'tapered']


def gen_mindmap_by_pygraphviz(input_data, output_dir, filename, lang='en'):
    if random.random() > 0.5:
        engines = random.sample(ENGINES, 1)
    else:
        engines = ['dot', ]  # simpler layout
    use_edge_color = True if random.random() > 0.5 else False
    use_colorscheme = True if random.random() > 0.5 else False
    if use_colorscheme:
        colorscheme = random.sample(COLORSCHEMES, 1)[0]
        colorrank = random.randint(3,8) if colorscheme in ['accent', 'dark2', 'pastel2', 'set2'] else random.randint(3,9)
        colorscheme = colorscheme + str(colorrank)

    for idx in range(len(engines)):
        # Clear the canvas
        G = pgv.AGraph(directed=True)
        for edge in input_data:
            G.add_node(edge[0])
            G.add_node(edge[1])
            G.add_edge(edge[0], edge[1])

        # Set the properties of the graph
        G.graph_attr['overlap'] = 'false'
        if use_colorscheme:
            G.graph_attr['colorscheme'] = colorscheme
        if engines[idx] in ['circo', 'dot']:  # avoid node overlap
            G.graph_attr['sep'] = '+25,25'
        if engines[idx] == 'dot':
            rankdir = random.sample(['TB', 'LR', 'BT', 'RL'], 1)
            for rr in rankdir:
                G.graph_attr['rankdir'] = rr
        G.graph_attr['bgcolor'] = 'transparent'
        G.graph_attr['labeljust'] = random.choice(['r', 'l', 'c'])

        # Set the font
        if lang == 'en':
            if random.random() > 0.5:
                random_font = random.choice(FONTS)
            else:
                random_font = random.choice(DEFAULT_FONTS)
        else:
            if random.random() > 0.5:
                random_font = random.choice(ZH_FONTS)
            else:
                random_font = random.choice(ZH_DEFAULT_FONTS)
        if random.random() > 0.75:
            random_font = random_font + ' Bold'
        if random.random() > 0.75:
            random_font = random_font + ' italic'
        G.node_attr['fontname'] = random_font

        # Set the properties of a node
        for ino, node in enumerate(G.nodes()):
            node.attr['label'] = node
            if use_colorscheme:
                node.attr['colorscheme'] = colorscheme
                node.attr['fillcolor'] = str(random.randint(1, colorrank))
                node.attr['color'] = str(random.randint(3, colorrank))
            else:
                node.attr['fillcolor'] = random.choice(LIGHT_COLORS)
                node.attr['color'] = random.choice(COLORS)
            if random.random() > 0.1:
                node.attr['style'] = 'filled'
            node.attr['fontcolor'] = random.choice(FONT_COLORS)
            node.attr['fontsize'] = random.randint(40, 60)
            if ino == 0:
                node.attr['fontsize'] = 60

            node.attr['penwidth'] = random.randint(0, 8)

            if random.random() > 0.75:
                node_shape = random.choice(SHAPES)
            else:
                node_shape = random.choice(USUAL_SHAPES)

            if node_shape == 'polygon':
                node.attr['sides'] = random.randint(5, 8)
            if node_shape in ['polygon', 'ellipse', 'circle']:
                node.attr['peripheries'] = random.randint(1, 3)
            node.attr['shape'] = node_shape

        # Set the properties of an edge
        for edge in G.edges():
            edge.attr['arrowhead'] = random.choice(ARROWHEADS) if random.random() > 0.5 else 'normal'
            edge.attr['arrowsize'] = random.choice(ARROWSIZES)
            edge.attr['penwidth'] = random.randint(6, 12)
            edge.attr['style'] = random.choice(STYLES) if random.random() > 0.5 else 'solid'
            if use_edge_color:
                edge.attr['color'] = random.choice(DARK_COLORS)
            elif use_colorscheme:
                edge.attr['color'] = str(random.randint(3, colorrank))

        # Use the layout engine to generate the layout file and render the image
        G.layout(prog='{}'.format(engines[idx]))
        G.write(os.path.join(output_dir, '{}.gv'.format(filename)))
        graph_img = os.path.join(output_dir, '{}.png'.format(filename))
        G.draw(graph_img)

        # Save the content and coordinate of the nodes
        node_attributes = []
        bb = G.graph_attr['bb']
        left, bottom, right, top = map(float, bb.split(','))
        graph_width = right - left
        graph_height = top - bottom

        for node in G.nodes():
            label = node
            pos = node.attr['pos']
            # Convert inches to pixels
            width = float(node.attr['width'])*72.0
            height = float(node.attr['height'])*72.0

            pos = pos.strip('"').split(',')
            pos_x = float(pos[0])
            pos_y = float(pos[1])

            # Change the coordinate origin from the lower-left corner to the upper-left corner
            pos_y = graph_height - pos_y
            # Scale the coordinates to integers between 0 and 999
            x_scaled = (pos_x / graph_width) * 999
            y_scaled = (pos_y / graph_height) * 999
            w_scaled = (width / graph_width) * 999
            h_scaled = (height / graph_height) * 999

            top_left_x = int(x_scaled - w_scaled / 2)
            top_left_y = int(y_scaled - h_scaled / 2)
            bottom_right_x = int(x_scaled + w_scaled / 2)
            bottom_right_y = int(y_scaled + h_scaled / 2)

            node_dict = {"label": label, "xyxy": f"{top_left_x},{top_left_y},{bottom_right_x},{bottom_right_y}"}
            node_attributes.append(node_dict)

        json_data = json.dumps(node_attributes)
        with open(os.path.join(output_dir, '{}.json'.format(filename)), 'w') as file:
            file.write(json_data)


def find_edges(node, parent_text, edges):
    if "node" in node:
        for child in node["node"]:
            edge = (parent_text, child["text"])
            edges.append(edge)
            find_edges(child, child["text"], edges)
