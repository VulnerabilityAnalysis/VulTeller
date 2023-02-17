import json
import re
import pandas as pd
import numpy as np
import os
from antlr4 import *
from CPP14Lexer import CPP14Lexer
from nltk.tokenize import word_tokenize, sent_tokenize
import networkx as nx
import signal


def get_data_flow(id_, line_map, valid_ids):
    paths = []
    if os.path.exists("taint/%s.ta" % id_):
        with open("taint/%s.ta" % id_) as ft:
            path = []
            for line in ft.readlines():
                res = re.findall(r'\| lineNumber\|', line)
                if res:
                    if path and path not in paths:
                        paths.append(path)
                    path = []
                    continue
                else:
                    res = re.findall(r'\|\s+(\d+)\s+\|', line)
                    if res:
                        point = int(res[0])
                        if point in line_map:
                            point = valid_ids.index(line_map[point])
                            if point not in path:
                                path.append(point)
    return paths


def parse_src(code_lines, locs):
    nodes = []
    labels = []
    for i, cl in enumerate(code_lines):
        if cl:
            lexer = CPP14Lexer(InputStream(cl))
            stream = CommonTokenStream(lexer)
            stream.fill()
            nodes.append(' '.join([token.text for token in stream.tokens[:-1]]))
            labels.append(1 if i + 1 in locs else 0)
    paths = [[i for i in range(len(nodes) - 1)]]
    return nodes, paths, labels


def simplify(row):
    id_, func, locs = str(row['id']), row['function'], eval(row['location'])
    lines = func.split('\n')
    start, stop = 2, len(lines) + 1
    root = "./"
    with open(root + 'prop/%s_all.json' % id_) as fj:
        props = json.load(fj)
    if not os.path.exists(root + "cfg/%s_cfg.dot" % id_):
        return parse_src(lines, locs)
    with open(root + "cfg/%s_cfg.dot" % id_) as fr:
        cfg_dot = fr.read()

    """
    "7" [label = <(METHOD,max)<SUB>1</SUB>> ]
    """
    print(id_)
    node_ids = sorted([eval(v) for v in re.findall(r'"(\d+)" \[label = <', cfg_dot)])
    try:
        begin_node = int(re.findall(r'"(\d+)" \[label = <\(METHOD,', cfg_dot)[0])
        exit_node = int(re.findall(r'"(\d+)" \[label = <\(METHOD_RETURN,', cfg_dot)[0])
    except:
        return parse_src(lines, locs)
    nodes = []
    labels = []
    valid_ids = []
    replace_ids = {}
    unique_nodes = dict()
    line_map = dict()

    for idx, nid in enumerate(node_ids):
        for prop in props:
            if 'id' in prop and 'lineNumber' in prop:
                if prop['id'] == nid:

                    line_number = prop['lineNumber']
                    st_code = prop['code']
                    if line_number in unique_nodes and nid not in [begin_node, exit_node]:
                        if len(st_code) > len(unique_nodes.get(line_number)[0]):
                            unique_nodes[line_number] = (st_code, nid)
                        else:
                            replace_ids[nid] = unique_nodes.get(line_number)[1]
                            break
                    else:
                        unique_nodes[line_number] = (st_code, nid)
                    lexer = CPP14Lexer(InputStream(st_code))
                    stream = CommonTokenStream(lexer)
                    stream.fill()

                    replace_ids[nid] = nid
                    nodes.append(' '.join([token.text for token in stream.tokens[:-1]]))
                    valid_ids.append(nid)

                    line_map[line_number] = nid

                    if line_number - start + 1 in locs:
                        label = 1
                    else:
                        label = 0
                    labels.append(label)

                    break
    """
    "25" -> "27"  [ label = "DDG: &lt;RET&gt;"]
    """
    assert len(nodes) == len(labels)
    if sum(labels) == 0:
        print("ERROR: no positive labels found!")
        global error_num
        error_num += 1
        return parse_src(lines, locs)
    # check_graph(cfg_dot, replace_ids)
    graph = nx.DiGraph()
    relations = [(valid_ids.index(replace_ids.get(eval(v1))), valid_ids.index(replace_ids.get(eval(v2))))
                 for v1, v2 in re.findall(r'"(\d+)" -> "(\d+)"', cfg_dot)
                 if replace_ids.get(eval(v1)) != replace_ids.get(eval(v2)) and
                 replace_ids.get(eval(v1)) in valid_ids and replace_ids.get(eval(v2)) in valid_ids]

    graph.add_edges_from(list(set(relations)))

    # print(len(graph.nodes), len(graph.edges))

    def handle_timeout(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(120)

    try:
        cf_paths = nx.all_simple_edge_paths(graph, valid_ids.index(begin_node), valid_ids.index(exit_node))
        paths = []
        counts = {}

        taint_paths = get_data_flow(id_, line_map, valid_ids)

        if taint_paths:
            for i, p in enumerate(cf_paths):
                p = [e[0] for e in p] + [p[-1][1]]
                paths.append(p)
                counts[i] = 1
                for tp in taint_paths:
                    if set(tp) <= set(p):
                        if i in counts:
                            counts[i] += 1

            paths = [x for _, x in sorted(zip(counts, paths), reverse=True)][:K]
            # print(counts)
        else:
            for i, p in enumerate(cf_paths):
                if i == K:
                    break
                p = [e[0] for e in p] + [p[-1][1]]
                paths.append(p)

    except TimeoutError:
        print("Timeout Error")
        return parse_src(lines, locs)
    except Exception:
        print("Unknown Error")
        return parse_src(lines, locs)
    finally:
        signal.alarm(0)

    return nodes, paths, labels


def tokenize_desc(text):
    text = re.sub(r'((\d+\.)?(\d+\.)?(\*|\d+)(-([a-zA-Z0-9.]+))?(\+([a-zA-Z0-9.]+))?(,|\s+|and)*)+', '_VERSION_,', text)
    text = re.sub(r'\w+(/\w+)*\.(c|cc|cpp)', '_FILE_', text)
    sentences = sent_tokenize(text)[:2]
    words = sum([word_tokenize(s) for s in sentences], [])
    return ' '.join(words)


def get_subset(dataset):
    dataset = pd.read_csv(f'../data/{dataset}.csv')
    dataset['nodes'], dataset['edges'], dataset['labels'] = \
        zip(*dataset.apply(simplify, axis=1))
    dataset['description'] = dataset['description'].apply(tokenize_desc)
    dataset.dropna(inplace=True)
    dataset = dataset[['id', 'nodes', 'edges', 'labels', 'description']]
    return dataset


if __name__ == '__main__':
    error_num = 0
    K = 20
    train = get_subset('train')
    val = get_subset('val')
    test = get_subset('test')
    print(len(train), len(val), len(test))
    print(error_num)
    save_dir = f'../data'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    train.to_csv(f'{save_dir}/train.csv', index=False)
    val.to_csv(f'{save_dir}/valid.csv', index=False)
    test.to_csv(f'{save_dir}/test.csv', index=False)

