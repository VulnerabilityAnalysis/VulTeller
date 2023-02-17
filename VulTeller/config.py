import argparse


def get_configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default="./data/", help="Path to the dataset in csv format.")
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--max_tokens', type=int, default=30000)
    parser.add_argument('--max_words', type=int, default=10000)
    parser.add_argument('--max_path_num', type=int, default=20)
    parser.add_argument('--max_node_len', type=int, default=200)
    parser.add_argument('--max_token_len', type=int, default=20)
    parser.add_argument('--max_text_len', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--mode', '-m', help='Set mode', default='loc', type=str, choices=['all', 'loc', 'desc'])

    # Parse the argument
    args = parser.parse_args()
    return args
