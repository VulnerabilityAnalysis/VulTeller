from torchtext.data import Field, TabularDataset, BucketIterator, NestedField, Batch, Iterator
import torch
import os


def prepare_train_valid(args, device):
    TOK = Field(tokenize=lambda x: x.split()[:args.max_token_len], lower=True)
    NODE = NestedField(TOK, tokenize=lambda x: eval(x)[:args.max_node_len],
                       include_lengths=True)
    ROW = Field(pad_token=-1, use_vocab=False, preprocessing=lambda x: [i for i in x if i < args.max_node_len])
    PATH = NestedField(ROW, tokenize=lambda x: eval(x)[:args.max_path_num])
    LABEL = Field(tokenize=lambda x: eval(x)[:args.max_node_len],
                  use_vocab=False,
                  pad_token=-1,
                  batch_first=True)

    TEXT = Field(tokenize=lambda x: x.split()[:args.max_text_len],
                 init_token='<sos>',
                 eos_token='<eos>',
                 lower=True)

    train_data, val_data = TabularDataset.splits(args.path, train='train.csv', validation='valid.csv', format='csv',
                                                 fields=[('id', None), ('nodes', NODE), ('paths', PATH),
                                                         ('tags', LABEL), ('trg', TEXT)],
                                                 skip_header=True)

    NODE.build_vocab(train_data, max_size=args.max_tokens)
    TEXT.build_vocab(train_data, max_size=args.max_words)

    torch.save(NODE.vocab, os.path.join(args.path, 'node.pth'))
    torch.save(TEXT.vocab, os.path.join(args.path, 'text.pth'))

    train_iter, val_iter = BucketIterator.splits(
        (train_data, val_data), batch_size=args.batch_size, sort_within_batch=True,
        sort_key=lambda x: len(x.nodes), device=device)
    return NODE, LABEL, TEXT, train_iter, val_iter


def prepare_test(args, device=None):
    ID = Field(tokenize=lambda x: eval(x), use_vocab=False, sequential=False)
    TOK = Field(tokenize=lambda x: x.split()[:args.max_token_len], lower=True)

    NODE = NestedField(TOK, tokenize=lambda x: eval(x)[:args.max_node_len],
                       include_lengths=True)
    ROW = Field(pad_token=-1, use_vocab=False, preprocessing=lambda x: [i for i in x if i < args.max_node_len])
    PATH = NestedField(ROW, tokenize=lambda x: eval(x)[:args.max_path_num])

    LABEL = Field(tokenize=lambda x: eval(x)[:args.max_node_len],
                  use_vocab=False,
                  pad_token=-1)

    TEXT = Field(tokenize=lambda x: x.split(),
                 init_token='<sos>',
                 eos_token='<eos>',
                 lower=True)
    node_vocab = torch.load(os.path.join(args.path, 'node.pth'))
    text_vocab = torch.load(os.path.join(args.path, 'text.pth'))
    fields = [('id', ID), ('nodes', NODE), ('paths', PATH), ('tags', LABEL),
              ('trg', TEXT)]
    test_data = TabularDataset(args.path + 'test.csv', format='csv', fields=fields,
                               skip_header=True)
    # NODE.build_vocab(test_data)
    # TEXT.build_vocab(test_data)
    setattr(TOK, 'vocab', node_vocab)
    setattr(NODE, 'vocab', node_vocab)
    setattr(TEXT, 'vocab', text_vocab)

    test_iter = Iterator(test_data, batch_size=1, train=False, sort=False, sort_within_batch=False,
                         shuffle=False, device=device)

    return NODE, TEXT, test_iter
