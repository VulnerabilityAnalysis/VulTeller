import data
import config
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def compute_accuracy(preds, y, mask):
    preds = torch.round(preds)
    # correct = preds[non_pad_elements].eq(y[non_pad_elements])
    # return correct.sum() / y[non_pad_elements].shape[0]
    correct = (preds == y).float()
    return torch.sum(correct * mask) / torch.sum(mask)


def task_localize(tag_output, tags, tag_pad_idx, criterion_tag):
    tag_output = tag_output.squeeze()
    tag_output = tag_output.view(-1)
    tags = tags.view(-1)
    #
    non_pad_elements = (tags != tag_pad_idx).nonzero()
    mask = (tags != tag_pad_idx).float()
    loss_tag = criterion_tag(tag_output[non_pad_elements], tags[non_pad_elements])
    # loss_tag = torch.sum(criterion_tag(tag_output, tags)*mask)/torch.sum(mask)
    acc = compute_accuracy(tag_output, tags, mask)
    return loss_tag, acc


def task_generate(txt_output, trg, criterion_txt):
    txt_output_dim = txt_output.shape[-1]

    # txt_output = txt_output.view((-1, txt_output_dim))
    # trg = trg.reshape((-1,))

    txt_output = txt_output[1:].view(-1, txt_output_dim)
    trg = trg[1:].view(-1)

    loss_txt = criterion_txt(txt_output, trg)
    return loss_txt


def train(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print("Read data...")
    src_field, tag_field, txt_field, train_iter, val_iter = data.prepare_train_valid(args, device)
    input_dim = len(src_field.vocab)
    txt_output_dim = len(txt_field.vocab)
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim

    tag_pad_idx = -1
    txt_pad_idx = txt_field.vocab.stoi[txt_field.pad_token]

    model = MTModel(input_dim, embed_dim, hidden_dim, txt_output_dim, device).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)

    criterion_tag = nn.BCELoss()
    criterion_txt = nn.CrossEntropyLoss(ignore_index=txt_pad_idx)
    model = model.to(device)

    print("Start training...")
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_acc = 0
        model.train()

        for i, batch in enumerate(tqdm(train_iter)):
            nodes, node_len, token_len = batch.nodes
            paths = batch.paths
            src = nodes, paths
            src_len = node_len, token_len
            tags = batch.tags.float()
            trg = batch.trg

            optimizer.zero_grad()

            tag_output, txt_output = model(src, src_len, args.mode, trg)

            if args.mode == 'loc':
                loss, acc = task_localize(tag_output, tags, tag_pad_idx, criterion_tag)
                epoch_acc += acc.item()
                loss.backward()
            elif args.mode == 'desc':
                loss = task_generate(txt_output, trg, criterion_txt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                trg_values = trg[trg != txt_pad_idx]
                pred_values = txt_output.argmax(-1).squeeze()[trg != txt_pad_idx]
                acc = (pred_values == trg_values).sum()/trg_values.numel()
                epoch_acc += acc.item()
            else:
                loss_tag, acc = task_localize(tag_output, tags, tag_pad_idx, criterion_tag)
                loss_txt = task_generate(txt_output, trg, criterion_txt)
                loss = loss_tag + loss_txt
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                trg_values = trg[trg != txt_pad_idx]
                pred_values = txt_output.argmax(-1).squeeze()[trg != txt_pad_idx]
                acc = (pred_values == trg_values).sum() / trg_values.numel()
                epoch_acc += acc.item()

            optimizer.step()

            epoch_loss += loss.item()
            # print(f'Iter {i} -- Tag loss: {loss_tag.item()}, Txt loss: {loss_txt.item()}')

        train_loss = epoch_loss / len(train_iter)
        train_acc = epoch_acc / len(train_iter)

        valid_loss, valid_acc = evaluate(model, val_iter, criterion_tag, criterion_txt,
                                         tag_pad_idx, txt_pad_idx, args.mode)

        scheduler.step(valid_acc)
        print(scheduler.state_dict())

        if not os.path.exists('./checkpoints/'):
            os.mkdir('./checkpoints/')
        torch.save(model, './checkpoints/%s-model-epoch-%d.pt' % (args.mode, epoch))

        print(f'Epoch: {epoch + 1:02} \tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}'
              f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}')


def evaluate(model, iterator, criterion_tag, criterion_txt, tag_pad_idx, txt_pad_idx, mode='all'):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            nodes, node_len, token_len = batch.nodes
            paths = batch.paths
            src = nodes, paths
            src_len = node_len, token_len
            tags = batch.tags.float()
            trg = batch.trg

            tag_output, txt_output = model(src, src_len, mode, trg, 0)

            if mode == 'loc':
                loss, acc = task_localize(tag_output, tags, tag_pad_idx, criterion_tag)
                epoch_acc += acc.item()
            elif mode == 'desc':
                loss = task_generate(txt_output, trg, criterion_txt)
                # acc = (txt_output.argmax(-1).squeeze() == trg).sum()/trg.numel()
                trg_values = trg[trg != txt_pad_idx]
                pred_values = txt_output.argmax(-1).squeeze()[trg != txt_pad_idx]
                acc = (pred_values == trg_values).sum()/trg_values.numel()
                epoch_acc += acc.item()
            else:
                loss_tag, acc = task_localize(tag_output, tags, tag_pad_idx, criterion_tag)
                loss_txt = task_generate(txt_output, trg, criterion_txt)
                loss = loss_tag + loss_txt
                trg_values = trg[trg != txt_pad_idx]
                pred_values = txt_output.argmax(-1).squeeze()[trg != txt_pad_idx]
                acc = (pred_values == trg_values).sum() / trg_values.numel()
                epoch_acc += acc.item()
            epoch_loss += loss.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == "__main__":
    configs = config.get_configs()
    train(configs)
