import torch
import config
from data import prepare_test
from tqdm import tqdm
from metrics import corpus_bleu, Rouge


def translate_sentence(src, src_len, trg_field, model, device, max_len=50):
    model.eval()

    node_len, token_len = src_len
    with torch.no_grad():
        encoder_outputs, enc_final_hs = model.encoder(src, src_len)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
    # model.decoder.init_state(enc_final_hs)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    attentions = []
    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        mask = model.sequence_mask(src_len[0])
        with torch.no_grad():
            output, enc_final_hs, attention = model.decoder(
                trg_tensor, enc_final_hs, encoder_outputs, mask)
        attentions.append(attention)
        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions


def compute_accuracy(preds, y, mask):
    preds = torch.round(preds)
    correct = (preds == y).float()
    return torch.sum(correct * mask) / torch.sum(mask)


def task_localize(tag_output, tags, tag_pad_idx):
    tag_output = tag_output.squeeze()

    mask = (tags != tag_pad_idx).float()
    acc = compute_accuracy(tag_output, tags, mask)
    return acc


def infer(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    best_model = torch.load(f'./checkpoints/{args.mode}-model-epoch-{9 if args.mode == "loc" else 49}.pt').to(device)
    src_field, tgt_field, test_iter = prepare_test(args, device)
    test_acc = 0
    precisions, recalls, f1_scores = [], [], []
    hyps = []
    refs = []
    for i, item in enumerate(tqdm(test_iter)):

        nodes, node_len, token_len = item.nodes
        paths = item.paths
        src = nodes, paths
        src_len = node_len, token_len
        tags = item.tags.squeeze().float()
        trg = item.trg
        tag_output, txt_output = best_model(src, src_len, args.mode, trg)
        if args.mode == 'loc':
            true_tags = tags.tolist()
            pred_tags = (tag_output.squeeze() > 0.1).float()
            pred_tags = pred_tags.tolist()

            test_acc = test_acc + 1 if pred_tags == true_tags else test_acc
            true_lines = [i for i, val in enumerate(true_tags) if val == 1]
            pred_lines = [i for i, val in enumerate(pred_tags) if val == 1]

            intersection = len(list(set(true_lines).intersection(pred_lines)))
            if pred_lines and true_lines:
                p = intersection / len(list(set(pred_lines)))
                r = intersection / len(list(set(true_lines)))
            else:
                p, r, f1 = 0, 0, 0
            precisions.append(p)
            recalls.append(r)

        elif args.mode == 'desc':

            trg = item.dataset.examples[i].trg
            trans, _ = translate_sentence(src, src_len, tgt_field, best_model, device)

            trans = trans[:-1]
            hyps.append(' '.join(trans))
            refs.append(' '.join(trg))
        else:
            pass
    if precisions:
        total_num = len(test_iter)
        test_p = sum(precisions) / len(precisions)
        test_r = sum(recalls) / len(recalls)
        print(
            f'Acc: {test_acc / total_num * 100:.2f}, P: {test_p * 100:.2f}, '
            f'R: {test_r * 100:.2f}, '
            f'F1: {2 * test_p * test_r / (test_p + test_r) * 100:.2f}')

    if hyps:
        bleu = corpus_bleu(hyps, refs)
        print(f'BLEU score = {bleu * 100:.2f}')

        # Compute ROUGE scores
        rouge_calculator = Rouge()
        rouge_l = rouge_calculator.compute_score(hyps, refs)
        print(f'ROUGE-L score = {rouge_l * 100:.2f}')


if __name__ == "__main__":
    args = config.get_configs()
    infer(args)
