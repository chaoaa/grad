"""
tools/build_keyword_mask.py

离线预计算关键词 mask，输出与 coco_train_target.pkl 结构对齐的 pkl 文件。

使用方法：
    python tools/build_keyword_mask.py \
        --target_path ./mscoco/sent/coco_train_target.pkl \
        --vocab_path  ./mscoco/txt/coco_vocabulary.txt   \
        --output_path ./mscoco/sent/coco_train_keyword_mask.pkl

关键词词性（POS tag）：
    名词: NN, NNS, NNP, NNPS
    动词: VB, VBD, VBG, VBN, VBP, VBZ
    形容词: JJ, JJR, JJS

输出结构：
    dict {image_id: np.ndarray[n_caps, seq_len], dtype=int32}
    关键词位置为 1，其余（含 pad / ignore）为 0。
"""

import argparse
import pickle
import sys
import numpy as np

# 关键词 POS 标签集合
KEYWORD_TAGS = {
    'NN', 'NNS', 'NNP', 'NNPS',          # 名词
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # 动词
    'JJ', 'JJR', 'JJS',                   # 形容词
}


def check_nltk_resources():
    """检查 nltk 资源是否可用，缺失时给出提示。"""
    try:
        import nltk
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print(
            "[ERROR] nltk 'averaged_perceptron_tagger' 未找到。\n"
            "请在 Python 中运行：\n"
            "  import nltk\n"
            "  nltk.download('averaged_perceptron_tagger')\n"
            "  nltk.download('punkt')\n"
        )
        sys.exit(1)


def load_vocab(vocab_path):
    """
    与 lib/utils.py::load_vocab 保持一致：
        vocab[0] = '.'
        vocab[i] = line_i_of_file  (从 i=1 开始)
    返回 idx -> word 字典。
    """
    idx2word = {0: '.'}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            word = line.strip()
            if word:
                idx2word[idx] = word
    return idx2word


def is_keyword(word, tag):
    """判断 (word, tag) 是否属于关键词。"""
    return tag in KEYWORD_TAGS


def build_mask_for_seq(token_ids, idx2word, vocab_size):
    """
    给定一条 token_id 序列（1-D array），返回同长度的 keyword mask（int32）。
    规则：
        token <= 0 或 token >= vocab_size+1  → 0 (pad / ignore / EOS)
        其余                                 → 按 POS tag 决定 0/1
    为减少 pos_tag 调用开销，先收集所有有效词再批量 tag。
    """
    import nltk
    seq_len = len(token_ids)
    mask = np.zeros(seq_len, dtype=np.int32)

    # 收集有效位置和对应词
    valid_positions = []
    words = []
    for pos, tid in enumerate(token_ids):
        tid_int = int(tid)
        if tid_int <= 0 or tid_int > vocab_size:
            continue
        word = idx2word.get(tid_int, None)
        if word is None or word in ('<pad>', '<unk>', '<BOS>', '<EOS>'):
            continue
        valid_positions.append(pos)
        words.append(word)

    if not words:
        return mask

    # 批量 POS 标注
    tagged = nltk.pos_tag(words)
    for pos, (word, tag) in zip(valid_positions, tagged):
        if is_keyword(word, tag):
            mask[pos] = 1

    return mask


def build_keyword_mask(target_path, vocab_path, output_path):
    check_nltk_resources()
    import nltk  # 确保已导入

    print(f"[INFO] 读取 target: {target_path}")
    with open(target_path, 'rb') as f:
        target_data = pickle.load(f, encoding='bytes')

    print(f"[INFO] 读取 vocab: {vocab_path}")
    idx2word = load_vocab(vocab_path)
    vocab_size = max(idx2word.keys())   # 最大合法 token index
    print(f"[INFO] vocab size = {vocab_size}")

    keyword_mask_data = {}
    image_ids = list(target_data.keys())
    total = len(image_ids)

    print(f"[INFO] 开始处理 {total} 张图像...")
    for i, image_id in enumerate(image_ids):
        seqs = target_data[image_id]   # [n_caps, seq_len]，numpy array or list
        seqs = np.array(seqs)
        if seqs.ndim == 1:
            seqs = seqs[np.newaxis, :]  # 兼容单条 caption 的情况

        n_caps, seq_len = seqs.shape
        kw_mask = np.zeros((n_caps, seq_len), dtype=np.int32)
        for cap_idx in range(n_caps):
            kw_mask[cap_idx] = build_mask_for_seq(seqs[cap_idx], idx2word, vocab_size)

        keyword_mask_data[image_id] = kw_mask

        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{total}] done")

    print(f"[INFO] 保存到: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(keyword_mask_data, f)

    print("[INFO] 完成！")
    # 简单统计关键词密度
    total_kw = sum(v.sum() for v in keyword_mask_data.values())
    total_tok = sum(v.size for v in keyword_mask_data.values())
    print(f"[INFO] 关键词 token 占比: {100.0 * total_kw / max(total_tok, 1):.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build keyword mask for KSC loss.')
    parser.add_argument('--target_path', type=str,
                        default='./mscoco/sent/coco_train_target.pkl',
                        help='Path to coco_train_target.pkl')
    parser.add_argument('--vocab_path', type=str,
                        default='./mscoco/txt/coco_vocabulary.txt',
                        help='Path to coco_vocabulary.txt')
    parser.add_argument('--output_path', type=str,
                        default='./mscoco/sent/coco_train_keyword_mask.pkl',
                        help='Output path for keyword mask pkl')
    args = parser.parse_args()

    build_keyword_mask(args.target_path, args.vocab_path, args.output_path)
