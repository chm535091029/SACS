import torch
# import dgl
# from dgl.batch import batch as bc

def vectorize(ex, model):
    """Vectorize a single example."""
    src_dict = model.src_dict
    tgt_dict = model.tgt_dict

    code, summary = ex['code'], ex['summary']
    vectorized_ex = dict()
    vectorized_ex['id'] = code.id
    vectorized_ex['language'] = code.language

    vectorized_ex['code'] = code.text
    vectorized_ex['code_tokens'] = code.tokens
    vectorized_ex['code_char_rep'] = None
    vectorized_ex['code_type_rep'] = None
    vectorized_ex['code_mask_rep'] = None
    vectorized_ex['use_code_mask'] = False
    vectorized_ex['code_struc_rep'] = None
    vectorized_ex['use_code_struc'] = False
    vectorized_ex['src_tokens'] = code.src_tokens



    vectorized_ex['code_word_rep'] , vectorized_ex['src_code_rep']= code.vectorize(word_dict=src_dict) #
    vectorized_ex['code_word_rep'] = torch.LongTensor(vectorized_ex['code_word_rep'])
    vectorized_ex['src_code_rep'] = torch.LongTensor(vectorized_ex['src_code_rep'])
    if model.args.use_src_char:
        vectorized_ex['code_char_rep'] = torch.LongTensor(code.vectorize(word_dict=src_dict, _type='char'))
    if model.args.use_code_type:
        vectorized_ex['code_type_rep'] = torch.LongTensor(code.type)
    if code.mask:
        vectorized_ex['code_mask_rep'] = torch.LongTensor(code.mask)
        vectorized_ex['use_code_mask'] = True
    if code.struc is not None:
        vectorized_ex['code_struc_rep'] = torch.LongTensor(code.struc)
        vectorized_ex['code_level_rep'] = torch.LongTensor(code.level)
        vectorized_ex['edge_rep'] = torch.LongTensor(code.edge)
        vectorized_ex['map'] = code.map
        vectorized_ex['use_code_struc'] = True

    vectorized_ex['summ'] = None
    vectorized_ex['summ_tokens'] = None
    vectorized_ex['init_summ_tokens'] = None
    vectorized_ex['stype'] = None
    vectorized_ex['summ_word_rep'] = None
    vectorized_ex['summ_char_rep'] = None
    vectorized_ex['target'] = None

    if summary is not None:
        vectorized_ex['summ'] = summary.text
        vectorized_ex['summ_tokens'] = summary.tokens
        vectorized_ex['init_summ_tokens'] = summary.init_tokens
        vectorized_ex['stype'] = summary.type
        vectorized_ex['summ_word_rep'],vectorized_ex['init_summ_word_rep'] = summary.vectorize(word_dict=tgt_dict)
        vectorized_ex['summ_word_rep'] = torch.LongTensor(vectorized_ex['summ_word_rep'])
        vectorized_ex['init_summ_word_rep'] = torch.LongTensor(vectorized_ex['init_summ_word_rep'])
        if model.args.use_tgt_char:
            vectorized_ex['summ_char_rep'] = torch.LongTensor(summary.vectorize(word_dict=tgt_dict, _type='char'))
        # target is only used to compute loss during training
        vectorized_ex['target'] = vectorized_ex['summ_word_rep']

    vectorized_ex['src_vocab'] = code.src_vocab
    vectorized_ex['use_src_word'] = model.args.use_src_word
    vectorized_ex['use_tgt_word'] = model.args.use_tgt_word
    vectorized_ex['use_src_char'] = model.args.use_src_char
    vectorized_ex['use_tgt_char'] = model.args.use_tgt_char
    vectorized_ex['use_code_type'] = model.args.use_code_type

    return vectorized_ex


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    use_src_word = batch[0]['use_src_word']
    use_tgt_word = batch[0]['use_tgt_word']
    use_src_char = batch[0]['use_src_char']
    use_tgt_char = batch[0]['use_tgt_char']
    use_code_type = batch[0]['use_code_type']
    use_code_mask = batch[0]['use_code_mask']
    use_code_struc = batch[0]['use_code_struc']
    ids = [ex['id'] for ex in batch]
    language = [ex['language'] for ex in batch]

    # --------- Prepare Code tensors ---------
    src_codes = [ex['src_code_rep'] for ex in batch]
    code_words = [ex['code_word_rep'] for ex in batch]
    code_chars = [ex['code_char_rep'] for ex in batch]
    code_type = [ex['code_type_rep'] for ex in batch]
    code_mask = [ex['code_mask_rep'] for ex in batch]
    max_src_code_len = max([d.size(0) for d in src_codes])
    max_code_len = max([d.size(0) for d in code_words])
    maps = [ex['map'] for ex in batch]
    if use_src_char:
        max_char_in_code_token = code_chars[0].size(1)
    code_struc = [ex['code_struc_rep'] for ex in batch]
    code_level = [ex['code_level_rep'] for ex in batch]
    edges = [ex['edge_rep'] for ex in batch]
    edge_rep = torch.cat(edges,dim=1)
    node_rep = torch.cat(code_words,dim=0)
    node_batch = [torch.full((d.size(0),), i) for i, d in enumerate(code_words)]
    node_batch_rep = torch.cat(node_batch, dim=0)
    node_level_rep = [code_level[i].squeeze(0)[:d.size(0)] for i, d in enumerate(code_words)]
    node_level_rep = torch.cat(node_level_rep)
    # Batch Code Representations
    code_len_rep = torch.zeros(batch_size, dtype=torch.long)
    src_code_len_rep = torch.zeros(batch_size, dtype=torch.long)
    src_code_rep = torch.zeros(batch_size, max_src_code_len, dtype=torch.long)
    code_word_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_src_word else None
    code_type_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_code_type else None
    code_mask_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_code_mask else None
    code_char_rep = torch.zeros(batch_size, max_code_len, max_char_in_code_token, dtype=torch.long) \
        if use_src_char else None
    # max_code_len
    code_level_rep = torch.zeros(batch_size, 1, max_code_len, dtype=torch.long) \
        if use_code_struc else None

    code_struc_rep = torch.zeros(batch_size, max_code_len, max_code_len, dtype=torch.long) \
        if use_code_struc else None



    source_maps = []
    src_vocabs = []
    # batched_g = []
    for i in range(batch_size):
        code_len_rep[i] = code_words[i].size(0)
        src_code_len_rep[i] = src_codes[i].size(0)

        src_code_rep[i, :src_codes[i].size(0)].copy_(src_codes[i])
        if use_src_word:
            code_word_rep[i, :code_words[i].size(0)].copy_(code_words[i])
        if use_code_type:
            code_type_rep[i, :code_type[i].size(0)].copy_(code_type[i])
        if use_code_mask:
            code_mask_rep[i, :code_mask[i].size(0)].copy_(code_mask[i])
        if use_src_char:
            code_char_rep[i, :code_chars[i].size(0), :].copy_(code_chars[i])
        if use_code_struc:
            code_struc_rep[i, :, :].copy_(code_struc[i][:max_code_len, :max_code_len])
            code_level_rep[i, :, :].copy_(code_level[i][:, :max_code_len])

        #
        # g = dgl.DGLGraph().to(torch.device("cuda:0"))
        # g.add_nodes(max_code_len)
        # g.add_edges(code_struc_rep[i, 0, :].tolist(),code_struc_rep[i, 1, :].tolist())
        # batched_g.append(g)
        context = batch[i]['code_tokens']
        vocab = batch[i]['src_vocab']
        src_vocabs.append(vocab)
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([vocab[w] for w in context])
        source_maps.append(src_map)

    # --------- Prepare Summary tensors ---------
    no_summary = batch[0]['summ_word_rep'] is None
    if no_summary:
        init_summ_len_rep = None
        init_summ_rep = None
        summ_len_rep = None
        summ_word_rep = None
        summ_char_rep = None
        tgt_tensor = None
        alignments = None
    else:
        summ_words = [ex['summ_word_rep'] for ex in batch]
        init_summ_tokens = [ex['init_summ_word_rep'] for ex in batch]
        summ_chars = [ex['summ_char_rep'] for ex in batch]
        max_sum_len = max([q.size(0) for q in summ_words])
        max_init_sum_len = max([q.size(0) for q in init_summ_tokens])
        if use_tgt_char:
            max_char_in_summ_token = summ_chars[0].size(1)
        init_summ_len_rep = torch.zeros(batch_size, dtype=torch.long)
        init_summ_rep = torch.zeros(batch_size,max_init_sum_len, dtype=torch.long)
        summ_len_rep = torch.zeros(batch_size, dtype=torch.long)
        summ_word_rep = torch.zeros(batch_size, max_sum_len, dtype=torch.long) \
            if use_tgt_word else None
        summ_char_rep = torch.zeros(batch_size, max_sum_len, max_char_in_summ_token, dtype=torch.long) \
            if use_tgt_char else None

        max_tgt_length = max([ex['target'].size(0) for ex in batch])
        tgt_tensor = torch.zeros(batch_size, max_tgt_length, dtype=torch.long)
        alignments = []
        for i in range(batch_size):
            summ_len_rep[i] = summ_words[i].size(0)
            init_summ_len_rep[i] = init_summ_tokens[i].size(0)
            if use_tgt_word:
                summ_word_rep[i, :summ_words[i].size(0)].copy_(summ_words[i])
                try:
                    init_summ_rep[i, :init_summ_tokens[i].size(0)].copy_(init_summ_tokens[i])
                except:
                    print(f"init_summ_rep:{init_summ_tokens[i].size(0)} init_summ_rep:{init_summ_rep.shape} ")
                    raise
            if use_tgt_char:
                summ_char_rep[i, :summ_chars[i].size(0), :].copy_(summ_chars[i])
            #
            # tgt_len = batch[i]['target'].size(0)
            tgt_len = batch[i]['target'].size(0)
            tgt_tensor[i, :tgt_len].copy_(batch[i]['target'])
            target = batch[i]['summ_tokens']
            align_mask = torch.LongTensor([src_vocabs[i][w] for w in target])
            alignments.append(align_mask)

    return {
        'ids': ids,
        'language': language,
        'batch_size': batch_size,
        'code_word_rep': code_word_rep,
        'code_char_rep': code_char_rep,
        'code_type_rep': code_type_rep,
        'code_mask_rep': code_mask_rep,
        'code_len': code_len_rep,
        'summ_word_rep': summ_word_rep,
        'summ_char_rep': summ_char_rep,
        'summ_len': summ_len_rep,
        'tgt_seq': tgt_tensor,
        'code_text': [ex['code'] for ex in batch],
        'code_tokens': [ex['code_tokens'] for ex in batch],
        'summ_text': [ex['summ'] for ex in batch],
        'summ_tokens': [ex['summ_tokens'] for ex in batch],
        'src_vocab': src_vocabs,
        'src_map': source_maps,
        'alignment': alignments,
        'stype': [ex['stype'] for ex in batch],
        'code_struc_rep': code_struc_rep,
        'init_summ_rep':init_summ_rep,
        'init_summ_len_rep':init_summ_len_rep,
        'code_level_rep':code_level_rep,
        # 'batched_g':bc(batched_g)
        'src_code_len_rep':src_code_len_rep,
        'src_code_rep':src_code_rep,
        'map':maps,
        'edge_rep':edge_rep,
        'node_rep':node_rep,
        'node_batch_rep':node_batch_rep,
        'node_level_rep':node_level_rep
    }
