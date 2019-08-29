  # Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch
from torch import nn
from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerModel  # , TRANSFORMER_LAYER_PARAMS
from .transformer_xl import TransformerModelXL
from .transformer_adaptive_span import TransformerModelAdpSpn
from .memory import HashingMemory

import ipdb

logger = getLogger()

def init_weight(weight, args):
  if args.init == 'uniform':
    nn.init.uniform_(weight, -args.init_range, args.init_range)
  elif args.init == 'normal':
    nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
  nn.init.constant_(bias, 0.0)


def weights_init(m, args):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    if hasattr(m, 'weight') and m.weight is not None:
      init_weight(m.weight, args)
    if hasattr(m, 'bias') and m.bias is not None:
      init_bias(m.bias)
  elif classname.find('AdaptiveEmbedding') != -1:
    if hasattr(m, 'emb_projs'):
      for i in range(len(m.emb_projs)):
        if m.emb_projs[i] is not None:
          nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
  elif classname.find('Embedding') != -1:
    if hasattr(m, 'weight'):
      init_weight(m.weight, args)
  elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
    if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
      init_weight(m.cluster_weight, args)
    if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
      init_bias(m.cluster_bias)
    if hasattr(m, 'out_projs'):
      for i in range(len(m.out_projs)):
        if m.out_projs[i] is not None:
          nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
  elif classname.find('LayerNorm') != -1:
    if hasattr(m, 'weight'):
      nn.init.normal_(m.weight, 1.0, args.init_std)
    if hasattr(m, 'bias') and m.bias is not None:
      init_bias(m.bias)
  elif classname.find('TransformerLM') != -1:
    if hasattr(m, 'r_emb'):
      init_weight(m.r_emb, args)
    if hasattr(m, 'r_w_bias'):
      init_weight(m.r_w_bias, args)
    if hasattr(m, 'r_r_bias'):
      init_weight(m.r_r_bias, args)
    if hasattr(m, 'r_bias'):
      init_bias(m.r_bias)


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # memory
    if params.use_memory:
        HashingMemory.check_params(params)
        if not isinstance(params.mem_enc_positions, list): # params are loaded from checkpoint
            s_enc = [x for x in params.mem_enc_positions.split(',') if x != '']
            assert len(s_enc) == len(set(s_enc))
            assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_enc)
            params.mem_enc_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_enc]

        if not isinstance(params.mem_dec_positions, list): # params are loaded from checkpoint
            s_dec = [x for x in params.mem_dec_positions.split(',') if x != '']
            assert len(s_dec) == len(set(s_dec))
            assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_dec)
            params.mem_dec_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_dec]

        assert len(params.mem_enc_positions) + len(params.mem_dec_positions) > 0
        assert len(params.mem_enc_positions) == 0 or 0 <= min([x[0] for x in params.mem_enc_positions]) <= max([x[0] for x in params.mem_enc_positions]) <= params.n_layers - 1
        assert len(params.mem_dec_positions) == 0 or 0 <= min([x[0] for x in params.mem_dec_positions]) <= max([x[0] for x in params.mem_dec_positions]) <= params.n_layers - 1

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def build_model(params, dico):
    """
    Build model.
    """
    if params.encoder_only:
        
        # build
        if params.model_type == 'transformer':
            model = TransformerModel(params, dico, is_encoder=True, with_output=True)
        elif params.model_type == 'transformer_xl':
            model = TransformerModelXL(params)
            def init_func(m):
                return weights_init(m, args=params)
            model.apply(init_func)
            model.word_emb.apply(init_func)  # ensure embedding init is not
        elif params.model_type == 'transformer_adaptive_span':
            model = TransformerModelAdpSpn(params)
            

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            # # HACK to reload models with less layers
            # for i in range(12, 24):
            #     for k in TRANSFORMER_LAYER_PARAMS:
            #         k = k % i
            #         if k in model.state_dict() and k not in reloaded:
            #             logger.warning("Parameter %s not found. Ignoring ..." % k)
            #             reloaded[k] = model.state_dict()[k]

            model.load_state_dict(reloaded)

        if 'xl' not in params.model_type:
            logger.info("Model: {}".format(model))
        logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

        return model.cuda()

    else:
        # build
        encoder = TransformerModel(params, dico, is_encoder=True, with_output=True)  # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
        decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(encoder, dico, word2id, embeddings)
            set_pretrain_emb(decoder, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            enc_path, dec_path = params.reload_model.split(',')
            assert not (enc_path == '' and dec_path == '')

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder from %s ..." % enc_path)
                enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}
                encoder.load_state_dict(enc_reload)

            # reload decoder
            if dec_path != '':
                logger.info("Reloading decoder from %s ..." % dec_path)
                dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
                if all([k.startswith('module.') for k in dec_reload.keys()]):
                    dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
                for i in range(params.n_layers):
                    for name in DECODER_ONLY_PARAMS:
                        if name % i not in dec_reload:
                            logger.warning("Parameter %s not found." % (name % i))
                            dec_reload[name % i] = decoder.state_dict()[name % i]
                decoder.load_state_dict(dec_reload)

        logger.debug("Encoder: {}".format(encoder))
        logger.debug("Decoder: {}".format(decoder))
        logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

        return encoder.cuda(), decoder.cuda()
