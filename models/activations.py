

import numpy as np

from models.cornet import get_cornet_model
from models.resnet import resnet_model
from models.dino import dino_model, dino_model_with_hooks

def get_cornet_activations_batch(model, ims, output_path = None, layer='V4', sublayer='avgpool', time_step=0):
    """
    Kwargs:
        - model 
        - ims in format tensor[batch_size, channels, im_h, im_w]
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
    """
    model.eval()

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        output = output.cpu().numpy()
        _model_feats.append(np.reshape(output, (len(output), -1)))

    try:
        m = model.module
    except:
        m = model

    model_layer = getattr(getattr(m, layer), sublayer)
    hook = model_layer.register_forward_hook(_store_feats)

    model_feats = []
    with torch.no_grad():
        _model_feats = []
        model(ims)
        model_feats.append(_model_feats[time_step])
        model_feats = np.concatenate(model_feats)

    hook.remove()    

    return model_feats



def get_cornet_activations(model, ims, output_path = None, layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    Suitable for small image sets. If you have thousands of images or it is
    taking too long to extract features, consider using
    `torchvision.datasets.ImageFolder`, using `ImageNetVal` as an example.

    Kwargs:
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
        - imsize (resize image to how many pixels, default: 224)
    """

    model.eval()

    model_feats = []
    with torch.no_grad():
        model_feats = []
        fnames = sorted(glob.glob(os.path.join(FLAGS.data_path, '*.*')))
        if len(fnames) == 0:
            raise FileNotFoundError(f'No files found in {FLAGS.data_path}')
        for fname in tqdm.tqdm(fnames):
            try:
                im = Image.open(fname).convert('RGB')
            except:
                raise FileNotFoundError(f'Unable to load {fname}')
            im = transform(im)
            im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
            _model_feats = []
            model(im)
            model_feats.append(_model_feats[time_step])
        model_feats = np.concatenate(model_feats)


    if output_path is not None:
        fname = f'CORnet_{layer}_{sublayer}_feats.npy'
        np.save(os.path.join(output_path, fname), model_feats)


def get_transformer_activations(model):

    # use lists to store the outputs via up-values
    inp_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = []
    hooks = [
        model.input_proj.register_forward_hook(
            lambda self, input, output: inp_features.append(output)
        ),
    ]

    # encoder tokens
    for i in range(args.encoder_layers):
        hooks.append(model.transformer.encoder.layers[-i].register_forward_hook(
                lambda self, input, output: enc_output.append(output)))
    # encoder attention
    for i in range(args.encoder_layers):
        hooks.append(model.transformer.encoder.layers[-i].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output)))
    #decoder tokens
    for i in range(args.decoder_layers):
        hooks.append(model.transformer.decoder.layers[-i].register_forward_hook(
                lambda self, input, output: dec_output.append(output[1])))
    #decoder attention 
    for i in range(args.decoder_layers):
        hooks.append(model.transformer.decoder.layers[-i].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])))

    # propagate through the model
    outputs = model(img)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    return 0


def get_activations(model, ):



    return 0

def get_activations_pretrained(model_name, layer_index, return_interm_layers):

    
    if 'resnet' in model_name: 
        model = resnet_model(model_name, False, return_interm_layers, dilation=0)

    elif 'cornet' in model_name:
        model = get_cornet_model(model_name[-1], pretrained=True) 

    elif model_name == 'dinov2':
        model = dino_model(-1*layer_index, return_interm_layers)

    elif model_name == 'dinov2_with_hooks':
        model = dino_model_with_hooks(-1*layer_index, return_interm_layers)





    

