import torch.utils.checkpoint as cp
from src.model.song.song_unet import SongUNet
from src.config import cfg
from src.model.song.song_helper_classes import *
#from torch.cuda.amp import autocast

class UNet(SongUNet):
    """
    Subclass of SongUNet that applies activation checkpointing
    to all UNetBlock instances.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_checkpoint = cfg.do_checkpointing

    def forward_block(self, block, x, emb=None):
        """
        Wrap UNetBlock in checkpoint.
        Mixed precision is handled externally in training loop.
        """
        if emb is None:
            func = lambda x: block(x)
        else:
            func = lambda x: block(x, emb)

        if self.use_checkpoint:
            return cp.checkpoint(func, x)
        else:
            return func(x)


    def forward_block3(self, block, x, emb=None):
        """
        Wrap UNetBlock in torch checkpoint with proper autocast support.
        """
        if emb is None:
            func = lambda x: block(x)
        else:
            func = lambda x: block(x, emb)

        if self.use_checkpoint:
            # Wrap in autocast to make sure recomputation uses same dtype
            return cp.checkpoint(lambda x: func(x), x)
        else:
            return func(x)


    def forward_block2(self, block, x, emb=None):
        """
        Wrap UNetBlock in torch checkpoint.
        """
        if emb is None:
            func = lambda x: block(x)
        else:
            func = lambda x: block(x, emb)

        if self.use_checkpoint:
            return cp.checkpoint(func, x)
        else:
            return func(x)

    def forward(self, x, noise_labels, conditioning=None, class_labels=None, augment_labels=None):
        # Replace all calls to block(x, emb) with self.forward_block(block, x, emb)
        # You can copy your original forward here and just call self.forward_block
        # for every UNetBlock in enc/dec.
        # Encoder
        skips = []
        aux = x
        emb = self._compute_embedding(noise_labels, class_labels, augment_labels)
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = self.forward_block(block, x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = self.forward_block(block, x, emb)

        return aux

    def _compute_embedding(self, noise_labels, class_labels=None, augment_labels=None):
        """
        Extract the embedding part from the original forward
        to keep the code clean.
        """
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([emb.shape[0], 1], device=emb.device) >= self.label_dropout).to(
                    tmp.dtype
                )
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))
        return emb
