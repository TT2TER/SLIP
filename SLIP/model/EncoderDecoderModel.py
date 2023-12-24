import torch
from torch import nn
from .EncoderModel import VisionEncoder
from .DecoderModel import TextDecoder


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size=30522, max_position_embeddings=512, 
                 embedding_dim=768, hidden_dim=2048, n_head = 8, n_layers=6, dropout=0.1, encoder_model = "vit_b_16"):
        super().__init__()
        self.encoder = VisionEncoder(model=encoder_model)
        if encoder_model == "vit_b_16":
            embedding_dim = 768
        elif encoder_model == "vit_b_32":
            embedding_dim = 768
        elif encoder_model == "vit_l_16":
            embedding_dim = 1024
        elif encoder_model == "vit_l_32":
            embedding_dim = 1024
        elif encoder_model == "vit_h_14":
            embedding_dim = 1280
        self.decoder = TextDecoder(vocab_size, max_position_embeddings, 
                                   embedding_dim, hidden_dim, n_head, n_layers, dropout)

    def forward(self, image, tokens):
        # image encoder
        image_embed = self.encoder(image)

        # text decoder
        output = self.decoder(image_embed, tokens)

        return output
    

if __name__ == "__main__":
    model = EncoderDecoder()
    x = torch.randn(2, 3, 224, 224)
    tokens = torch.randint(0, 30522, (2, 10))
    y = model(x, tokens)
    print(y.shape)