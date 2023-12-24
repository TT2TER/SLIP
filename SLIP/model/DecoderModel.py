import torch
from torch import nn


class TextDecoder(nn.Module):
    def __init__(self, vocab_size=30522, max_position_embeddings=512, 
                 embedding_dim=768, hidden_dim=2048, n_head = 8, n_layers=6, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(max_position_embeddings, embedding_dim)
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )
        decoder_lay = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=n_head, dim_feedforward=hidden_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_lay, num_layers=n_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, image_embed, tokens: torch.LongTensor) -> torch.Tensor:
        position_ids = self.position_ids[:,:tokens.size(1)]
        text_embed = self.word_embeddings(tokens) + self.pos_embed(position_ids)
        att_mask = torch.full((tokens.size(1), tokens.size(1)), float('-inf'))
        att_mask = torch.triu(att_mask, diagonal=1).to(tokens.device)
        key_padding_mask = torch.full((tokens.size(0), tokens.size(1)), False).to(tokens.device)
        key_padding_mask = key_padding_mask.masked_fill(tokens==0, True)
        # att_mask = att_mask.masked_fill(tokens==0, float(-100)).to(tokens.device)
        output = self.decoder(text_embed, image_embed, tgt_mask = att_mask, tgt_key_padding_mask=key_padding_mask)
        # print(att_mask)
        return self.fc(output)
    

if __name__ == "__main__":
    model = TextDecoder()
    x = torch.randn(2, 196, 768)
    tokens = torch.randint(0, 3, (2, 10))
    y = model(x, tokens)
    print(y.shape)