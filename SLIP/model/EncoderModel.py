import torch.nn as nn
import torch
import torchvision


class VisionEncoder(nn.Module):
    def __init__(self, model = "vit_b_16") -> None:
        super().__init__()
        if model == "vit_b_16":
            self.vision_transformer = torchvision.models.vit_b_16(
                weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif model == "vit_b_32":
            self.vision_transformer = torchvision.models.vit_b_32(
                weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1)
        elif model == "vit_l_16":
            self.vision_transformer = torchvision.models.vit_l_16(
                weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1)
        elif model == "vit_l_32":
            self.vision_transformer = torchvision.models.vit_l_32(
                weights=torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1)
        elif model == "vit_h_14":
            self.vision_transformer = torchvision.models.vit_h_14(
                weights=torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        else:
            raise ValueError("model must be one of vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14")

    def forward(self, x):
        x = self.vision_transformer._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vision_transformer.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vision_transformer.encoder(x)
        f = x[:, 1:]
        return f
    

if __name__ == "__main__":
    model = VisionEncoder()
    x = torch.randn(2, 3, 224, 224)
    f = model(x)
    print(f.shape)