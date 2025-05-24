import clip
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from datasets.imagenet100_origin import ImageNet100Origin
from datasets.imagenet100_irr import ImageNet100Irr

device = 'cpu'
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.requires_grad:
            p.grad.data = p.grad.data.float()


def train( model,Typo,Origin):


    ImageNet100Typo = DataLoader(Typo, batch_size=1024, shuffle=True, num_workers=2)
    ImageNet100Origin = DataLoader(Origin, batch_size=1024, shuffle=True, num_workers=2)
    batch_size = 1024*2
    num_epochs = 5
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.visual.transformer.resblocks[-1].parameters():
        param.requires_grad = True

    model.visual.ln_post.weight.requires_grad=True
    model.visual.ln_post.bias.requires_grad=True


    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for i,(data, data_typo) in enumerate(zip(ImageNet100Origin,ImageNet100Typo)):
            image, label, catogery,*_ = data
            img = image.to(device)
            typographic_image, label, catogery_typo,*_ = data_typo
            typo_img =  typographic_image.to(device)

            tensors = [img,typo_img]
            img = torch.cat(tensors, dim=0)
            actual_batch_size = label.size(0)
            if actual_batch_size*2 < batch_size:
                break

            text1 = ["a photo of " + catogery[i] for i in range(len(catogery))]
            text3 = ["a photo of " + catogery_typo[i] for i in range(len(catogery_typo))]


            lists = [text1,text3]
            text = []
            [text.extend(lst) for lst in lists]


            text = clip.tokenize(text).to(device)
            logits_per_image, logits_per_text = model(img, text)

            if device == "cpu":
                ground_truth = torch.arange(batch_size).long().to(device)
            else:
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            optimizer.zero_grad()
            total_loss.backward()
            if device == "cpu":
                optimizer.step()

            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            print('[Epoch %d, Batch %d] loss: %.3f' % (epoch + 1, i + 1, total_loss))


    return model

def main():

    model, preprocess = clip.load("ViT-B/32", device=device)
    Origin = ImageNet100Origin(root='/d/data/imagenet62',split='train',preprocess=preprocess)
    Typo = ImageNet100Irr(root='/d/data/imagenet62', split='train', preprocess=preprocess)


    train(model,Typo,Origin)





if __name__ == '__main__':
    main()