import clip
from eval_on_attack.utils import eval
from datasets.ImageNet100 import ImageNet100

device = 'cpu'


def main():

    model, preprocess = clip.load("ViT-B/32", device=device)
    data_test = ImageNet100(root='/d/data/imagenet62',split='test',preprocess=preprocess)
    classes = data_test.classes

    templates = [''] * 4
    templates[0] = 'a photo of a {}.'
    templates[1] = 'a photo of a word written over a picture of a {}.'
    templates[2] = 'a photo of the word {X} written over a picture of a {Y}.'
    templates[3] = 'a photo of a nonsense word written over a picture of a {}.'

    eval(model, data_test, classes, templates)




if __name__ == '__main__':
    main()
