import os
import random
import string
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from string import Template

color = ["White", "Blue", "Green", "Red", "Magenta", "Cyan", "Yellow", "Black"]
sentence_templates = [
    Template("This is a $text."),
    Template("A snapshot of $text."),
    Template("An image of $text."),
    Template("A picture of $text."),
    Template("This image showcases $text.")
]
font = ["Roman2.ttf", "courier.ttf", "times.ttf"]
random.seed(300)

def _transform(image):
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC), #type: ignore
        transforms.CenterCrop(224),
    ])
    return transform(image)

def create_image(image, text, font_path, fill, stroke):
    if image.mode != "RGB":
        image = image.convert("RGB")

    W, H = image.size
    draw = ImageDraw.Draw(image)
    if 2 <= len(text.split(' ')) <= 3:
        text = text.split(' ')
        text = [text[0], ' '.join(text[1:])]
        text = '\n'.join(text)
    elif len(text.split(' ')) > 3:
        text = text.split(' ')
        text = [' '.join(text[:2]), ' '.join(text[2:])]
        text = '\n'.join(text)

    font, txpos, _ = adjust_font_size((W, H), draw, text, font_path)
    draw.text(txpos, text, font=font, fill=fill, stroke_fill=stroke, stroke_width=1)
    return image

def adjust_font_size(img_size, imagedraw, text, font_path):
    W, H = img_size
    font_size = 80
    font = ImageFont.truetype(font_path, font_size)
    
    step = 1
    w, h = imagedraw.textsize(text, font=font)
    while w >= W or h >= H:
        font_size -= step
        font = ImageFont.truetype(font_path, font_size)
        w, h = imagedraw.textsize(text, font=font)

    txpos = ((W - w) * random.random(), (H - h) * random.random())

    return font, txpos, (w, h)



def make_image_text(file, classes, img_dir, target_dir, idx, font_path="/s/caltech101/Fonts"):

    img = Image.open(img_dir / file)
    text = classes
    font_path = os.path.join(font_path, random.choice(font))
    fill, stroke = random.choice(color), random.choice(color)
    while fill == stroke:
        stroke = random.choice(color)

    img = create_image(img, text, font_path, fill, stroke)
    dir = target_dir / "/".join(str(file).split("/")[:-1])

    os.makedirs(dir, exist_ok=True)
    img.save(target_dir / file, quality=100)


def make_image_text1(file, classes, img_dir, target_dir, label, font_path="/s/caltech101/Fonts"):
    sentence_templates = [
        Template("This is a $text."),
        Template("A snapshot of $text."),
        Template("An image of $text."),
        Template("A picture of $text."),
        Template("This image showcases $text.")
    ]
    mapping ={'ice cream': 'backpack', 'ruler': 'toaster', 'refrigerator': 'vase', 'table lamp': 'iPod', 'sock': 'table lamp', 'coffee mug': 'sunglass', 'hotdog': 'tabby cat', 'nail': 'violin', 'broom': 'bagel', 'mailbox': 'lighter', 'peacock': 'barrel', 'bottlecap': 'Granny Smith', 'toaster': 'microwave', 'pillow': 'iron', 'espresso': 'bow', 'panda': 'goose', 'bee': 'banana', 'koala': 'flamingo', 'llama': 'cucumber', 'umbrella': 'crib', 'tray': 'umbrella', 'red wine': 'refrigerator', 'jean': 'tub', 'basketball': 'koala', 'suit': 'tray', 'iron': 'wallet', 'switch': 'snail', 'cucumber': 'swing', 'dough': 'broom', 'plastic bag': 'espresso', 'tub': 'nail', 'stove': 'fountain', 'orange': 'shovel', 'shovel': 'mouse', 'pizza': 'projector', 'lighter': 'television', 'television': 'ruler', 'candle': 'tiger', 'carton': 'basketball', 'bucket': 'lotion', 'printer': 'daisy', 'cannon': 'perfume', 'radio': 'jean', 'wallet': 'panda', 'lion': 'paddle', 'tabby cat': 'ice cream', 'teapot': 'bee', 'cup': 'sock', 'potpie': 'lipstick', 'Chihuahua': 'pizza', 'pajama': 'necklace', 'banana': 'laptop', 'jellyfish': 'mailbox', 'violin': 'Chihuahua', 'daisy': 'jellyfish', 'perfume': 'purse', 'laptop': 'bucket', 'zebra': 'suit', 'dragonfly': 'radio', 'swing': 'pajama', 'mask': 'hotdog', 'whistle': 'printer', 'Granny Smith': 'cup', 'bikini': 'orange', 'strawberry': 'pillow', 'paddle': 'coffeepot', 'cardigan': 'whistle', 'coffeepot': 'stove', 'lipstick': 'peacock', 'volcano': 'corn', 'buckle': 'red wine', 'sunglass': 'candle', 'microwave': 'dough', 'corn': 'cardigan', 'mushroom': 'lion', 'vacuum': 'ladle', 'tiger': 'sea lion', 'goose': 'zebra', 'envelope': 'plastic bag', 'snail': 'volcano', 'crib': 'mitten', 'mitten': 'hay', 'purse': 'switch', 'bagel': 'mask', 'backpack': 'cannon', 'vase': 'pig', 'barrel': 'bottlecap', 'iPod': 'dragonfly', 'mouse': 'envelope', 'hay': 'potpie', 'pig': 'buckle', 'lotion': 'coffee mug', 'ladle': 'teapot', 'projector': 'strawberry', 'bow': 'lemon', 'necklace': 'llama', 'flamingo': 'bikini', 'fountain': 'carton', 'sea lion': 'mushroom', 'lemon': 'vacuum'}

    img = Image.open(img_dir / file)
    text = random.choice(classes)

    while text == label:
        text = random.choice(classes)
        print("repeat, choose again")
    sentence = text
    font_path = os.path.join(font_path, random.choice(font))
    fill, stroke = random.choice(color), random.choice(color)
    while fill == stroke:
        stroke = random.choice(color)
    img = create_image(img, sentence, font_path, fill, stroke)
    dir = target_dir / "/".join(str(file).split("/")[:-1])
    os.makedirs(dir, exist_ok=True)
    img.save(target_dir / file, quality=100)

    return text

def make_image_text2(file, classes, img_dir, target_dir, idx, font_path="/s/caltech101/Fonts"):
    img = Image.open(img_dir / file)

    text = classes 
    font_path = os.path.join(font_path, random.choice(font))
    fill, stroke = random.choice(color), random.choice(color)
    while fill == stroke:
        stroke = random.choice(color)
    img = create_image(img, text, font_path, fill, stroke)
    dir = target_dir / "/".join(str(file).split("/")[:-1])
    os.makedirs(dir, exist_ok=True)
    img.save(target_dir / file, quality=100)


def make_image_text3(file, classes, img_dir, target_dir, idx, font_path="/s/caltech101/Fonts"):

    img = Image.open(img_dir / file)
    #
    # mapping = {'peacock': 'agdild', 'goose': 'TkKqIU', 'koala': 'FOnWBI', 'jellyfish': 'OfzmMX', 'snail': 'hGhiZU', 'flamingo': 'dwWasL', 'sea lion': 'QXFGmA', 'Chihuahua': 'HjFXWw', 'tabby cat': 'TGbDkP', 'lion': 'idvsNI', 'tiger': 'zxzkBW', 'bee': 'VXvImW', 'dragonfly': 'ypggll', 'zebra': 'fdPxMz', 'pig': 'NLrEBE', 'llama': 'SXLzyf', 'panda': 'QfETHk', 'backpack': 'VlmsXG', 'barrel': 'IUXrcM', 'basketball': 'HtlHgE', 'bikini': 'RDfbkd', 'bottlecap': 'mBwTSU', 'bow': 'GmohEO', 'broom': 'GdjbIu', 'bucket': 'CvPwig', 'buckle': 'hBLPtI', 'candle': 'AWEmxZ', 'cannon': 'lzfLXt', 'cardigan': 'ivzGIP', 'carton': 'fQpINT', 'coffee mug': 'ZwQiNz', 'coffeepot': 'rzbUSl', 'crib': 'QzczHl', 'envelope': 'DwaVRX', 'fountain': 'QDAkgE', 'iPod': 'ptyCiS', 'iron': 'tgdhbi', 'jean': 'oxSAXK', 'ladle': 'ShrTNv', 'laptop': 'guSZiF', 'lighter': 'xrvmqc', 'lipstick': 'XhTQjF', 'lotion': 'hlolrh', 'mailbox': 'qLjvkR', 'mask': 'ykkqmr', 'microwave': 'gNwsrO', 'mitten': 'cVhNlT', 'mouse': 'DdrqpK', 'nail': 'WNiceg', 'necklace': 'GyhaiT', 'paddle': 'pjxFOg', 'pajama': 'XLyFMq', 'perfume': 'UBPIwO', 'pillow': 'yNYKcH', 'plastic bag': 'ELohPt', 'printer': 'yVfweC', 'projector': 'eFnels', 'purse': 'aHdlYO', 'radio': 'nCpyKX', 'refrigerator': 'RcoVuR', 'ruler': 'nEMeGH', 'shovel': 'pONwiK', 'sock': 'elzExJ', 'stove': 'bkRGqU', 'suit': 'Rqqftk', 'sunglass': 'DlshGy', 'swing': 'THYNDR', 'switch': 'FpmVhE', 'table lamp': 'XTGvru', 'teapot': 'PpEQyN', 'television': 'CvYWYV', 'toaster': 'MfeBdo', 'tray': 'tNYJPu', 'tub': 'zHeLFp', 'umbrella': 'bvJlvm', 'vacuum': 'LtnxDP', 'vase': 'ZLJmnt', 'violin': 'AYiMDl', 'wallet': 'ZIpsms', 'whistle': 'CCslxq', 'ice cream': 'QYVNvm', 'bagel': 'IrFqpr', 'hotdog': 'lFlYHK', 'cucumber': 'gjtzpn', 'mushroom': 'AIkRvs', 'Granny Smith': 'kNmbVO', 'strawberry': 'KxwPgS', 'orange': 'HGFWGy', 'lemon': 'MxlRgR', 'banana': 'ouTPgu', 'hay': 'unhMga', 'dough': 'SZXoXK', 'pizza': 'vvaNwC', 'potpie': 'KjCfuA', 'red wine': 'ipolnW', 'espresso': 'EcOqCG', 'cup': 'GYmFlq', 'volcano': 'HQmTMp', 'daisy': 'CwuIBi', 'corn': 'poHbme'}


    text = ''.join(random.choices(string.ascii_letters, k=6))
    font_path = os.path.join(font_path, random.choice(font))
    fill, stroke = random.choice(color), random.choice(color)
    while fill == stroke:
        stroke = random.choice(color)
    img = create_image(img, text, font_path, fill, stroke)
    dir = target_dir / "/".join(str(file).split("/")[:-1])
    os.makedirs(dir, exist_ok=True)
    img.save(target_dir / file, quality=100)

if __name__ == "__main__":
    classes = ["apple"]
    make_image_text('sample.jpg', classes, '.', 'results', 0, font_path='../font/AdobeVFPrototype.ttf')
