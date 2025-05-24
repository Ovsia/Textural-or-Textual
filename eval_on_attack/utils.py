

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
device = 'cpu'
def eval_on_dataset_nonsense_easy( model,test_dataset,classes,templates):

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    text_inputs = torch.cat([clip.tokenize(templates[0].format(c)) for c in classes]).to(device)
    text_inputs = text_inputs.to(device)
    model.eval()

    acc_baseline, total = 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for _, _, _, nonsense_img, _, target, *_ in test_loader:
            #img, irr_img, cons_img, nons_img, small_irr_img, target, right_class, image_description, img_id,attack_text,attack_label
            target = target.to(device)
            image = nonsense_img.to(device)
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]
            acc_baseline += probs_baseline.eq(target).sum().item()
            total += target.size(0)

    return acc_baseline / total

def eval_on_dataset_nonsense_med( model,test_dataset,classes,templates):

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    text_inputs = torch.cat([clip.tokenize(templates[1].format(c)) for c in classes]).to(device)
    text_inputs = text_inputs.to(device)
    model.eval()

    acc_baseline, total = 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for _, _, _, nonsense_img, _, target, *_ in test_loader:
            target = target.to(device)
            image = nonsense_img.to(device)
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]
            acc_baseline += probs_baseline.eq(target).sum().item()
            total += target.size(0)

    return acc_baseline / total

def eval_on_dataset_nonsense_hard( model,test_dataset,classes,templates):

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    text_inputs = torch.cat([clip.tokenize(templates[3].format(c)) for c in classes]).to(device)
    text_inputs = text_inputs.to(device)
    model.eval()

    acc_baseline, total = 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for _, _, _, nonsense_img, _, target,  *_ in test_loader:
            target = target.to(device)
            image = nonsense_img.to(device)
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]
            acc_baseline += probs_baseline.eq(target).sum().item()
            total += target.size(0)

    return acc_baseline / total


def eval_on_dataset_cons_easy( model, test_dataset,classes,templates):

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    text_inputs = torch.cat([clip.tokenize(templates[0].format(c)) for c in classes]).to(device)
    text_inputs = text_inputs.to(device)
    model.eval()

    acc_baseline, total = 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for _, _, cons_img, _, _, target,  *_ in test_loader:
            target = target.to(device)
            image = cons_img.to(device)
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]
            acc_baseline += probs_baseline.eq(target).sum().item()
            total += target.size(0)

    return acc_baseline / total

def eval_on_dataset_cons_med( model, test_dataset,classes,templates):

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    text_inputs = torch.cat([clip.tokenize(templates[1].format(c)) for c in classes]).to(device)
    text_inputs = text_inputs.to(device)
    model.eval()

    acc_baseline, total = 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for _, _, cons_img, _, _, target,  *_ in test_loader:
            target = target.to(device)
            image = cons_img.to(device)
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]
            acc_baseline += probs_baseline.eq(target).sum().item()
            total += target.size(0)

    return acc_baseline / total

def eval_on_dataset_cons_hard( model, test_dataset,classes,templates):

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    text_inputs = []
    lenth = len(classes)
    for x_class in classes:
        for y_class in classes:
            X = x_class
            Y = y_class
            formatted_text = templates[2].format(X=X, Y=Y)
            tokenized_text = clip.tokenize(formatted_text)
            text_inputs.append(tokenized_text)

    text_inputs = torch.cat(text_inputs).to(device)

    model.eval()

    acc_baseline, total = 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for _, _, cons_img,  _, _, target,  *_ in test_loader:
            target = target.to(device)
            image = cons_img.to(device)
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]

            acc_baseline += probs_baseline.eq(target*lenth+target).sum().item()
            total += target.size(0)

    return acc_baseline / total

def eval_on_dataset_irr_easy(model,test_dataset,classes,templates):

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    text_inputs = torch.cat([clip.tokenize(templates[0].format(c)) for c in classes]).to(device)
    text_inputs = text_inputs.to(device)
    model.eval()

    acc_baseline, total = 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for _, irr_img, _, _, _, target,  *_ in test_loader:
            target = target.to(device)
            image = irr_img.to(device)
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]
            acc_baseline += probs_baseline.eq(target).sum().item()
            total += target.size(0)

    return acc_baseline / total

def eval_on_dataset_irr_med(model,test_dataset,classes,templates):

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    text_inputs = torch.cat([clip.tokenize(templates[1].format(c)) for c in classes]).to(device)
    text_inputs = text_inputs.to(device)
    model.eval()

    acc_baseline, total = 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for _, irr_img, _, _, _, target,  *_ in test_loader:
            target = target.to(device)
            image = irr_img.to(device)
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]
            acc_baseline += probs_baseline.eq(target).sum().item()
            total += target.size(0)

    return acc_baseline / total

def eval_on_dataset_irr_hard(model,test_dataset,classes,templates):

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    text_inputs = []
    length = len(classes)
    for x_class in classes:
        for y_class in classes:
            X = x_class
            Y = y_class
            formatted_text = templates[2].format(X=X, Y=Y)
            tokenized_text = clip.tokenize(formatted_text)
            text_inputs.append(tokenized_text)


    text_inputs = torch.cat(text_inputs).to(device)
    model.eval()
    acc_baseline, total = 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for _, irr_img,  _, _, _, target, right_class, _, _, _,attack_label in test_loader:
            target = target.to(device)

            right_class = list(right_class)
            mapped_numbers = attack_label
            mapped_numbers = torch.tensor(mapped_numbers).to(device)
            image = irr_img.to(device)
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]
            acc_baseline += probs_baseline.eq(mapped_numbers*length+target).sum().item()
            total += target.size(0)

    return acc_baseline / total


def eval_on_dataset_origin( model,test_dataset,classes,templates):

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    text_inputs = torch.cat([clip.tokenize(templates[0].format(c)) for c in classes]).to(device)
    text_inputs = text_inputs.to(device)
    model.eval()

    acc_baseline, total = 0, 0
    with torch.no_grad():
        text_feature = model.encode_text(text_inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

        for img,  _, _, _, _, target, *_ in test_loader:

            image = img.to(device)
            target = target.to(device)
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            similarity_baseline = 100. * (img_features @ text_feature.T)
            probs_baseline = F.softmax(similarity_baseline, dim=-1).max(-1)[1]
            acc_baseline += probs_baseline.eq(target).sum().item()
            total += target.size(0)

    return acc_baseline / total

def eval(model, test_dataset, classes, templates):
    eval_cases = [
        ("Origin Dataset", eval_on_dataset_origin),
        ("Cons easy", eval_on_dataset_cons_easy),
        ("Irr easy", eval_on_dataset_irr_easy),
        ("nonsense easy", eval_on_dataset_nonsense_easy),
        ("Cons med", eval_on_dataset_cons_med),
        ("Irr med", eval_on_dataset_irr_med),
        ("nonsense med", eval_on_dataset_nonsense_med),
        ("Cons hard", eval_on_dataset_cons_hard),
        ("Irr hard", eval_on_dataset_irr_hard),
        ("nonsense hard", eval_on_dataset_nonsense_hard),
    ]

    for name, func in eval_cases:
        acc = func(model, test_dataset, classes, templates)
        print(f"{name}\n{acc * 100:.2f}%")


# mapping_attack=  {'ice cream': 'backpack', 'ruler': 'toaster', 'refrigerator': 'vase', 'table lamp': 'iPod', 'sock': 'table lamp', 'coffee mug': 'sunglass', 'hotdog': 'tabby cat', 'nail': 'violin', 'broom': 'bagel', 'mailbox': 'lighter', 'peacock': 'barrel', 'bottlecap': 'Granny Smith', 'toaster': 'microwave', 'pillow': 'iron', 'espresso': 'bow', 'panda': 'goose', 'bee': 'banana', 'koala': 'flamingo', 'llama': 'cucumber', 'umbrella': 'crib', 'tray': 'umbrella', 'red wine': 'refrigerator', 'jean': 'tub', 'basketball': 'koala', 'suit': 'tray', 'iron': 'wallet', 'switch': 'snail', 'cucumber': 'swing', 'dough': 'broom', 'plastic bag': 'espresso', 'tub': 'nail', 'stove': 'fountain', 'orange': 'shovel', 'shovel': 'mouse', 'pizza': 'projector', 'lighter': 'television', 'television': 'ruler', 'candle': 'tiger', 'carton': 'basketball', 'bucket': 'lotion', 'printer': 'daisy', 'cannon': 'perfume', 'radio': 'jean', 'wallet': 'panda', 'lion': 'paddle', 'tabby cat': 'ice cream', 'teapot': 'bee', 'cup': 'sock', 'potpie': 'lipstick', 'Chihuahua': 'pizza', 'pajama': 'necklace', 'banana': 'laptop', 'jellyfish': 'mailbox', 'violin': 'Chihuahua', 'daisy': 'jellyfish', 'perfume': 'purse', 'laptop': 'bucket', 'zebra': 'suit', 'dragonfly': 'radio', 'swing': 'pajama', 'mask': 'hotdog', 'whistle': 'printer', 'Granny Smith': 'cup', 'bikini': 'orange', 'strawberry': 'pillow', 'paddle': 'coffeepot', 'cardigan': 'whistle', 'coffeepot': 'stove', 'lipstick': 'peacock', 'volcano': 'corn', 'buckle': 'red wine', 'sunglass': 'candle', 'microwave': 'dough', 'corn': 'cardigan', 'mushroom': 'lion', 'vacuum': 'ladle', 'tiger': 'sea lion', 'goose': 'zebra', 'envelope': 'plastic bag', 'snail': 'volcano', 'crib': 'mitten', 'mitten': 'hay', 'purse': 'switch', 'bagel': 'mask', 'backpack': 'cannon', 'vase': 'pig', 'barrel': 'bottlecap', 'iPod': 'dragonfly', 'mouse': 'envelope', 'hay': 'potpie', 'pig': 'buckle', 'lotion': 'coffee mug', 'ladle': 'teapot', 'projector': 'strawberry', 'bow': 'lemon', 'necklace': 'llama', 'flamingo': 'bikini', 'fountain': 'carton', 'sea lion': 'mushroom', 'lemon': 'vacuum'}
# mapping_nonsense = {'peacock': 'agdild', 'goose': 'TkKqIU', 'koala': 'FOnWBI', 'jellyfish': 'OfzmMX', 'snail': 'hGhiZU', 'flamingo': 'dwWasL', 'sea lion': 'QXFGmA', 'Chihuahua': 'HjFXWw', 'tabby cat': 'TGbDkP', 'lion': 'idvsNI', 'tiger': 'zxzkBW', 'bee': 'VXvImW', 'dragonfly': 'ypggll', 'zebra': 'fdPxMz', 'pig': 'NLrEBE', 'llama': 'SXLzyf', 'panda': 'QfETHk', 'backpack': 'VlmsXG', 'barrel': 'IUXrcM', 'basketball': 'HtlHgE', 'bikini': 'RDfbkd', 'bottlecap': 'mBwTSU', 'bow': 'GmohEO', 'broom': 'GdjbIu', 'bucket': 'CvPwig', 'buckle': 'hBLPtI', 'candle': 'AWEmxZ', 'cannon': 'lzfLXt', 'cardigan': 'ivzGIP', 'carton': 'fQpINT', 'coffee mug': 'ZwQiNz', 'coffeepot': 'rzbUSl', 'crib': 'QzczHl', 'envelope': 'DwaVRX', 'fountain': 'QDAkgE', 'iPod': 'ptyCiS', 'iron': 'tgdhbi', 'jean': 'oxSAXK', 'ladle': 'ShrTNv', 'laptop': 'guSZiF', 'lighter': 'xrvmqc', 'lipstick': 'XhTQjF', 'lotion': 'hlolrh', 'mailbox': 'qLjvkR', 'mask': 'ykkqmr', 'microwave': 'gNwsrO', 'mitten': 'cVhNlT', 'mouse': 'DdrqpK', 'nail': 'WNiceg', 'necklace': 'GyhaiT', 'paddle': 'pjxFOg', 'pajama': 'XLyFMq', 'perfume': 'UBPIwO', 'pillow': 'yNYKcH', 'plastic bag': 'ELohPt', 'printer': 'yVfweC', 'projector': 'eFnels', 'purse': 'aHdlYO', 'radio': 'nCpyKX', 'refrigerator': 'RcoVuR', 'ruler': 'nEMeGH', 'shovel': 'pONwiK', 'sock': 'elzExJ', 'stove': 'bkRGqU', 'suit': 'Rqqftk', 'sunglass': 'DlshGy', 'swing': 'THYNDR', 'switch': 'FpmVhE', 'table lamp': 'XTGvru', 'teapot': 'PpEQyN', 'television': 'CvYWYV', 'toaster': 'MfeBdo', 'tray': 'tNYJPu', 'tub': 'zHeLFp', 'umbrella': 'bvJlvm', 'vacuum': 'LtnxDP', 'vase': 'ZLJmnt', 'violin': 'AYiMDl', 'wallet': 'ZIpsms', 'whistle': 'CCslxq', 'ice cream': 'QYVNvm', 'bagel': 'IrFqpr', 'hotdog': 'lFlYHK', 'cucumber': 'gjtzpn', 'mushroom': 'AIkRvs', 'Granny Smith': 'kNmbVO', 'strawberry': 'KxwPgS', 'orange': 'HGFWGy', 'lemon': 'MxlRgR', 'banana': 'ouTPgu', 'hay': 'unhMga', 'dough': 'SZXoXK', 'pizza': 'vvaNwC', 'potpie': 'KjCfuA', 'red wine': 'ipolnW', 'espresso': 'EcOqCG', 'cup': 'GYmFlq', 'volcano': 'HQmTMp', 'daisy': 'CwuIBi', 'corn': 'poHbme'}


