import json
from tqdm import tqdm
import random

with open("train.json", "r") as json_file:
    dataset = json.load(json_file)
    tail_set = {
            'info': {}, 'licenses': {}, 'images': [], 'categories': dataset['categories'], 'annotations': []
    }
    images = dataset['images']
    annos = dataset['annotations']
    annos_bin = [[] for _ in range(13)]
    image_id_bin = [set() for _ in range(13)]

    print("divide images in bin")
    for anno in tqdm(annos):
        annos_bin[anno['category_id']-1].append(anno)

    print("random sort out")
    for i in range(13):
        annos_bin[i] = random.sample(annos_bin[i], 5000) if len(annos_bin[i]) > 5000 else annos_bin[i]
        for anno in annos_bin[i]:
            image_id_bin[anno['category_id']-1].add(anno['image_id'])
            tail_set['annotations'].append(anno)
        print(f'sorted out {i+1} category annos: {len(annos_bin[i])}, img: {len(image_id_bin[i])}')

    for image_cat in tqdm(image_id_bin):
        for img in images:
            if img['id'] in image_cat:
                tail_set['images'].append(img)

    print(f"filtered         images: {len(tail_set['images'])}, annos: {len(tail_set['annotations'])}")
    with open('train_uni.json', 'w') as tail_file:
        json.dump(tail_set, tail_file)

