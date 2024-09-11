import os
import time
import torch
from tqdm import tqdm
import numpy as np

import clip.clip as clip

import src.templates as templates

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing

import src.datasets as datasets

def get_zeroshot_classifier(args, clip_model):
    template = getattr(templates, args.template)
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(None, location=args.data_location, batch_size=args.batch_size, classnames=args.classnames)
    device = args.device
    clip_model.train()
    clip_model.to(device)
    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = [t(classname) for t in template]
            texts = clip.tokenize(texts).to(device)  # tokenize
            embeddings = clip_model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            zeroshot_weights.append(embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        zeroshot_weights *= clip_model.logit_scale.exp()
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
    return ClassificationHead(normalize=True, weights=zeroshot_weights)

def finetune(args):
    # if args.load is not None:
    #     image_classifier = ImageClassifier.load(args.load)
    # else:
    #     # Load a CLIP model instead
    #     image_encoder = ImageEncoder(args, keep_lang=True)
    #     classification_head = get_zeroshot_classifier(args, image_encoder.model)
    #     image_classifier = ImageClassifier(image_encoder, classification_head, process_images=True)



    image_encoder = ImageEncoder(args, keep_lang=True)
    classification_head = get_zeroshot_classifier(args, image_encoder.model)
    image_classifier = ImageClassifier(image_encoder, classification_head, process_images=True)


    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        print_every = 100

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(preprocess_fn, location=args.data_location, batch_size=args.batch_size)
    num_batches = len(dataset.train_loader)
    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    for epoch in range(args.epochs):
        model.train()
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
        for i, batch in enumerate(data_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            batch_time = time.time() - start_time
            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\tLoss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True)
        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            image_classifier = model.module
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
            print('Saving model to', model_path)
            image_classifier.save(model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            torch.save(optimizer.state_dict(), optim_path)
        args.current_epoch = epoch
        eval_results = evaluate(image_classifier, args)
    if args.save is not None:
        return model_path

if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)
