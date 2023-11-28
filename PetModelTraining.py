import torch
from multiprocessing.spawn import freeze_support
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchmetrics import Accuracy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
base_dir = 'dogImages'


mps_device = torch.device("cuda:0")
class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch],
                                           return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch],
                                           dtype=torch.long)
        return encodings
model_name_or_path = 'google/vit-base-patch16-224'
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

target_size = (224, 224)
transform = transforms.Resize(target_size)
transforms = transforms.Compose([transform, ToTensor()])

dataset = base_dir

train_ds = torchvision.datasets.ImageFolder(dataset + '/train/', transform=transforms)
valid_ds = torchvision.datasets.ImageFolder(dataset + '/valid/', transform=transforms)
test_ds = torchvision.datasets.ImageFolder(dataset + '/test/', transform=transforms)
label2id = {}
id2label = {}
for i, class_name in enumerate(train_ds.classes):
    label2id[class_name] = str(i)
    id2label[str(i)] = class_name

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
collator = ImageClassificationCollator(feature_extractor)
train_loader = DataLoader(train_ds, batch_size=128,
                          collate_fn=collator, num_workers=8, shuffle=True)
val_loader = DataLoader(valid_ds, batch_size=128, collate_fn=collator,
                        num_workers=8)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True)
model.to(mps_device)


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x[0] for x in batch]),
        'labels': torch.tensor([x[1] for x in batch])
    }
from datasets import load_metric

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
from transformers import TrainingArguments


if __name__ == '__main__':
    freeze_support()
    training_args = TrainingArguments(
        output_dir="./PetModel",
        per_device_train_batch_size=64,
        per_device_eval_batch_size = 64,
        evaluation_strategy="steps",
        num_train_epochs=6,
        fp16=True,
        save_steps=200,
        eval_steps=100,
        logging_steps=10,
        learning_rate=4e-4,
        save_total_limit=10,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        dataloader_num_workers=8,
        weight_decay = 0.01
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=feature_extractor,
    )
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    metrics = trainer.evaluate(test_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
import gc
gc.collect()

