from torchvision import transforms

# CLIP normalization constants (from the original CLIP paper)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

IMAGE_SIZE = 224


def get_train_transform() -> transforms.Compose:
    """Random crop + flip + colour jitter, then CLIP normalization."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """Bicubic resize + centre crop + CLIP normalization (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
