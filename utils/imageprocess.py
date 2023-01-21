# imageprocess.py

class RotateTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        if isinstance(self.angles, list):
            angle = random.choice(self.angles)
        else:
            angle = self.angles
        return TF.rotate(x, angle)


def image_transformer(input_image=None, train=True):
    """
    Using torchvision.transforms, make PIL image to tensor image
    with normalizing and flipping augmentations
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    if train:
        transformer = transforms.Compose([        
            RotateTransform([0, 0, 0, -90, 90, 180]),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
            #transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    transformed_image = transformer(input_image)
    
    return transformed_image



def tta_transformer(input_image, angle):
    """
    Test Time Augmentation for creating final test labels.
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transformer = transforms.Compose([        
        RotateTransform(angle),
        transforms.ToTensor(),
        normalize,
    ])

    transformed_image = transformer(input_image)
    
    return transformed_image
