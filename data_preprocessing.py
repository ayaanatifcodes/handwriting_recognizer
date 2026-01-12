import cv2  # Importing OpenCV for image processing
import torch  # Importing PyTorch
import numpy as np  # Importing NumPy and using 'np' for convenience
from torchvision import transforms  # Importing transforms for data augmentation
from typing import Tuple  # Tuple is used for type hinting and immutability of size values

class Preprocessor:  # Class definition
    def __init__(  # Constructor (initialization method)
            self,  # Refers to the current instance of the class
            image_size: Tuple[int, int] = (244, 244),  # Target image size with a default value
            augmentation: bool = False,  # Boolean to enable or disable data augmentation
            vocab: str = ""  # Vocabulary string used for label encoding
    ):
        self.image_size = image_size  # Stores the image size
        self.augment = augmentation  # Stores whether augmentation is enabled
        self.vocab = vocab  # Stores the vocabulary
        
        self.affine_transform = transforms.Compose([  # Compose applies multiple transforms sequentially
            transforms.ToPILImage(),  # Converts NumPy or Torch image to PIL format (required for torchvision transforms)
            transforms.RandomAffine(  # Applies random affine transformations for data augmentation
                degrees=25,  # Allows random rotation between -25 and +25 degrees
                translate=(0.1, 0.1),  # Allows translation up to 10% in both x and y directions
                scale=(0.7, 1.1),  # Allows scaling between 70% and 110% of the original size
                shear=10  # Allows shearing (skewing) of the image
            )
        ])

        def __call__(self, img: np.ndarray, label: str, max_len: int = 32): # Allows the class object to be called like a function
        img, label = self.preprocess_img(img, label) # Preprocesses the image (resize and padding) and keeps the label unchanged
        # preprocess_img resizes the image to fit a fixed size while preserving its aspect ratio and pads the remaining space with white pixels
        label = self.label_indexer(self.vocab, label) # Converts each character in the label into its corresponding index using the vocabulary
        # label_indexer converts each character in a label into its corresponding index based on a given vocabulary
        label = self.label_padding(len(self.vocab), max_len, label) # Pads the label to a fixed length using the vocabulary size as the padding value
        # label_padding truncates or pads a label so that all labels have a fixed length (vocab size used for padding)
        
        if self.augment: # Checks if data augmentation is enabled
            img = self.apply_augmentation(img) # Applies random augmentations to the image
        return img, label # Returns the processed image and the processed label

    def preprocess_img(self, img: np.ndarray, text: str):
        target_w, target_h = self.image_size
        h, w = img.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = cv2.resize(img, (new_w, new_h))

        pad_w = target_w - new_w
        pad_h = target_h - new_h

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=255
        )

        return img, text

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            img = self.random_sharpen(img)

        if np.random.rand() < 0.5:
            img = np.array(self.affine_transform(img))

        return img

    @staticmethod
    def random_sharpen(
        image: np.ndarray,
        alpha_range: Tuple[float, float] = (0.25, 1.0),
        lightness_range: Tuple[float, float] = (0.75, 2.0),
    ) -> np.ndarray:
        alpha = np.random.uniform(*alpha_range)
        lightness = np.random.uniform(*lightness_range)

        kernel_identity = np.array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
            dtype=np.float32
        )

        kernel_sharpen = np.array(
            [[-1, -1, -1],
             [-1,  8 + lightness, -1],
             [-1, -1, -1]],
            dtype=np.float32
        )

        kernel = (1 - alpha) * kernel_identity + alpha * kernel_sharpen
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def label_indexer(vocab: str, label: str) -> np.ndarray:
        return np.array([vocab.index(c) for c in label if c in vocab])

    @staticmethod
    def label_padding(padding_value: int, max_len: int, label: np.ndarray) -> np.ndarray:
        label = label[:max_len]
        return np.pad(
            label,
            (0, max_len - len(label)),
            mode="constant",
            constant_values=padding_value
        )

    def single_image_preprocessing(self, img: np.ndarray) -> torch.Tensor:
        img, _ = self.preprocess_img(img, "")
        img = torch.from_numpy(img).float()
        img = img / 255.0
        return img


