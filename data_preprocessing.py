import cv2  # OpenCV for image resizing, padding, and filtering
import torch  # PyTorch for tensor operations
from torchvision import transforms  # Torchvision for data augmentation
import numpy as np  # NumPy for numerical and array operations
from typing import Tuple, Dict  # Type hints for clarity


class Preprocessor:  # Define preprocessing class

    def __init__(self, image_size: Tuple[int, int] = (224, 224), augmentation: bool = False, vocab: str = ""):  # Constructor
        self.image_size = image_size  # Store target image size
        self.augment = augmentation  # Store augmentation flag
        self.vocab = vocab  # Store vocabulary string
        self.vocab_dict: Dict[str, int] = {c: i for i, c in enumerate(vocab)}  # Create fast char→index mapping

        self.affine_transform = transforms.Compose([  # Combine multiple image transforms
            transforms.ToPILImage(),  # Convert NumPy image to PIL image
            transforms.RandomAffine(  # Apply random affine transformation
                degrees=25,  # Random rotation angle
                translate=(0.1, 0.1),  # Random translation
                shear=10,  # Random shear
                scale=(0.8, 1.2)  # Random scaling
            )
        ])

    def __call__(self, img: np.ndarray, label: str, max_len: int = 35):  # Make object callable
        img, label = self.preprocess_img(img, label)  # Resize and pad image

        label = label.lower().strip()  # Normalize label text
        label = self.label_indexer(self.vocab_dict, label)  # Convert label to indices
        label = self.label_padding(len(self.vocab), max_len, label)  # Pad label to fixed length

        if self.augment:  # Check if augmentation is enabled
            img = self.apply_augmentation(img)  # Apply augmentation

        img = torch.from_numpy(img)  # Convert NumPy array to PyTorch tensor
        img = img.permute(2, 0, 1)  # Change format from HWC to CHW
        img = img.float() / 255.0  # Normalize pixel values to [0, 1]

        return img, label  # Return processed image and label

    def preprocess_img(self, img: np.ndarray, text: str):  # Resize and pad image
        target_w, target_h = self.image_size  # Extract target dimensions
        h, w = img.shape[:2]  # Get original image height and width
        scale = min(target_w / w, target_h / h)  # Compute scale while preserving aspect ratio
        new_w, new_h = int(w * scale), int(h * scale)  # Compute scaled dimensions
        img = cv2.resize(img, (new_w, new_h))  # Resize image

        pad_w = target_w - new_w  # Compute horizontal padding
        pad_h = target_h - new_h  # Compute vertical padding

        left = pad_w // 2  # Left padding
        right = pad_w - left  # Right padding
        top = pad_h // 2  # Top padding
        bottom = pad_h - top  # Bottom padding

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)  # Add white padding

        return img, text  # Return padded image and text

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:  # Apply random augmentations
        if np.random.rand() < 0.5:  # 50% chance
            img = self.random_sharpen(img)  # Apply sharpening

        if np.random.rand() < 0.5:  # 50% chance
            img = np.array(self.affine_transform(img))  # Apply affine transform

        return img  # Return augmented image

    @staticmethod
    def random_sharpen(image: np.ndarray, alpha_range: Tuple[float, float] = (0.25, 1.0), lightness_range: Tuple[float, float] = (0.75, 2.0)) -> np.ndarray:
        alpha = np.random.uniform(*alpha_range)  # Random sharpening strength
        lightness = np.random.uniform(*lightness_range)  # Random brightness factor

        kernel_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)  # Identity kernel
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 8 + lightness, -1], [-1, -1, -1]], dtype=np.float32)  # Sharpen kernel

        kernel = (1 - alpha) * kernel_identity + alpha * kernel_sharpen  # Blend kernels
        return cv2.filter2D(image, -1, kernel)  # Apply convolution

    @staticmethod
    def label_indexer(vocab_dict: Dict[str, int], label: str) -> np.ndarray:
        return np.array([vocab_dict[c] for c in label if c in vocab_dict], dtype=np.int64)  # Map chars to indices

    @staticmethod
    def label_padding(padding_value: int, max_len: int, label: np.ndarray) -> np.ndarray:
        label = label[:max_len]  # Truncate label if too long
        return np.pad(label, (0, max_len - len(label)), mode="constant", constant_values=padding_value)  # Pad label

    def single_image_preprocessing(self, img: np.ndarray) -> torch.Tensor:
        img, _ = self.preprocess_img(img, "")  # Resize and pad image
        img = torch.from_numpy(img)  # Convert to tensor
        img = img.permute(2, 0, 1)  # Convert to CHW format
        img = img.float() / 255.0  # Normalize image
        return img  # Return processed tensor
