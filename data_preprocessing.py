import cv2  # OpenCV for image processing
import numpy as np  # NumPy for array operations
import torch  # PyTorch for tensor operations
from torchvision import transforms  # For image augmentations
from typing import Tuple, Dict  # For type hints

class Preprocessor:
    def __init__(
            self,
            image_size: Tuple[int, int] = (224, 244),  # Target image size (width, height)
            augmentation: bool = False,  # Whether to apply augmentation
            vocab: str = ""  # String of characters used for labeling
    ):
        self.image_size = image_size  # Store target image size
        self.augment = augmentation  # Store augmentation flag
        self.vocab = vocab  # Store vocabulary string
        self.vocab_dict = {}  # Initialize empty dictionary for char-to-index mapping
        for i, c in enumerate(vocab):  
            self.vocab_dict[c] = i  # Map each character to a unique integer index

        self.affine_transform = transforms.Compose([  # Compose transformations for augmentation
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image
            transforms.RandomAffine(
                degrees=15,  # Random rotation in [-15, 15] degrees
                shear=10,  # Random shear angle
                scale=(0.8, 1.2),  # Random scaling factor
                translate=(0.2, 0.4)  # Random translation as fraction of width/height
            )
        ])    

    def __call__(
            self,
            image: np.ndarray,
            label: str,
            max_len: int = 40  # Maximum label length for padding
    ):
        img, label = self.preprocess_img(image, label)  # Resize and pad image
        label = label.lower().strip()  # Lowercase and remove leading/trailing spaces
        label = label.indexer(self.vocab_dict, label)  # Convert label chars to indices
        label = self.label_padding(len(self.vocab), max_len, label)  # Pad label to max_len

        if self.augment:  
            img = self.apply_augmentation(img)  # Apply random augmentations if enabled
        img = torch.from_numpy(img)  # Convert NumPy array to PyTorch tensor
        img = img.permute(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
        img = img.float() / 255.0  # Normalize pixel values to [0,1]
        return img, label  # Return processed image and label
    
    def preprocess_img(
            self,
            img: np.ndarray,
            text: str
    ):
        target_width, target_height = self.image_size  # Get target width and height
        h, w = img.shape[:2]  # Original image height and width
        scale = min(target_width / w, target_height / h)  # Scaling factor to maintain aspect ratio
        new_width, new_height = int(w * scale), int(h * scale)  # Compute new dimensions
        img = cv2.resize(img, (new_width, new_height))  # Resize image

        pad_w = target_width - new_width  # Total horizontal padding needed
        pad_h = target_height - new_height  # Total vertical padding needed
        left = pad_w // 2  # Left padding
        right = pad_w - left  # Right padding
        top = pad_h // 2  # Top padding
        bottom = pad_h - top  # Bottom padding

        img = cv2.copyMakeBorder(
            img,
            top,  # Top border
            bottom,  # Bottom border
            left,  # Left border
            right,  # Right border
            cv2.BORDER_CONSTANT,  # Use constant color for padding
            value=255  # White padding
        )

        return img, text  # Return padded image and unchanged label
    
    def apply_augmentation(
            self,
            img: np.ndarray
    ) -> np.ndarray:
        
        if np.random.rand() < 0.5:  # 50% chance to apply sharpening
            img = self.random_sharpen(img)  # Apply random sharpening

        if np.random.rand() < 0.5:  # 50% chance to apply affine transform
            img = np.array(self.affine_transform(img))  # Apply affine transform and convert back to NumPy
        return img  # Return augmented image
    
    @staticmethod
    def random_sharpen(
        image: np.ndarray,  # Input image as NumPy array
        strength_range: Tuple[float, float] = (0.25, 1.0),  # Range for sharpening strength
        lightness_range: Tuple[float, float] = (0.75, 2.0)  # Range for brightness adjustment
    ) -> np.ndarray:
        alpha = np.random.uniform(*strength_range)  # Randomly pick sharpening strength
        lightness = np.random.uniform(*lightness_range)  # Randomly pick center weight for kernel

        kernel_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)  # Identity kernel (no change)
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 8 + lightness, -1], [-1, -1, -1]], dtype=np.float32)  # Sharpen kernel emphasizing edges

        kernel = (1 - alpha) * kernel_identity + alpha * kernel_sharpen  # Blend identity and sharpen kernels
        return cv2.filter2D(image, -1, kernel)  # Apply convolution with blended kernel

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
