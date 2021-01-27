import numpy as np
import torch
from PIL import Image
from skimage import exposure, morphology
from skimage.filters import threshold_otsu
from torch import nn
from torchvision import transforms


class AllInOneModel(nn.Module):
    def __init__(self, weight_path, device="cpu"):
        super(AllInOneModel, self).__init__()
        self.num_channels = 7
        self.device = device
        self.model = torch.jit.load(weight_path).to(device)

    def proc_img(self, image, focus_frame_idx):
        img_arr = []

        for chan_idx in range(
            focus_frame_idx - self.num_channels // 2,
            focus_frame_idx + self.num_channels // 2 + self.num_channels % 2,
        ):
            try:
                image.seek(chan_idx)
            except:
                assert 0, f"Input image does not have {chan_idx} channel"

            imarray_idx = np.array(image)
            img_arr.append(imarray_idx)

        image = np.stack(img_arr, axis=0).astype("float32")
        image = self.image_preprocessing(image)
        image = transforms.ToTensor()(np.ascontiguousarray(image))
        return image[None]

    def forward(self, img_path, focus_frame_idx):
        image = Image.open(img_path)
        image = self.proc_img(image, focus_frame_idx)

        pred_mask = self.tta(self.model, image.to(self.device))
        pred_mask = pred_mask[0][0].data.cpu().numpy()
        pred_mask = self.prediction_postprocessing(pred_mask)

        return pred_mask

    def image_preprocessing(self, image):
        for idx in range(image.shape[0]):
            image[idx] = self.normalize_img(image[idx])

        image = image.transpose((1, 2, 0))
        return image

    def prediction_postprocessing(self, all_pred):
        global_thresh = threshold_otsu(all_pred)
        all_pred = all_pred >= global_thresh
        all_pred = morphology.binary_opening(all_pred).astype("float32")
        return all_pred

    def normalize_img(self, image):
        p2, p98 = np.percentile(image, (2, 98))
        new_imarray = exposure.rescale_intensity(image, in_range=(p2, p98))
        new_imarray = self.data_norm(new_imarray, new_imarray.min(), new_imarray.max())
        return new_imarray

    def data_norm(self, x, cur_min, cur_max):
        return (x - cur_min) / (cur_max - cur_min)

    def tta(self, model, image):
        """Test-time augmentation for image classification that averages predictions
        of all D4 augmentations applied to input image.
        For segmentation we need to reverse the augmentation after making a prediction
        on augmented input.
        :param model: Model to use for making predictions.
        :param image: Model input.
        :return: Arithmetically averaged predictions
        """
        output = model(image).data.cpu()

        for aug, deaug in zip(
            [self.torch_rot90, self.torch_rot180, self.torch_rot270],
            [self.torch_rot270, self.torch_rot180, self.torch_rot90],
        ):
            x = deaug(model(aug(image)).data.cpu())
            output = output + x

        image = self.torch_transpose(image)

        for aug, deaug in zip(
            [self.torch_none, self.torch_rot90, self.torch_rot180, self.torch_rot270],
            [self.torch_none, self.torch_rot270, self.torch_rot180, self.torch_rot90],
        ):
            x = deaug(model(aug(image)).data.cpu())
            output = output + self.torch_transpose(x)

        one_over_8 = float(1.0 / 8.0)
        return output * one_over_8

    def torch_none(self, x):
        return x

    def torch_rot90_(self, x):
        return x.transpose_(2, 3).flip(2)

    def torch_rot90(self, x):
        return x.transpose(2, 3).flip(2)

    def torch_rot180(self, x):
        return x.flip(2).flip(3)

    def torch_rot270(self, x):
        return x.transpose(2, 3).flip(3)

    def torch_flipud(self, x):
        """
        Flip image tensor vertically
        :param x:
        :return:
        """
        return x.flip(2)

    def torch_fliplr(self, x):
        """
        Flip image tensor horizontally
        :param x:
        :return:
        """
        return x.flip(3)

    def torch_transpose(self, x):
        return x.transpose(2, 3)

    def torch_transpose_(self, x):
        return x.transpose_(2, 3)

    def torch_transpose2(self, x):
        return x.transpose(3, 2)
