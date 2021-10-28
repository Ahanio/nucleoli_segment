import numpy as np
import torch
from inferno.extensions import model as inf_model
from skimage import exposure, morphology
from skimage.filters import threshold_otsu
from torch import nn


class AllInOneModel(nn.Module):
    def __init__(self, focus_frame_idx):
        super(AllInOneModel, self).__init__()
        self.num_channels = 7
        self.model = build_standart_model(self.num_channels, 1, no_sigm=False)
        self.model.eval()
        self.focus_frame_idx = focus_frame_idx

    def forward(self, image):  # BxCxHxW
        input = []
        for i in range(len(image)):
            cur_image = image[i].cpu().data.numpy()
            cur_image = self.filter_channels(cur_image, self.focus_frame_idx) #[i]
            cur_image = self.image_preprocessing(cur_image)
            input.append(cur_image)
        input = np.stack(input, axis=0)
        input = torch.Tensor(input).float()

        result = []
        pred_mask = self.tta(self.model, input.to(next(self.model.parameters()).device))
        for i in range(pred_mask.shape[0]):
            cur_pred = pred_mask[i][0].data.cpu().numpy()
            cur_pred = self.prediction_postprocessing(cur_pred)
            result.append(cur_pred[None])

        result = np.stack(result, axis=0)
        result = torch.Tensor(result).float()
        return result

    def filter_channels(self, image, focus_frame_idx):
        return image[
            focus_frame_idx
            - self.num_channels // 2 : focus_frame_idx
            + self.num_channels // 2
            + self.num_channels % 2
        ]

    def image_preprocessing(self, image):
        for idx in range(image.shape[0]):
            image[idx] = self.normalize_img(image[idx])
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


def build_standart_model(image_channels, pred_channels=1, no_sigm=False):
    if no_sigm:
        return torch.nn.Sequential(
            inf_model.ResBlockUNet(dim=2, in_channels=image_channels, out_channels=pred_channels, activated=False),
        )
    else:
        return torch.nn.Sequential(
            inf_model.ResBlockUNet(dim=2, in_channels=image_channels, out_channels=pred_channels, activated=False),
            torch.nn.Sigmoid()
        )