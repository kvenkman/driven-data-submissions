from pathlib import Path
import numpy as np
import pandas as pd
from pandas_path import path
import random
import rasterio
import datetime

import torch
import albumentations
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import cv2

from torchsampler import ImbalancedDatasetSampler

assert torch.cuda.is_available() == True

random.seed(29)

DATA_PATH = Path("training_data")
train_metadata = pd.read_csv(DATA_PATH/"flood-training-metadata.csv", parse_dates=["scene_start"])


train_metadata["feature_path"] = (
    str(DATA_PATH / "train_features")
    / train_metadata.image_id.path.with_suffix(".tif").path
)

train_metadata["dem_path"] = (
    str(DATA_PATH / "nasadem")
    / train_metadata.chip_id.path.with_suffix(".tif").path
)

train_metadata["occurrence_path"] = (
    str(DATA_PATH / "jrc_occurrence")
    / train_metadata.chip_id.path.with_suffix(".tif").path
)

train_metadata["seasonality_path"] = (
    str(DATA_PATH / "jrc_seasonality")
    / train_metadata.chip_id.path.with_suffix(".tif").path
)

train_metadata["extent_path"] = (
    str(DATA_PATH / "jrc_extent")
    / train_metadata.chip_id.path.with_suffix(".tif").path
)

train_metadata["change_path"] = (
    str(DATA_PATH / "jrc_change")
    / train_metadata.chip_id.path.with_suffix(".tif").path
)

train_metadata["recurrence_path"] = (
    str(DATA_PATH / "jrc_recurrence")
    / train_metadata.chip_id.path.with_suffix(".tif").path
)

train_metadata["transitions_path"] = (
    str(DATA_PATH / "jrc_transitions")
    / train_metadata.chip_id.path.with_suffix(".tif").path
)

train_metadata["label_path"] = (
    str(DATA_PATH / "train_labels")
    / train_metadata.chip_id.path.with_suffix(".tif").path
)

chip_ids = train_metadata.chip_id.unique().tolist()

def get_paths_by_chip(image_level_df):
    """
    Returns a chip-level dataframe with pivoted columns
    for vv_path and vh_path.

    Args:
        image_level_df (pd.DataFrame): image-level dataframe

    Returns:
        chip_level_df (pd.DatzaFrame): chip-level dataframe
    """
    paths = []
    for chip, group in image_level_df.groupby("chip_id"):
        location = group[group.polarization == "vv"]["location"].values[0]
        vv_path = group[group.polarization == "vv"]["feature_path"].values[0]
        vh_path = group[group.polarization == "vh"]["feature_path"].values[0]
        
        dem_path = group[group.polarization == "vv"]["dem_path"].values[0]
        
        occurrence_path = group[group.polarization == "vv"]["occurrence_path"].values[0]
        seasonality_path = group[group.polarization == "vv"]["seasonality_path"].values[0]
        extent_path = group[group.polarization == "vv"]["extent_path"].values[0]
        
        change_path = group[group.polarization == "vv"]["change_path"].values[0]
        recurrence_path = group[group.polarization == "vv"]["recurrence_path"].values[0]
        transitions_path = group[group.polarization == "vv"]["transitions_path"].values[0]
        
        paths.append([chip, location, vv_path, vh_path, dem_path, occurrence_path, seasonality_path, 
                      extent_path, change_path, recurrence_path, transitions_path])
    
    return pd.DataFrame(paths, columns=["chip_id", "location", "vv_path", "vh_path", 
                                        "dem_path", "occurrence_path", "seasonality_path", "extent_path",
                                       "change_path", "recurrence_path", "transitions_path"])

# Given an array, normalize it
def normalize(array, min_val=None, max_val=None):
    
    if not min_val:
        min_val = np.nanmin(array)

    if not max_val:
        max_val = np.nanmax(array)
        
    if min_val == max_val:
        min_val, max_val = 0, 1
    
    array = np.clip(array, min_val, max_val)
    array = (array - min_val)/(max_val - min_val)
    return np.nan_to_num(array)

# Given a tif file path, read and return the requested channel
def read_tif(filename, channel=1):
    with rasterio.open(filename) as tif_file:
        return tif_file.read(channel).astype(np.float32)

class FloodDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self, x_paths, y_paths=None, transforms=None):
        self.data = x_paths
        self.label = y_paths
        self.transforms = transforms    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Loads a 2-channel image from a chip-level dataframe
        img = self.data.loc[idx]

        vv_img = read_tif(img.vv_path)
        vh_img = read_tif(img.vh_path)
        
        occurrence_img = normalize(read_tif(img.occurrence_path))
        seasonality_img = normalize(read_tif(img.seasonality_path))
        extent_img = normalize(read_tif(img.extent_path))
        
        dem_img = (normalize(read_tif(img.dem_path)) < 0.5).astype(np.float32)
        o_s_e_combined = ((0.25*occurrence_img + 0.25*seasonality_img + 0.5*extent_img)>0.5).astype(np.float32)
        
        x_arr = np.stack([vv_img, vh_img, o_s_e_combined, dem_img], axis = -1)

        # Load label if available - training only
        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1)

            # Apply same data augmentations to sample and label
            if self.transforms:
                transformed = self.transforms(image=x_arr, mask=y_arr)
                x_arr = transformed["image"]
                y_arr = transformed["mask"]

            x_arr = np.transpose(x_arr, [2, 0, 1])
            
            sample = {"chip_id": img.chip_id, "chip": x_arr, "label" : y_arr}
        else: # No labels - validation set only
            if self.transforms:
                x_arr = self.transforms(image=x_arr)['image']
            
            x_arr = np.transpose(x_arr, [2, 0, 1])
            sample = {"chip_id": img.chip_id, "chip": x_arr}

        return sample
        
# These transformations will be passed to our model class
training_transformations = albumentations.Compose(
    [
        albumentations.RandomRotate90(),
        albumentations.Flip(),
        albumentations.Transpose(),
    ]
)

class XEDiceLoss(torch.nn.Module):
    """
    Computes (0.5 * CrossEntropyLoss) + (0.5 * DiceLoss).
    """

    def __init__(self):
        super().__init__()
        self.xe = torch.nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, pred, true):
        valid_pixel_mask = true.ne(255)  # valid pixel mask

        # Cross-entropy loss
        temp_true = torch.where((true == 255), 0, true)  # cast 255 to 0 temporarily
        xe_loss = self.xe(pred, temp_true)
        xe_loss = xe_loss.masked_select(valid_pixel_mask).mean()

        # Dice loss
        pred = torch.softmax(pred, dim=1)[:, 1]
        pred = pred.masked_select(valid_pixel_mask)
        true = true.masked_select(valid_pixel_mask)
        dice_loss = 1 - (2.0 * torch.sum(pred * true)) / (torch.sum(pred + true) + 1e-7)

        return (0.5 * xe_loss) + (0.5 * dice_loss)

def intersection_and_union(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    return intersection.sum(), union.sum()


class FloodModel(pl.LightningModule):
    def __init__(self, hparams):
        super(FloodModel, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.backbone = self.hparams.get("backbone", "resnet34")
        self.weights = self.hparams.get("weights", "imagenet")
        self.learning_rate = self.hparams.get("lr", 1e-3)
        self.max_epochs = self.hparams.get("max_epochs", 1000)
        self.min_epochs = self.hparams.get("min_epochs", 6)
        self.patience = self.hparams.get("patience", 4)
        self.num_workers = self.hparams.get("num_workers", 2)
        self.batch_size = self.hparams.get("batch_size", 32)
        self.x_train = self.hparams.get("x_train")
        self.y_train = self.hparams.get("y_train")
        self.x_val = self.hparams.get("x_val")
        self.y_val = self.hparams.get("y_val")
        self.output_path = self.hparams.get("output_path", "model-outputs")
        self.gpu = self.hparams.get("gpu", False)
        self.experiment_name = self.hparams.get("experiment_name", 
                                                str(int(datetime.datetime.today().timestamp()))) + ".pt"
        self.transform = self.hparams.get("transformations", None)

        # Where final model will be saved
        self.output_path = Path.cwd() / self.output_path
        self.output_path.mkdir(exist_ok=True)

        # Track validation IOU globally (reset each epoch)
        self.intersection = 0
        self.union = 0
        self.criterion = XEDiceLoss()

        # Instantiate datasets, model, and trainer params
        self.train_dataset = FloodDataset(
            self.x_train, self.y_train, transforms=self.transform
        )
        self.val_dataset = FloodDataset(self.x_val, self.y_val, transforms=None)
        self.model = self._prepare_model()
        self.trainer_params = self._get_trainer_params()

    ## Required LightningModule methods ##
    def forward(self, image):
        # Forward pass
        return self.model(image)

    def training_step(self, batch, batch_idx):
        # Switch on training mode
        self.model.train()
        torch.set_grad_enabled(True)

        # Load images and labels
        x = batch["chip"]
        y = batch["label"].long()
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass
        preds = self.forward(x)

        # Calculate training loss
        xe_dice_loss = self.criterion(preds, y)

        # Log batch xe_dice_loss
        self.log(
            "xe_dice_loss",
            xe_dice_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return xe_dice_loss

    def validation_step(self, batch, batch_idx):
        # Switch on validation mode
        self.model.eval()
        torch.set_grad_enabled(False)

        # Load images and labels
        x = batch["chip"]
        y = batch["label"].long()
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass & softmax
        preds = self.forward(x)
        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > 0.5) * 1

        # Calculate validation IOU (global)
        intersection, union = intersection_and_union(preds, y)
        self.intersection += intersection
        self.union += union

        # Log batch IOU
        batch_iou = intersection / union
        self.log(
            "iou", batch_iou, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return batch_iou

    def train_dataloader(self):
        # DataLoader class for training
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Define scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.patience,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "val_loss",
        }  # logged value to monitor
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs):
        # Calculate IOU at end of epoch
        epoch_iou = self.intersection / self.union

        # Reset metrics before next epoch
        self.intersection = 0
        self.union = 0

        # Log epoch validation IOU
        self.log("val_loss", epoch_iou, on_epoch=True, prog_bar=True, logger=True)
        return epoch_iou

    ## Convenience Methods ##
    def _prepare_model(self):
        unet_model = smp.Unet(
            encoder_name=self.backbone,
            encoder_weights=self.weights,
            in_channels=4,
            classes=2,
        )
        if self.gpu:
            unet_model.cuda()
        return unet_model

    def _get_trainer_params(self):
        # Define callback behavior
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            monitor="val_loss",
            mode="max",
            verbose=True,
        )
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss",
            patience=(self.patience * 3),
            mode="max",
            verbose=True,
        )

        # Specify where TensorBoard logs will be saved
        self.log_path = Path.cwd() / self.hparams.get("log_path", "tensorboard-logs")
        self.log_path.mkdir(exist_ok=True)
        logger = pl.loggers.TensorBoardLogger(self.log_path, name=self.experiment_name)

        trainer_params = {
            "callbacks": [checkpoint_callback, early_stop_callback],
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "default_root_dir": self.output_path,
            "logger": logger,
            "gpus": None if not self.gpu else 1,
            "fast_dev_run": self.hparams.get("fast_dev_run", False),
            "num_sanity_val_steps": self.hparams.get("val_sanity_checks", 0),
        }
        return trainer_params

    def fit(self):
        # Set up and fit Trainer object
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)

train_chip_ids = random.sample(chip_ids, int(0.8*len(chip_ids))) # Train on 80% of the data

train = train_metadata[train_metadata.chip_id.isin(train_chip_ids)] # training dataset
val = train_metadata[~train_metadata.chip_id.isin(train_chip_ids)] # validation dataset

# Separate features from labels
val_x = get_paths_by_chip(val)
val_y = val[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)

train_x = get_paths_by_chip(train)
train_y = train[["chip_id", "label_path"]].drop_duplicates().reset_index(drop=True)

hparams = {
    # Required hparams
    "x_train": train_x,
    "x_val": val_x,
    "y_train": train_y,
    "y_val": val_y,
    # Optional hparams
    "backbone": "resnet101",
    "weights": "imagenet",
    "lr": 3e-5,
    "min_epochs": 70,
    "max_epochs": 1000,
    "patience": 8,
    "batch_size": 4,
    "num_workers": 16,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": "model-outputs",
    "log_path": "tensorboard_logs",
    "gpu": torch.cuda.is_available(),
    "transformations":training_transformations,
    "experiment_name":"stac-submission"
} 

flood_model = FloodModel(hparams=hparams)
flood_model.fit()

# Save model
submission_path = Path("model-outputs")
submission_path.mkdir(exist_ok=True)
submission_assets_path = submission_path / "assets"
submission_assets_path.mkdir(exist_ok=True)

weight_path = submission_assets_path / Path(hparams["experiment_name"] + ".pt")
torch.save(flood_model.state_dict(), weight_path)
