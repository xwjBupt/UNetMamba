import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + ".png"
        mask_tif = label2rgb(mask)
        mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + ".png"
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-tp",
        "--test_weights_path",
        type=str,
        required=True,
        help="Path to trained weights",
    )
    arg(
        "-c",
        "--config_path",
        type=Path,
        default="/home/wjx/data/code/UNetMamba/config/loveda/unetmamba.py",
        required=False,
        help="Path to config",
    )
    arg(
        "-t",
        "--tta",
        help="Test time augmentation.",
        default=None,
        choices=[None, "d4", "lr"],
    )  ## lr is flip TTA, d4 is multi-scale TTA
    arg("--rgb", help="whether output rgb masks", default=True)
    arg("--val", help="whether eval Val set", default=True)
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    test_weights_path = args.test_weights_path
    if args.tta:
        output_path = test_weights_path.replace("model_weights", "fig_results")
    else:
        output_path = test_weights_path.replace("model_weights", "fig_results_NOTTA")
    os.makedirs(output_path, exist_ok=True)
    logger.add(os.path.join(output_path, "metric.log"))
    model = Supervision_Train.load_from_checkpoint(
        test_weights_path,
        config=config,
    )

    model.cuda()
    model.eval()
    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(
                    scales=[0.75, 1.0, 1.25, 1.5],
                    interpolation="bicubic",
                    align_corners=False,
                ),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset
    if args.val:
        evaluator = Evaluator(num_class=config.num_classes)
        evaluator.reset()
        test_dataset = config.val_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input["img"].cuda())

            image_ids = input["img_id"]
            if args.val:
                masks_true = input["gt_semantic_seg"]

            img_type = input["img_type"]

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids[i]
                mask_type = img_type[i]
                if args.val:
                    if not os.path.exists(os.path.join(output_path, mask_type)):
                        os.mkdir(os.path.join(output_path, mask_type))
                    evaluator.add_batch(
                        pre_image=mask, gt_image=masks_true[i].cpu().numpy()
                    )
                    results.append(
                        (
                            mask,
                            os.path.join(output_path, mask_type, mask_name),
                            args.rgb,
                        )
                    )
                else:
                    results.append(
                        (mask, os.path.join(output_path, mask_name), args.rgb)
                    )
    if args.val:
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()
        for class_name, class_iou, class_f1 in zip(
            config.classes, iou_per_class, f1_per_class
        ):
            logger.info(
                "mF1_{}:{}, IOU_{}:{}".format(
                    class_name, class_f1, class_name, class_iou
                )
            )
        logger.info(
            "mF1:{}, mIOU:{}, OA:{}".format(
                np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA
            )
        )

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    logger.info("images writing spends: {} s".format(img_write_time))


if __name__ == "__main__":
    main()
