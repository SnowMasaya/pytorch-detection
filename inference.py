import argparse
import os
import random

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from reshape import reshape_model
from datasets import FaceDataset
from datasets.label_image import LabelImage
import cv2
import uuid
import matplotlib.pylab as plt
from torch2trt import torch2trt


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model_path', type=str, default='model_best.pth.tar', help="path to input PyTorch model (default: model_best.pth.tar)")
parser.add_argument('--image-dir', type=str, default='',
                    help='path to desired input directory for image data '
                         'png (default: current directory)')
parser.add_argument('--resolution', default=224, type=int, metavar='N',
                    help='input NxN image resolution of model (default: 224x224) '
                         'note than Inception models should use 299x299')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 8), this is the total '
                         'batch size of all GPUs on the current node ')
parser.add_argument('--data', default="FaceDataset", type=str,
                    help='set dataset type {FaceDataset, LabelImg}')
parser.add_argument('--predict_image_path', default="./predict_image/", type=str,
                    help='set predict image path (default: predict_image)')
parser.add_argument('--camera', help='set camera mode', action="store_true")



def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def inference():
    args = parser.parse_args()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.camera is False:
        if args.data == "FaceDataset":
            inference_dataset = FaceDataset(
                root_dir='datasets/raw/originalPics',
                fold_dir='datasets/raw/originalPics',
                fold_range=range(1,11),
                transform=transforms.Compose([
                    transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    normalize,
                ]),
                predict_mode=True,
            )
        elif args.data == "LabelImg":
            inference_dataset = LabelImage(
                data_dir='./test_image/',
                transform=transforms.Compose([
                    transforms.Resize((args.resolution, args.resolution)),
                    transforms.ToTensor(),
                    normalize,
                ]),
                predict_mode=True,
            )

        inference_loader = torch.utils.data.DataLoader(
            inference_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    checkpoint = torch.load(args.model_path)
    arch = checkpoint['arch']
    model = models.__dict__[arch](pretrained=True)
    # reshape the model's output
    model = reshape_model(model, arch, checkpoint['output_dims'])

    # load the model weights
    model.load_state_dict(checkpoint['state_dict'])

    model.eval().cuda()

    x = torch.ones((1, 3, 224, 224)).cuda()

    # convert to TensorRT feeding sample data as input
    model_trt = torch2trt(model, [x])

    target_predict_names = ["target", "predict"]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in target_predict_names]

    if not os.path.isdir(args.predict_image_path):
        os.makedirs(args.predict_image_path)

    if args.camera:
        capture = cv2.VideoCapture(0)

        while(True):
            ret, frame = capture.read()
            img, orig_im, dim = prep_image(frame, args.resolution)
 
            img = img.cuda()
            prediction = model_trt(img)
            prediction = prediction.cpu().detach().numpy()[0]
            org_width, org_height = dim
            width, height = args.resolution, args.resolution
            
            predict_values = [prediction[0], prediction[1]]
            scale_values = [org_width, width]
            prediction[0], prediction[1] = renormalize(predict_values, scale_values)

            predict_values = [prediction[2], prediction[3]]
            scale_values = [org_height, height]
            prediction[2], prediction[3] = renormalize(predict_values, scale_values)

            c1, c2 = (int(prediction[0]), int(prediction[2])), (int(prediction[1]), int(prediction[3]))
            color = colors[1]
            tl = round(0.005 * max(orig_im.shape[0:1]))
            cv2.rectangle(frame, c1, c2, color, thickness=tl)
 
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        capture.release()
        cv2.destroyAllWindows()

    else:
        for i, (images, target, org_image, rescale_value) in enumerate(inference_loader):
            inference_images = images.cuda()
            prediction = model_trt(inference_images)
            prediction = prediction.cpu().detach().numpy()[0]
            target = target.cpu().detach().numpy()[0]
            images = org_image.numpy()[0]
            org_width, org_height, width, height = rescale_value.cpu().detach().numpy()[0]

            target_values = [target[0], target[1]]
            scale_values = [org_width, width]
            target[0], target[1] = renormalize(target_values, scale_values)

            target_values = [target[2], target[3]]
            scale_values = [org_height, height]
            target[2], target[3] = renormalize(target_values, scale_values)

            c1, c2 = (int(target[0]), int(target[2])), (int(target[1]), int(target[3]))
            color = colors[0]
            tl = round(0.005 * max(images.shape[0:2]))
            cv2.rectangle(images, c1, c2, color, thickness=tl)
            display_txt = "Ground Truth"
            tf = max(tl - 1, 1)
            cv2.putText(images, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)

            predict_values = [prediction[0], prediction[1]]
            scale_values = [org_width, width]
            prediction[0], prediction[1] = renormalize(predict_values, scale_values)

            predict_values = [prediction[2], prediction[3]]
            scale_values = [org_height, height]
            prediction[2], prediction[3] = renormalize(predict_values, scale_values)

            c1, c2 = (int(prediction[0]), int(prediction[2])), (int(prediction[1]), int(prediction[3]))
            color = colors[1]
            tl = round(0.005 * max(images.shape[0:2]))
            cv2.rectangle(images, c1, c2, color, thickness=tl)
            display_txt = "Predict Result"
            tf = max(tl - 1, 1)
            cv2.putText(images, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)

            plt.figure(figsize=(20, 15))
            plt.imshow(images)
            plt.savefig(os.path.join(args.predict_image_path, str(uuid.uuid1()) + '.png'))


def renormalize(predict_values, scale_values):
    org_scale, scale = scale_values[0], scale_values[1]
    predict_top, predict_bottom = predict_values[0], predict_values[1]
    predict_bottom = predict_bottom * org_scale

    clac_scale = org_scale / scale
    predict_top = (predict_top / 2 + 0.5) * scale * clac_scale
    predict_bottom = predict_bottom + predict_top
    return predict_top, predict_bottom


if __name__ == '__main__':
    inference()
