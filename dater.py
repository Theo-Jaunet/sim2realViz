import argparse
import csv

import torch
import cv2
import ujson as ujson
from torchvision import transforms, utils
from posest import Posest
from posest2 import Posest2
from utils import get_label, img_to_b64, depth_to_b64, adjust_angle, fullDist, euclidean, get_real_label
import numpy as np
import umap


def parsearg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=250, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.03, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='interval between model checkpoints')
    parser.add_argument('--num_workers', type=int, default=2, help='Workers for the dataset')
    parser.add_argument('--dataset_size', type=int, default=10000, help='Size of the dataset, -1 for full size')
    parser.add_argument('--n_out', type=int, default=3, help='Number of output')
    parser.add_argument('--model', type=str, default='posest', help=' Kind of model to use')

    return parser.parse_args()


def run(model, image):
    return model(image.cuda())


def getInput(filename, args, fromReal=False):
    print(filename)
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(filename, cv2.IMREAD_UNCHANGED),
                                  (args.img_width, args.img_height)).astype(np.float32), cv2.COLOR_BGR2RGB)

    depth = cv2.resize(cv2.imread(filename.replace('rgb', 'depth'), cv2.IMREAD_UNCHANGED),
                       (args.img_width, args.img_height)).astype(np.float32)

    depth = np.expand_dims(depth, axis=2)

    # print(depth.shape)
    # print(rgb.shape)

    # rgb = np.asarray(Image.open(filename))
    # depth = np.expand_dims(np.asarray(Image.open(filename.replace('rgb', 'depth'))), axis=2)

    merge = np.append(rgb, depth, axis=2).astype(np.float32)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
        ]
    )

    merge *= (1.0 / 255.0)
    merge = transform(merge)

    if fromReal:
        tlab = get_real_label(filename.split("/")[-1], 3)
    else:
        tlab = get_label(filename.split("/")[-1], 3)

    return torch.unsqueeze(merge, 0), torch.unsqueeze(tlab, 0).cuda()


def loadsave(model, modelpath):
    model.load_state_dict(torch.load(modelpath, map_location='cuda:0'))
    return model


if __name__ == '__main__':
    args = parsearg()
    model = loadsave(Posest(args), "models/new/editedv2/last.pth").cuda()
    # model = loadsave(Posest(args), "models/new/fine/last.pth").cuda()
    model.eval()

    # file = "/home/theo/Downloads/registered_jpeg_real2sim_locobot_citi_21-01-26/out/sim/expe_man_ctrl/match.csv"
    # path = "/home/theo/Downloads/registered_jpeg_real2sim_locobot_citi_21-01-26/"

    file = "data_source/traj_cap/traj1/sim/match.csv"
    path = "data_source/traj_cap/traj1/sim/"

    path_sim = "/home/theo/Documents/clones/sim2real/data/try_256/test_res/"

    res = {}

    res["real_dots"] = []
    res["sim_dots"] = []

    stats_r = []
    stats_s = []
    embeds_r = []
    embeds_s = []
    with torch.no_grad():
        with open(file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                row = dict(row)
                # r_img, r_lab = getInput(path + row["REAL_RGB"], args, True)
                # s_img, s_lab = getInput(path + row["SIM_RGB"], args)

                print(path_sim + row["SIM_RGB"])



                r_img, r_lab = getInput(path.replace("sim", "real") + row["REAL_RGB"], args, True)
                s_img, s_lab = getInput(path_sim + row["SIM_RGB"], args)

                r_coords, r_embed = run(model, r_img)
                s_coords, s_embed = run(model, s_img)
                embeds_r.append(r_embed.squeeze().cpu().numpy())
                embeds_s.append(s_embed.squeeze().cpu().numpy())

                # print(r_embed.shape)

    embeds_r = umap.UMAP(n_neighbors=30, min_dist=0.3).fit_transform(embeds_r).tolist()
    embeds_s = umap.UMAP(n_neighbors=30, min_dist=0.3).fit_transform(embeds_s).tolist()

    print(embeds_r[0])

    max_x = 18.93
    max_y = 19.8

    with torch.no_grad():
        with open(file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            id = 0
            for row in csv_reader:
                row = dict(row)
                temp_r = {}
                temp_s = {}

                # TODO: REINVERT X and Y to match GT
                # r_img, r_lab = getInput(path + row["REAL_RGB"], args, True)
                r_img, r_lab = getInput(path.replace("sim", "real") + row["REAL_RGB"], args, True)
                s_img, s_lab = getInput(path_sim + row["SIM_RGB"], args)

                # s_img, s_lab = getInput(path + row["SIM_RGB"], args)

                r_coords, r_embed = run(model, r_img)
                s_coords, s_embed = run(model, s_img)
                # ------------------------------------------------------------ Real
                temp_r["gt_x"] = round(s_lab[0][0].item(), 3)
                temp_r["gt_y"] = round(s_lab[0][1].item(), 3)
                temp_r["gt_r"] = adjust_angle(round(s_lab[0][2].item(), 3))

                # print(s_coords)

                # tx = max_x - (-(round(r_coords[0][0].item(), 2)))
                # tx = max_x - (-(round(r_coords[0][0].item(), 2)))
                tx = (round(r_coords[0][0].item(), 2))

                # if tx > max_x:
                #     tx = max_x - tx

                # ty = max_y - ((round(r_coords[0][1].item(), 2)))
                ty = (round(r_coords[0][1].item(), 2))

                # if ty > max_y:
                #     ty = max_y - ty

                temp_r["x"] = tx
                temp_r["y"] = ty
                # temp_r["r"] = round(r_coords[0][2].item(), 2)
                temp_r["r"] = adjust_angle(round(r_coords[0][2].item(), 2))

                temp_r["rgb"] = img_to_b64(r_img)
                temp_r["depth"] = depth_to_b64(r_img[:, 3:])

                # ------------------------------------------------------------ SIM

                temp_s["gt_x"] = (round(s_lab[0][0].item(), 3))
                temp_s["gt_y"] = round(s_lab[0][1].item(), 3)
                temp_s["gt_r"] = adjust_angle(round(s_lab[0][2].item(), 3))

                # tx = (-(round(s_coords[0][0].item(), 2)))
                tx = (round(s_coords[0][0].item(), 2))

                # if tx > max_x:
                #     tx = max_x - tx

                # ty = max_y - ((round(s_coords[0][1].item(), 2)))
                ty = (round(s_coords[0][1].item(), 2))

                if ty > max_y:
                    ty = max_y - ty

                temp_s["x"] = tx
                temp_s["y"] = ty
                temp_s["r"] = adjust_angle(round(s_coords[0][2].item(), 2))
                # temp_s["r"] = round(s_coords[0][2].item(), 2)

                temp_s["rgb"] = img_to_b64(s_img)
                temp_s["depth"] = depth_to_b64(s_img[:, 3:])

                # temp_r["perf"] = fullDist([temp_r["gt_x"], temp_r["gt_y"]], [temp_r["x"], temp_r["y"]], temp_r["gt_r"],
                #                           temp_r["r"])
                #
                # temp_s["perf"] = fullDist([temp_s["gt_x"], temp_s["gt_y"]], [temp_s["x"], temp_s["y"]], temp_s["gt_r"],
                #                           temp_s["r"])

                temp_r["perf"] = euclidean([temp_r["gt_x"], temp_r["gt_y"]], [temp_r["x"], temp_r["y"]])
                temp_s["perf"] = euclidean([temp_s["gt_x"], temp_s["gt_y"]], [temp_s["x"], temp_s["y"]])

                temp_r["id"] = id
                temp_s["id"] = id

                temp_r["proj"] = embeds_r[id]
                temp_s["proj"] = embeds_s[id]

                # print(temp_r["x"])
                # print(temp_r["y"])

                # print(euclidean([-2, -2], [20, 20]))
                # break

                stats_r.append(temp_r["perf"])
                stats_s.append(temp_s["perf"])

                res["real_dots"].append(temp_r)
                res["sim_dots"].append(temp_s)

                id += 1
                # print(temp_r)

            res["real_perf"] = sum(stats_r) / len(stats_r)
            res["sim_perf"] = sum(stats_s) / len(stats_s)

            with open("model_edited_proj.json", 'w') as wjson:
                ujson.dump(res, wjson, ensure_ascii=False, sort_keys=True, indent=4)
            print(fullDist([temp_r["gt_x"], temp_r["gt_y"]], [temp_r["gt_x"], temp_r["gt_y"]], temp_r["gt_r"],
                           temp_r["r"]))

            # image_r = getInput(row)
