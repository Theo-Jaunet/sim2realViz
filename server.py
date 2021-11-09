import binascii
import csv
import io
import os

import torch
import ujson
import umap
from PIL import Image, ImageEnhance
from flask import Flask, render_template, request, session, redirect, logging, jsonify, Response
import base64

from flask_caching import Cache
from flask_compress import Compress
import numpy as np

from get_obs import reconfigure_rgbd_camera_in_sim, get_observation_from_sim
from pointer import *
from habitat.config.default import get_config
import cv2
from torch.nn import functional as F

from posest import Posest
from utils import loadsave, parsearg, get_label, depth_to_b64, adjust_angle, euclidean, img_to_b64
from torchvision import transforms, utils
import matplotlib.cm as cm

import habitat

app = Flask(__name__)

COMPRESS_MIMETYPES = ['text/html', 'text/css', 'text/csv', 'text/xml', 'application/json',
                      'application/javascript', 'image/jpeg', 'image/png']
COMPRESS_LEVEL = 6
COMPRESS_MIN_SIZE = 500

cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)
Compress(app)

app = Flask(__name__)
cfg = get_config("try.yaml")
app.secret_key = binascii.hexlify(os.urandom(24))
intrinsics = get_intrinsics_from_config(cfg, "depth")
rgb_intrinsics = get_intrinsics_from_config(cfg, "rgb")
map_bounds = (-1.6, -1.3, 18.93, 19.81)
map_height = 0.047

args = parsearg()
model = loadsave(Posest(args), "models/new/vanilla/140.pth").cuda()
model2 = loadsave(Posest(args), "models/new/dataAug/100.pth").cuda()
model3 = loadsave(Posest(args), "models/new/fine/last.pth").cuda()
model4 = loadsave(Posest(args), "models/new/editedv2/last.pth").cuda()

model.eval()
model2.eval()
model3.eval()
model4.eval()

default_rgb = {
    "bright": 1.,
    "contrast": 1.,
    "sepia": 0.
}

default_depth = {
    "bright": 1.,
    "blur": 0.,

}

file = "data_source/traj_cap/traj1/sim/match.csv"
total = {}
with open(file, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for row in csv_reader:
        row = dict(row)
        total[row["SIM_RGB"]] = row["REAL_RGB"]

file2 = "data_source/traj_cap/traj2/sim/match.csv"

with open(file2, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for row in csv_reader:
        row = dict(row)
        total[row["SIM_RGB"]] = row["REAL_RGB"]


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/coords', methods=["POST"])
def coords():
    img = request.form['img']
    img_coord = request.form['img_coord']

    depth = cv2.resize(cv2.imread(img.replace('rgb', 'depth'), cv2.IMREAD_UNCHANGED),
                       (256, 256)).astype(np.float32)

    obs = depth * (1 / 255) * cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH \
          + cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH

    *sensor_pos, deg = (float(s[1:]) for s in img.split('/')[-1].split('_')[:4])
    sensor_heading = numpy.radians(deg)

    clk_loc = list(map(int, img_coord.split(",")))
    rel_clk_pos = project_depth_pixel_to_rel_pos(clk_loc, obs[clk_loc[1], clk_loc[0]],
                                                 intrinsics)
    clk_pos = tranform_rel_pos_to_abs(rel_clk_pos, sensor_pos, sensor_heading)

    return jsonify(clk_pos)


@app.route('/global_mapping', methods=["POST"])
def global_mapping():
    n_sq = 128  # Portion of image
    rep = 12  # portion of map

    nb_map = 22 * rep

    mapDat = np.zeros((nb_map, nb_map))

    n_sq = 6  # TODO: IF OCCLU

    path = "data_source/traj_cap/traj1/sim_edit"
    rpath = "data_source/traj_cap/traj1/real/"

    max_occlu = -999
    min_occlu = 999
    # if cam:
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            row = dict(row)
            key = path + row["SIM_RGB"]
            depth = cv2.resize(cv2.imread(key.replace('rgb', 'depth'), cv2.IMREAD_UNCHANGED),
                               (256, 256)).astype(np.float32)

            inp = getInput(key, args)

            inp2 = getInput(rpath + row["REAL_RGB"], args)

            dist = featDist(model2, inp, inp2)

            tmapDat = sal2map(np.copy(dist) * 255, depth, n_sq * 8, rep, key)
            mapDat = np.add(mapDat, tmapDat)

        mapDat = mapDat - np.min(mapDat)
        mapDat = mapDat / np.max(mapDat)

    with open("feat_dist_add.json", 'w') as wjson:
        ujson.dump(mapDat.tolist(), wjson, ensure_ascii=False, sort_keys=True, indent=4)

    return jsonify(mapDat.tolist())


def featDist(model, sim_inp, real_inp, act_sim=None, act_real=None):
    if act_sim is None or act_real is None:
        output_cam, act_sim, _ = getCam(model, sim_inp)
        output_cam2, act_real, _ = getCam(model, real_inp)

    dist = 1 - F.cosine_similarity(act_sim.clone(), act_real.clone(), dim=0)

    dist2 = torch.sum(F.l1_loss(act_sim.clone(), act_real.clone(), reduction="none"), 0)

    dist2 = dist2 + 0.011
    #
    dist = (dist.clone() * (dist2.clone() * 0.03)).numpy()

    # Those terms are the global min and max
    dist = dist - 0.0005779581  # from model 2
    dist = dist / 4.770105

    dist = np.uint8(255 * dist)
    #
    dist = cv2.resize(dist, (256, 256))

    return dist


@app.route('/mapping', methods=["POST"])
def mapping():
    img = request.form['img']
    occlu = request.form['occlu'] == "true"
    cam = request.form['cam'] == "true"
    feat = request.form['feat'] == "true"

    depth = cv2.resize(cv2.imread(img.replace('rgb', 'depth'), cv2.IMREAD_UNCHANGED),
                       (256, 256)).astype(np.float32)

    inp = getInput(img, args)

    real_path = "data_source/" + total["/".join(img.split("/")[1:])]

    # real_path = path + total["/".join(img.split("/")[1:])]

    inp2 = getInput(real_path, args)
    n_sq = 128
    rep = 12

    if cam:
        output_cam, temp1, _ = getCam(model, inp)
        output_cam2, temp2, _ = getCam(model, inp2)

        saver = io.BytesIO()

        mapDat = sal2map(np.copy(output_cam), depth, n_sq, rep, img)

        output_cam = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET)
        output_cam = cv2.cvtColor(output_cam, cv2.COLOR_BGR2RGB)

        Image.fromarray(output_cam, mode="RGB").save(saver, format="JPEG")  # TODO: SWAP FOR CAM

        saver2 = io.BytesIO()
        saver2 = io.BytesIO()

        output_cam2 = cv2.applyColorMap(output_cam2, cv2.COLORMAP_JET)
        output_cam2 = cv2.cvtColor(output_cam2, cv2.COLOR_BGR2RGB)

        Image.fromarray(output_cam2, mode="RGB").save(saver2, format="JPEG")  # TODO: SWAP FOR CAM

        return jsonify({"map": mapDat.tolist(),
                        "sim_sal": str(base64.b64encode(saver.getvalue())).replace("b'", "").replace("'", ""),
                        "real_sal": str(base64.b64encode(saver2.getvalue())).replace("b'", "").replace("'", "")
                        })

    if feat:
        dist = featDist(model, inp, inp2)
        n_sq = 128

        mapDat = sal2map(dist, depth, n_sq, rep, img)

        dist = cv2.applyColorMap(dist, cv2.COLORMAP_JET)
        dist = cv2.cvtColor(dist, cv2.COLOR_BGR2RGB)

        saver = io.BytesIO()
        Image.fromarray(dist, mode="RGB").save(saver, format="JPEG")  # TODO: SWAP FOR CAM

        saver2 = io.BytesIO()
        Image.fromarray(dist, mode="RGB").save(saver2, format="JPEG")  # TODO: SWAP FOR CAM

        return jsonify({"map": mapDat.tolist(),
                        "sim_sal": str(base64.b64encode(saver.getvalue())).replace("b'", "").replace("'", ""),
                        "real_sal": str(base64.b64encode(saver2.getvalue())).replace("b'", "").replace("'", "")
                        })
    if occlu:
        n_sq = 6
        sal = occlusion(model, inp2[0], inp[0], n_sq, img).numpy()

        mapDat = sal2map(np.copy(sal) * 255, depth, n_sq * 8, rep, img)

        scale = max(sal[sal > 0].max(), -sal[sal <= 0].min())

        sal = sal / scale * 0.5
        sal += 0.5

        sal = cm.bwr_r(sal)[..., :3]

        sal = np.uint8(sal * 255.0)
        sal = np.reshape(sal, (256, 256, 3))

        sal2 = occlusion(model2, inp[0], inp2[0], n_sq, img).numpy()

        scale2 = max(sal2[sal2 > 0].max(), -sal2[sal2 <= 0].min())

        sal2 = sal2 / scale2 * 0.5
        sal2 += 0.5

        sal2 = cm.bwr_r(sal2)[..., :3]

        sal2 = np.uint8(sal2 * 255.0)
        sal2 = np.reshape(sal2, (256, 256, 3))

        saver = io.BytesIO()
        Image.fromarray(sal, mode="RGB").save(saver, format="JPEG")

        saver2 = io.BytesIO()
        Image.fromarray(sal2, mode="RGB").save(saver2, format="JPEG")

        return jsonify({"map": mapDat.tolist(),
                        "sim_sal": str(base64.b64encode(saver2.getvalue())).replace("b'", "").replace("'", ""),
                        "real_sal": str(base64.b64encode(saver.getvalue())).replace("b'", "").replace("'", ""),
                        })


@app.route('/mapping2', methods=["POST"])
def mapping2():
    img = request.form['img']
    occlu = request.form['occlu'] == "true"
    cam = request.form['cam'] == "true"
    feat = request.form['feat'] == "true"
    mod = request.form['model']

    res = {}
    tmod = model

    if mod == "2":
        tmod = model2
    if mod == "3":
        tmod = model3

    depth = cv2.resize(cv2.imread(img.replace('rgb', 'depth'), cv2.IMREAD_UNCHANGED),
                       (256, 256)).astype(np.float32)

    path = "/".join(img.split("/")[:-1]).replace("sim", "real") + "/"
    inp = getInput(img, args)

    tFile = img.split("/")[-1]

    real_path = path + total[tFile]

    inp2 = getInput(real_path, args)
    n_sq = 128
    rep = 12

    output_cam, act_sim, _ = getCam(tmod, inp)
    output_cam2, act_real, _ = getCam(tmod, inp2)

    saver = io.BytesIO()

    output_cam = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET)
    output_cam = cv2.cvtColor(output_cam, cv2.COLOR_BGR2RGB)

    Image.fromarray(output_cam, mode="RGB").save(saver, format="JPEG")  # TODO: SWAP FOR CAM

    saver2 = io.BytesIO()

    output_cam2 = cv2.applyColorMap(output_cam2, cv2.COLORMAP_JET)
    output_cam2 = cv2.cvtColor(output_cam2, cv2.COLOR_BGR2RGB)

    Image.fromarray(output_cam2, mode="RGB").save(saver2, format="JPEG")  # TODO: SWAP FOR CAM

    if cam:
        mapDat = sal2map(np.copy(output_cam2), depth, n_sq, rep, img)
        res["map"] = mapDat.tolist()

    res["activ_sim_sal"] = str(base64.b64encode(saver.getvalue())).replace("b'", "").replace("'", "")
    res["activ_real_sal"] = str(base64.b64encode(saver2.getvalue())).replace("b'", "").replace("'", "")

    dist = featDist(tmod, inp, inp2, act_sim, act_real)

    if feat:
        mapDat = sal2map(dist, depth, n_sq, rep, img)
        res["map"] = mapDat.tolist()

    dist = cv2.applyColorMap(dist, cv2.COLORMAP_JET)
    dist = cv2.cvtColor(dist, cv2.COLOR_BGR2RGB)

    saver3 = io.BytesIO()
    Image.fromarray(dist, mode="RGB").save(saver3, format="JPEG")  # TODO: SWAP FOR CAM

    saver4 = io.BytesIO()
    Image.fromarray(dist, mode="RGB").save(saver4, format="JPEG")  # TODO: SWAP FOR CAM

    res["feat_sim_sal"] = str(base64.b64encode(saver3.getvalue())).replace("b'", "").replace("'", "")
    res["feat_real_sal"] = str(base64.b64encode(saver4.getvalue())).replace("b'", "").replace("'", "")

    n_sq = 5
    sal = occlusion(tmod, inp[0], inp2[0], n_sq, img).cpu().numpy()
    if occlu:
        mapDat = sal2map(np.copy(sal) * 255, depth, n_sq * 8, rep, img)
        res["map"] = mapDat.tolist()

    scale = 1 / (abs(-1) + 1)

    sal += 1
    sal = sal * scale
    sal[sal > 1] = 1.0
    sal[sal < 0] = 0

    sal = cm.bwr_r(sal)[..., :3]
    sal = np.uint8(sal * 255.0)
    sal = np.reshape(sal, (256, 256, 3))

    saver5 = io.BytesIO()
    Image.fromarray(sal, mode="RGB").save(saver5, format="JPEG")

    saver6 = io.BytesIO()

    res["occlu_sim_sal"] = str(base64.b64encode(saver5.getvalue())).replace("b'", "").replace("'", "")
    res["occlu_real_sal"] = str(base64.b64encode(saver6.getvalue())).replace("b'", "").replace("'", "")

    return jsonify(res)


def sal2map(sal, depth, n_sq, rep, img):
    bob = torch.from_numpy(sal).squeeze().unsqueeze_(0).float() * (1 / 255)
    pool_size = 256 / n_sq
    bob = F.max_pool2d(bob, int(pool_size)).squeeze().numpy()

    nb_map = 22 * rep

    mapDat = np.zeros((nb_map, nb_map))

    obs = depth * (1 / 255) * cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH \
          + cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH

    *sensor_pos, deg = (float(s[1:]) for s in img.split('/')[-1].split('_')[:4])
    sensor_heading = numpy.radians(deg)

    for i in range(n_sq):
        for j in range(n_sq):

            if bob[j][i] > 0.15:

                tx = int(i * pool_size)
                ty = int(j * pool_size)
                clk_loc = [tx, ty]

                rel_clk_pos = project_depth_pixel_to_rel_pos(clk_loc, obs[clk_loc[1], clk_loc[0]],
                                                             intrinsics)

                clk_pos = tranform_rel_pos_to_abs(rel_clk_pos, sensor_pos, sensor_heading)

                inds = coords2ind([clk_pos[0] + 2, clk_pos[2] + 2], rep)

                if inds[0] > nb_map:
                    inds[0] = 264
                if inds[1] > nb_map:
                    inds[1] = 264

                mapDat[inds[0] - 2][inds[1] - 2] = max(bob[j][i], mapDat[inds[0] - 2][inds[1] - 2])

    return mapDat


def coords2ind(coords, rep):
    estim_x = coords[0]
    estim_y = coords[1]

    fix_x = estim_x - int(estim_x)
    fix_y = estim_y - int(estim_y)

    minx = 1
    miny = 1
    xind = 0
    yind = 0

    step = 0
    for i in range(rep):
        if abs(step - fix_x) < minx:
            minx = abs(step - fix_x)
            xind = i
        if abs(step - fix_y) < miny:
            miny = abs(step - fix_y)
            yind = i

        step += 1 / rep

    return [int(estim_x) * rep + xind, int(estim_y) * rep + yind]


def getCam(model, inp):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    weight = model.fc3.weight.cpu().data.numpy()
    conv = list(model._modules["conv_head"]._modules.items())[-1][1]
    conv.register_forward_hook(get_activation('conv1'))
    output = model(inp[0].cuda())

    act = activation['conv1'].squeeze().cpu()

    temp = act.numpy()

    nc, h, w = temp.shape

    beforeDot = temp.reshape((nc, h * w))

    tempo = np.zeros((16, 16))

    for i in range(weight.shape[0]):
        tempo = np.add(tempo, weight[i].dot(beforeDot).reshape(h, w))

    tempo = F.relu(torch.from_numpy(tempo)).numpy()

    cam = tempo - np.min(tempo)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cv2.resize(cam_img, (256, 256)), act, output[0]


def getInput(filename, args):
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(filename, cv2.IMREAD_UNCHANGED),
                                  (args.img_width, args.img_height)).astype(np.float32), cv2.COLOR_BGR2RGB)

    depth = cv2.resize(cv2.imread(filename.replace('rgb', 'depth'), cv2.IMREAD_UNCHANGED),
                       (args.img_width, args.img_height)).astype(np.float32)

    depth = np.expand_dims(depth, axis=2)

    merge = np.append(rgb, depth, axis=2).astype(np.float32)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
        ]
    )

    merge *= (1.0 / 255.0)
    merge = transform(merge)

    return torch.unsqueeze(merge, 0).cuda(), torch.unsqueeze(get_label(filename.split("/")[-1], 3), 0).cuda()


@app.route('/single_pix', methods=["POST"])
def single_pix():
    rep = 6  # portion of map
    img = request.form['img']

    depth = cv2.resize(cv2.imread(img.replace('rgb', 'depth'), cv2.IMREAD_UNCHANGED),
                       (256, 256)).astype(np.float32)

    mapDat = []

    for i in range(depth.shape[0]):
        # mapDat.append([])
        temp = []
        for j in range(depth.shape[1]):
            obs = depth * (1 / 255) * cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH \
                  + cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH

            *sensor_pos, deg = (float(s[1:]) for s in img.split('/')[-1].split('_')[:4])
            sensor_heading = numpy.radians(deg)

            clk_loc = [i, j]

            rel_clk_pos = project_depth_pixel_to_rel_pos(clk_loc, obs[clk_loc[1], clk_loc[0]],
                                                         intrinsics)

            clk_pos = tranform_rel_pos_to_abs(rel_clk_pos, sensor_pos, sensor_heading)

            temp.append([clk_pos[0], clk_pos[2]])
        mapDat.append(temp)

    return jsonify(mapDat)


def apply_patch(x, y, size, source, target):
    source = source.clone()
    source[:, :, x:x + size, y:y + size] = target[:, :, x:min(x + size, 256), y:min(256, y + size)]
    return source


def occlusion(model, source, target, nb, label):
    label = get_label(label, 3).unsqueeze_(0).cuda()

    source_rgb = source[:, :3, :, :]
    source_depth = source[:, 3, :, :].unsqueeze_(0)

    target_rgb = target[:, :3, :, :]
    target_depth = target[:, 3, :, :].unsqueeze_(0)

    coords, _ = model(source.cuda())

    ref_loss = F.mse_loss(coords[:, :2], label[:, :2])

    res_both = torch.zeros_like(target_depth)

    size = int(256 / nb)
    for i in range(nb):
        for w in range(nb):
            temp_rgb = apply_patch(i * size, w * size, size, source_rgb, target_rgb)
            temp_depth = apply_patch(i * size, w * size, size, source_depth, target_depth)

            stacked_both = torch.cat((temp_rgb, temp_depth), 1).cuda()

            coords_both, _ = model(stacked_both)

            both_loss = F.mse_loss(coords_both[:, :2], label[:, :2])

            temp_both = torch.full_like(target_depth, ref_loss.item() - both_loss.item())
            res_both = apply_patch(i * size, w * size, size, res_both, temp_both)

    maps = res_both

    return maps


@app.route('/occlu', methods=["POST"])
def occlu():
    img = request.form['img']
    depth = cv2.resize(cv2.imread(img.replace('rgb', 'depth'), cv2.IMREAD_UNCHANGED),
                       (256, 256)).astype(np.float32)

    inp = getInput(img, args)

    real_path = "data_source/" + total["/".join(img.split("/")[1:])]
    inp2 = getInput(real_path, args)

    sal = occlusion(model, inp2, inp, 8, img)

    mapDat = sal2map(sal, depth, 8, 6, img)

    return jsonify(mapDat.tolist())


@app.route('/runmod', methods=["POST"])
def runMod():
    img = request.form['img']
    mods = ujson.loads(request.form['mods'])

    mod = request.form['model']
    traj = request.form['traj']

    path = "data_source/traj_cap/traj" + str(traj) + "/real/"

    tmod = model

    if mod == "2":
        tmod = model2
    if mod == "3":
        tmod = model3
    if mod == "4":
        tmod = model4

    inp = getInput(path + total[img], args)

    source = inp[0].cpu()

    source_rgb = source[:, :3, :, :]
    source_depth = source[:, 3, :, :]

    for key, val in mods["rgb"].items():
        val = float(val)
        if not val == default_rgb[key]:
            if not torch.is_tensor(source_rgb):
                source_rgb = torch.from_numpy(source_rgb)
            source_rgb = apply(source_rgb, key, val)

    for key, val in mods["depth"].items():
        val = float(val)
        if not val == default_depth[key]:
            source_depth = apply(source_depth, key, val)
            if not torch.is_tensor(source_depth):
                source_depth = torch.from_numpy(source_depth)

    transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
        ]
    )

    if not torch.is_tensor(source_rgb):
        source_rgb = torch.from_numpy(source_rgb)

        source = source_rgb.squeeze().permute((1, 2, 0))
        source.add_(-(-1)).div_(1 - (-1) + 1e-5)
        source = np.uint8(source * 255.0)
        source = np.clip(source, 0, 255)
        # source = Image.fromarray(source, 'RGB')
        # source.save("temp.jpg")

    if not torch.is_tensor(source_depth):
        source_depth = torch.from_numpy(source_depth)
    source_depth.unsqueeze_(0)

    real = torch.cat((source_rgb.float(), source_depth.float()), 1).cuda().squeeze_()
    real.unsqueeze_(0)
    real = (real, "")
    sim = getInput(path.replace("real", "sim") + img, args)

    res = makeHeats(sim, real, tmod, path.replace("real", "sim") + img)

    return jsonify(res)


@app.route('/run_all', methods=["POST"])
def runmodAll():
    res = {}

    res["real_dots"] = []
    res["sim_dots"] = []

    mods = ujson.loads(request.form['mods'])

    mod = request.form['model']
    traj = request.form['traj']

    tmod = model

    if mod == "2":
        tmod = model2
    if mod == "3":
        tmod = model3
    if mod == "4":
        tmod = model4

    file = "data_source/traj_cap/traj" + str(traj) + "/sim/match.csv"
    path = "data_source/traj_cap/traj" + str(traj) + "/sim/"

    stats_r = []
    stats_s = []
    embeds_r = []
    embeds_s = []

    with torch.no_grad():
        with open(file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            id = 0
            for row in csv_reader:
                row = dict(row)
                temp_r = {}
                temp_s = {}

                r_img, r_lab = getInput(path.replace("sim", "real") + row["REAL_RGB"], args)

                source = r_img.cpu()
                source_rgb = source[:, :3, :, :]
                source_depth = source[:, 3, :, :]  # .unsqueeze_(0)

                for key, val in mods["rgb"].items():
                    val = float(val)
                    if not val == default_rgb[key]:
                        if not torch.is_tensor(source_rgb):
                            source_rgb = torch.from_numpy(source_rgb)
                        source_rgb = apply(source_rgb, key, val)

                for key, val in mods["depth"].items():
                    val = float(val)
                    if not val == default_depth[key]:
                        source_depth = apply(source_depth, key, val)
                        if not torch.is_tensor(source_depth):
                            source_depth = torch.from_numpy(source_depth)
                source_depth = gaussian(source_depth, 5)

                if not torch.is_tensor(source_rgb):
                    source_rgb = torch.from_numpy(source_rgb)

                if not torch.is_tensor(source_depth):
                    source_depth = torch.from_numpy(source_depth)
                source_depth.unsqueeze_(0)

                r_img = torch.cat((source_rgb.type(torch.FloatTensor), source_depth.type(torch.FloatTensor)), 1).type(
                    torch.FloatTensor).cuda().squeeze_()
                r_img.unsqueeze_(0)

                s_img, s_lab = getInput(path + row["SIM_RGB"], args)

                r_coords, r_embed = tmod(r_img)
                s_coords, s_embed = tmod(s_img)
                embeds_r.append(r_embed.squeeze().cpu().numpy())
                embeds_s.append(s_embed.squeeze().cpu().numpy())

                # ------------------------------------------------------------ Real
                temp_r["gt_x"] = round(s_lab[0][0].item(), 3)
                temp_r["gt_y"] = round(s_lab[0][1].item(), 3)
                temp_r["gt_r"] = adjust_angle(round(s_lab[0][2].item(), 3))

                temp_r["x"] = round(r_coords[0][0].item(), 2)
                temp_r["y"] = round(r_coords[0][1].item(), 2)
                # temp_r["r"] = round(r_coords[0][2].item(), 2)
                temp_r["r"] = adjust_angle(round(r_coords[0][2].item(), 2))

                temp_r["rgb"] = img_to_b64(r_img.cpu())
                temp_r["depth"] = depth_to_b64(r_img.cpu()[:, 3:])

                # ------------------------------------------------------------ SIM

                temp_s["gt_x"] = round(s_lab[0][0].item(), 3)
                temp_s["gt_y"] = round(s_lab[0][1].item(), 3)
                temp_s["gt_r"] = adjust_angle(round(s_lab[0][2].item(), 3))

                temp_s["x"] = round(s_coords[0][0].item(), 2)
                temp_s["y"] = round(s_coords[0][1].item(), 2)
                temp_s["r"] = adjust_angle(round(s_coords[0][2].item(), 2))
                # temp_s["r"] = round(s_coords[0][2].item(), 2)

                temp_s["rgb"] = img_to_b64(s_img.cpu())
                temp_s["depth"] = depth_to_b64(s_img.cpu()[:, 3:])

                temp_r["perf"] = euclidean([temp_r["gt_x"], temp_r["gt_y"]], [temp_r["x"], temp_r["y"]])
                temp_s["perf"] = euclidean([temp_s["gt_x"], temp_s["gt_y"]], [temp_s["x"], temp_s["y"]])

                temp_r["id"] = id
                temp_s["id"] = id

                temp_r["proj"] = embeds_r[id]
                temp_s["proj"] = embeds_s[id]

                stats_r.append(temp_r["perf"])
                stats_s.append(temp_s["perf"])

                res["real_dots"].append(temp_r)
                res["sim_dots"].append(temp_s)

                id += 1

            res["real_perf"] = sum(stats_r) / len(stats_r)
            res["sim_perf"] = sum(stats_s) / len(stats_s)

        embeds_r = umap.UMAP(n_neighbors=30, min_dist=0.3).fit_transform(embeds_r).tolist()
        embeds_s = umap.UMAP(n_neighbors=30, min_dist=0.3).fit_transform(embeds_s).tolist()

        for i in range(len(res["real_dots"])):
            t_real = res["real_dots"][i]
            t_sim = res["sim_dots"][i]

            t_real["proj"] = embeds_r[i]
            t_sim["proj"] = embeds_s[i]

            res["real_dots"][i] = t_real
            res["sim_dots"][i] = t_sim

        return jsonify(res)


def makeHeats(sim, real, model, path):
    img = path
    res = {}
    tmod = model

    output_cam, temp1, _ = getCam(tmod, sim)
    output_cam2, temp2, r_coords = getCam(tmod, real)

    saver = io.BytesIO()

    output_cam = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET)
    output_cam = cv2.cvtColor(output_cam, cv2.COLOR_BGR2RGB)

    Image.fromarray(output_cam, mode="RGB").save(saver, format="JPEG")  # TODO: SWAP FOR CAM

    saver2 = io.BytesIO()

    output_cam2 = cv2.applyColorMap(output_cam2, cv2.COLORMAP_JET)
    output_cam2 = cv2.cvtColor(output_cam2, cv2.COLOR_BGR2RGB)

    Image.fromarray(output_cam2, mode="RGB").save(saver2, format="JPEG")  # TODO: SWAP FOR CAM

    res["activ_sim_sal"] = str(base64.b64encode(saver.getvalue())).replace("b'", "").replace("'", "")
    res["activ_real_sal"] = str(base64.b64encode(saver2.getvalue())).replace("b'", "").replace("'", "")

    dist = featDist(tmod, sim, real, temp1, temp2)

    dist = cv2.applyColorMap(dist, cv2.COLORMAP_JET)
    dist = cv2.cvtColor(dist, cv2.COLOR_BGR2RGB)

    saver3 = io.BytesIO()
    Image.fromarray(dist, mode="RGB").save(saver3, format="JPEG")  # TODO: SWAP FOR CAM

    saver4 = io.BytesIO()
    Image.fromarray(dist, mode="RGB").save(saver4, format="JPEG")  # TODO: SWAP FOR CAM

    res["feat_sim_sal"] = str(base64.b64encode(saver3.getvalue())).replace("b'", "").replace("'", "")
    res["feat_real_sal"] = str(base64.b64encode(saver4.getvalue())).replace("b'", "").replace("'", "")

    n_sq = 6
    sal = occlusion(tmod, real[0], sim[0], n_sq, img).cpu().numpy()

    scale = max(sal[sal > 0].max(), -sal[sal <= 0].min())

    sal = sal / scale * 0.5
    sal += 0.5

    sal = cm.bwr_r(sal)[..., :3]

    sal = np.uint8(sal * 255.0)
    sal = np.reshape(sal, (256, 256, 3))
    sal2 = occlusion(model, sim[0], real[0], n_sq, img).cpu().numpy()

    scale2 = max(sal2[sal2 > 0].max(), -sal2[sal2 <= 0].min())

    sal2 = sal2 / scale2 * 0.5
    sal2 += 0.5

    sal2 = cm.bwr_r(sal2)[..., :3]

    sal2 = np.uint8(sal2 * 255.0)
    sal2 = np.reshape(sal2, (256, 256, 3))

    saver5 = io.BytesIO()
    Image.fromarray(sal, mode="RGB").save(saver5, format="JPEG")

    saver6 = io.BytesIO()
    Image.fromarray(sal2, mode="RGB").save(saver6, format="JPEG")

    res["occlu_sim_sal"] = str(base64.b64encode(saver5.getvalue())).replace("b'", "").replace("'", "")
    res["occlu_real_sal"] = str(base64.b64encode(saver6.getvalue())).replace("b'", "").replace("'", "")

    res["x"] = round(r_coords[0][0].item(), 2)
    res["y"] = round(r_coords[0][1].item(), 2)
    res["r"] = adjust_angle(round(r_coords[0][2].item(), 2))

    del sim
    del real

    return res


def gaussian(depth, val):
    source = depth
    source.add_(-(-1)).div_(1 - (-1) + 1e-5)
    source = source * 255.0

    e_img = cv2.GaussianBlur(source.numpy(), (val, val), 0)
    e_img *= (1 / 255.0)

    source = torch.from_numpy(e_img)

    transform = transforms.Compose(
        [
            transforms.Normalize([0.5], [0.5]),
        ])

    source = transform(source)
    return source.numpy()


def apply(source, key, val):
    if key == "bright":
        source += 1
        source *= (val * 1.3)
        source -= 1
        source.clamp_(min=-1, max=1)
        return source.cpu().numpy()

    if key == "contrast":
        source = source.squeeze().permute((1, 2, 0))
        source.add_(-(-1)).div_(1 - (-1) + 1e-5)
        source = np.uint8(source * 255.0)
        source = np.clip(source, 0, 255)
        source = Image.fromarray(source, 'RGB')

        img_contr_obj = ImageEnhance.Contrast(source)
        e_img = np.array(img_contr_obj.enhance(val * 1.3), dtype=np.float)
        e_img *= (1 / 255.0)

        source = torch.from_numpy(e_img)
        source = source.permute((2, 0, 1))

        transform = transforms.Compose(
            [
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        source = transform(source)
        source.unsqueeze_(0)
        return source.numpy()

    if key == "sepia":
        source = source.squeeze().permute((1, 2, 0))
        source.add_(-(-1)).div_(1 - (-1) + 1e-5)
        source = np.uint8(source * 255.0)
        source = np.clip(source, 0, 255)

        rgb = sepia_np(source, val)
        rgb *= (1 / 255.0)

        source = torch.from_numpy(rgb)
        source = source.permute((2, 0, 1))

        transform = transforms.Compose(
            [
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        source = transform(source)
        source.unsqueeze_(0)
        return source.numpy()

    return source


@app.route('/fake_global', methods=["POST"])
def fake_global():
    meth = request.form['meth']
    prefix = request.form['prefix']
    suffix = request.form['suffix']
    mod = request.form['model']
    traj = request.form['traj']

    with open("heats/traj" + traj + "/model" + mod + "/" + prefix + '_' + str(meth) + "_" + suffix + ".json", "r") as f:
        data = ujson.load(f)

    return jsonify(data)


@app.route('/change_sim', methods=["POST"])
def ch_sim():
    cfg = habitat.get_config("try.yaml").SIMULATOR
    sim = habitat.sims.make_sim(cfg.TYPE, config=cfg)

    params = ujson.loads(request.data)
    temp = params["coords"]
    mod = params["mod"]

    tmod = model

    if mod == "2":
        tmod = model2
    if mod == "3":
        tmod = model3

    del params["coords"]
    del params["mod"]
    for k, v in params.items():
        if k in ("rgb_hfov", "depth_hfov"):
            params[k] = int(v)
        else:
            params[k] = float(v)

    reconfigure_rgbd_camera_in_sim(sim, **params)

    obs = get_observation_from_sim(sim, temp["x"], temp["z"], temp["o"],
                                   temp["y"])

    rgb = obs["rgb"]
    depth = obs["depth"]

    merge = np.append(rgb, depth, axis=2).astype(np.float32)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
        ]
    )

    merge *= (1.0 / 255.0)
    merge = transform(merge)

    merge = torch.unsqueeze(merge, 0)

    output, _ = tmod(merge.cuda())

    res = {}
    res["x"] = round(output[0][0].item(), 2)
    res["y"] = round(output[0][1].item(), 2)
    res["r"] = adjust_angle(round(output[0][2].item(), 2))

    sim.close()

    return jsonify({"rgb": img_to_b64(merge), "depth": depth_to_b64(merge[:, 3:]), "res": res})


def sepia_np(img, k):
    """
    Optimization on the sepia filter using numpy
    """

    lmap = np.matrix([[0.393 + 0.607 * (1 - k), 0.769 - 0.769 * (1 - k), 0.189 - 0.189 * (1 - k)],
                      [0.349 - 0.349 * (1 - k), 0.686 + 0.314 * (1 - k), 0.168 - 0.168 * (1 - k)],
                      [0.272 - 0.349 * (1 - k), 0.534 - 0.534 * (1 - k), 0.131 + 0.869 * (1 - k)]])

    filt = np.array([x * lmap.T for x in img])

    filt[np.where(filt > 255)] = 255

    filt = np.clip(filt, 0, 255)
    return filt


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
