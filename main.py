import json
import os.path
import shutil
import warnings
import cv2
from QATM import *
import pandas as pd
from tqdm import tqdm
from PIL import Image
import Metashape as ms
from py360convert import e2p
from pyproj import Transformer
from scipy.spatial import KDTree

model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=25, use_cuda=True)
warnings.filterwarnings("ignore")


def template_matching(template_fd, image_fps, FOV, image_size, out_fd, xyz):
    GSV_EOP_fp = '\\'.join(['\\'.join(out_fd.split('\\')[0:2]), 'GSV_EOP.csv'])
    GSV_EOP_file = open(GSV_EOP_fp, 'w')
    qid = 0
    img_fps = []

    for image_idx in tqdm(range(len(image_fps)), desc=' Projecting GSV patch: '):
        image_fp = image_fps[image_idx]
        GSV_source_img = cv2.imread(image_fp)
        dataset = ImageDataset(Path(template_fd), image_fp)
        sh, sw = dataset.h, dataset.w

        for j, sample in enumerate(dataset):
            if j == qid:
                score = run_one_sample(model, sample['template'], sample['image'], sample['image_name'])
                score = score[0]
                dots = np.array(np.where(score == score.max()))

                u = 180 * (dots[1][0] / (sw / 2) - 1)
                if u > 180:
                    u = u - 360
                v = 90 * (1 - dots[0][0] / (sh / 2))
                GSV_prj = e2p(GSV_source_img, FOV, u, v, image_size)
                cv2.imwrite(os.path.join(out_fd, f'{image_idx}.jpg'), GSV_prj)
                img_fps.append(os.path.join(out_fd, f'{image_idx}.jpg'))
                GSV_EOP_file.write(f'{image_idx}.jpg,{xyz[image_idx][0]},{xyz[image_idx][1]},{xyz[image_idx][2]}\n')

        qid = 0 if qid < 2 else qid + 1

    GSV_EOP_file.close()
    return img_fps


if __name__ == "__main__":
    GSV_img_fd = 'E:\\VPN\\proj_file\\PUVBN\\PU_GSV\\GSV\\IMG\\1819'
    query_img_fd = 'E:\\VPN\\proj_file\\PUVBN\\PU_GSV\\Query\\Query_Images'
    block_rst_fp = 'E:\\VPN\\proj_file\\PUVBN\\PU_GSV\\matching_results\\block_matching_outputs.txt'
    GSV_info_fd = 'E:\\VPN\\proj_file\\PUVBN\\PU_GSV\\GSV\\GSV_meta\\1819'
    query_info_fp = 'E:\\VPN\\proj_file\\PUVBN\\PU_GSV\\Query\\Query.csv'
    query_fnames = os.listdir(query_img_fd)
    query_fps = [os.path.join(query_img_fd, fp) for fp in query_fnames]
    block_rst_f = open(block_rst_fp, 'r')
    block_rst_s = block_rst_f.readlines()
    block_size = 3

    query_sample = Image.open(query_fps[0])
    FOV = (69.39, 49.55)
    out_img_size = (query_sample.size[1], query_sample.size[0])
    F = 3028.66307
    query_info = pd.read_csv(query_info_fp)
    transformer = Transformer.from_crs(4326, 6345)

    block_fps, block_query_xyz = [], []
    for i in range(0, len(query_fps), block_size):
        block_fps.append(query_fps[i:i + block_size])
        xy = transformer.transform(query_info['lat'][i:i + block_size].to_list(),
                                   query_info['lng'][i:i + block_size].to_list())
        xyz = [xy[0],
               xy[1],
               query_info['locationAltitude_m_'][i:i + block_size].to_list()]
        block_query_xyz.append(np.array(xyz).T)

    GSV_names, GSV_xyz = [], []
    for fn in os.listdir(GSV_info_fd):
        fp = os.path.join(GSV_info_fd, fn)
        item = json.loads(open(fp, 'r').read())
        GSV_names.append(item['filename'])
        xy = transformer.transform(item['lat'], item['lng'])
        GSV_xyz.append([xy[0], xy[1], item['elevation']])
    GSV_xyz = np.array(GSV_xyz)
    GSV_tree = KDTree(GSV_xyz[:, 0:2])

    BA_temp_fd = 'BA_temp'
    if os.path.exists(BA_temp_fd):
        shutil.rmtree(BA_temp_fd)
    os.mkdir(BA_temp_fd)

    doc = ms.Document()
    doc.save('BA_temp\\BA.psx')
    BA_rst_final_fp = 'BA_rst.csv'
    BA_rst_f = open(BA_rst_final_fp, 'w')
    BA_rst_f.write('query,x,y,z,x_est,y_est,z_est,x_err,y_err,z_err\n')
    BA_rst_f.close()

    for i in range(len(block_fps)):
        block_temp_fd = os.path.join(BA_temp_fd, f'block_{i}')
        template_fd = os.path.join(block_temp_fd, f'template')
        GSV_patch_fd = os.path.join(block_temp_fd, f'GSV_patch')
        query_fd = os.path.join(block_temp_fd, f'query')

        if not os.path.exists(block_temp_fd):
            os.mkdir(block_temp_fd)
        if not os.path.exists(template_fd):
            os.mkdir(template_fd)
        if not os.path.exists(GSV_patch_fd):
            os.mkdir(GSV_patch_fd)
        if not os.path.exists(query_fd):
            os.mkdir(query_fd)

        source_template_fd = (f'E:\\VPN\\proj_file\\PUVBN\\PU_GSV\\matching_results\\'
                              f'block_matching_rst\\block_{i}\\template')
        for fn in os.listdir(source_template_fd):
            shutil.copy(os.path.join(source_template_fd, fn), os.path.join(template_fd, fn))

        query_EOP_fp = os.path.join(block_temp_fd, 'query_EOP.csv')
        GSV_EOP_fp = os.path.join(block_temp_fd, 'GSV_EOP.csv')
        BA_rst_fp = os.path.join(block_temp_fd, 'BA_rst.csv')
        query_EOP_GT = block_query_xyz[i]

        block = block_fps[i]
        block_rst = json.loads(block_rst_s[i])
        PG_dists = np.array(block_rst['PG_dists'])
        best_GSV_idx = GSV_names.index(block_rst['GSV_matched'][PG_dists.mean(axis=1).argmin()])

        best_GSV_ids = GSV_tree.query(GSV_xyz[best_GSV_idx, 0:2], 9)[1]
        best_xyz = GSV_xyz[best_GSV_ids]
        best_GSV_names = [GSV_names[idx] for idx in best_GSV_ids]
        best_GSV_fps = [os.path.join(GSV_img_fd, item) for item in best_GSV_names]

        print(f'Working on block_{i}...')
        try:
            img_fps = template_matching(source_template_fd, best_GSV_fps, FOV, out_img_size, GSV_patch_fd, best_xyz)

            for idx, fp in enumerate(block_fps[i]):
                shutil.copy(block_fps[i][idx], os.path.join(query_fd, f'q{idx}.jpg'))
                img_fps.append(os.path.join(query_fd, f'q{idx}.jpg'))

            chunk = doc.addChunk()
            chunk.crs = ms.CoordinateSystem('EPSG::6345')
            chunk.addPhotos(img_fps)
            chunk.importReference(GSV_EOP_fp, ms.ReferenceFormatCSV, columns='nxyz',
                                  delimiter=",", skip_rows=0, crs=ms.CoordinateSystem('EPSG::6345'))

            qcamera = chunk.cameras[-1]

            for camera in chunk.cameras[:-3]:
                camera.sensor.focal_length = qcamera.sensor.focal_length
                camera.sensor.pixel_height = qcamera.sensor.pixel_height
                camera.sensor.pixel_width = qcamera.sensor.pixel_width

            chunk.matchPhotos(downscale=1, keypoint_limit=40000, tiepoint_limit=10000, generic_preselection=False,
                              reference_preselection=True, filter_stationary_points=True, keep_keypoints=True)
            chunk.alignCameras()
            chunk.exportReference(BA_rst_fp, columns='nuvwdef', delimiter=',')
            camera_est_ = pd.read_csv(BA_rst_fp, header=1)
            camera_est = np.array([camera_est_[-3:]['X_est'].tolist(),
                                   camera_est_[-3:]['Y_est'].tolist(),
                                   camera_est_[-3:]['Z_est'].tolist()]).T

            BA_rst_f = open(BA_rst_final_fp, 'a')
            for j in range(block_size):
                xgt, ygt, zgt = query_EOP_GT[j]
                xe, ye, ze = camera_est[j]
                xerr, yerr, zerr = xe - xgt, ye - ygt, ze - zgt
                BA_rst_f.write(f'block_{i}_{j},{xgt:6f},{ygt:6f},{zgt:6f},{xe:6f},{ye:6f},{ze:6f},'
                               f'{xerr:6f},{yerr:6f},{zerr:6f}\n')
            BA_rst_f.close()
            doc.remove(chunk)
            doc.save()
        except:
            print(' failed BA')
        c = 1
