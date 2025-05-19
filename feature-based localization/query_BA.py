import os
import cv2
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import Metashape as ms
from py360convert import e2p
from pyproj import Transformer
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


if __name__ == "__main__":
    matching_rst_fd = 'data\\matching_rst'
    GSV_meta_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\GSV_meta'
    GSV_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\IMG'
    calib_fp = 'data\\query\\calibration\\s7_intrinsics.yml'
    query_fds = [os.path.join(matching_rst_fd, item) for item in os.listdir(matching_rst_fd)]
    GSV_info_all = pd.read_csv('D:\\li2887\\VBN\\Richmond_VBN\\data\\shps\\GSV.csv', index_col='label')

    crs_prj = 6347
    transformer = Transformer.from_crs(4326, crs_prj)
    xys = transformer.transform(GSV_info_all['latitude'], GSV_info_all['longitude'])
    GSV_info_all['E'], GSV_info_all['N'] = xys[0], xys[1]
    xys = np.array(xys).T

    FOV = (74, 50)
    out_size = (1200, 1920)

    if os.path.exists('BA_temp'):
        shutil.rmtree('BA_temp')
    os.mkdir('BA_temp')

    for fd in query_fds:
        print(f'Determining query image {os.path.basename(fd)}')
        global_rst = pd.read_csv(os.path.join(fd, 'global_matching.csv'), index_col='GSV_label')
        local_rst = pd.read_csv(os.path.join(fd, 'local_matching.csv'))
        local_scores = local_rst['local_score']

        if local_scores.max() > 0.1:
            BA_fd = os.path.join('BA_temp', os.path.basename(fd))
            img_fd = os.path.join(BA_fd, 'IMG')
            if os.path.exists(BA_fd):
                shutil.rmtree(BA_fd)
            os.mkdir(BA_fd)
            os.mkdir(img_fd)
            EOP_fp = os.path.join(BA_fd, 'EOP.csv')
            rst_fp = os.path.join(BA_fd, 'BA_rst.csv')
            GSV_info = []

            GSV_lbls = local_rst['GSV_label'][local_scores.argmax()].split(',')

            for lbl in GSV_lbls:
                name = lbl.split('.')[0]
                GSV_info.append([GSV_info_all.loc[name]['E'], GSV_info_all.loc[name]['N'],
                                 GSV_info_all.loc[name]['elevation']])

            GSV_info = np.array(GSV_info)
            x0 = GSV_info[:, 0:3].mean(axis=0)
            dists = cdist(xys, GSV_info[:, 0:2]).min(axis=1)
            ids = np.where(dists <= 15)[0]

            # for p in GSV_info:
            #     plt.arrow(p[0], p[1], 10 * p[3], 10 * p[4], width=0.05)
            #     plt.scatter(p[0], p[1], marker='^', color='green')
            # plt.arrow(x0[0], x0[1], 10 * pm[0], 10 * pm[1], width=0.05, color='red')

            SV_labels, xs, ys, zs, xacc, yacc, zacc = [], [], [], [], [], [], []
            img_fps = [os.path.join(fd, 'template.jpg')]

            for idx in tqdm(ids, desc=f'Projecting GSV'):
                try:
                    lbl = GSV_info_all.index[idx]
                    GSV_name = lbl + '.jpg'
                    u_deg = global_rst.loc[lbl]['u_deg']
                    v_deg = global_rst.loc[lbl]['v_deg']
                    out_prj_fp = os.path.join(img_fd, GSV_name)

                    img = cv2.imread(os.path.join(GSV_fd, GSV_name))
                    # factor = 2
                    # sh, sw, _ = img.shape
                    # down_size = (int(sw / factor), int(sh / factor))
                    # img = cv2.resize(img, down_size)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img[:, :, 0] = gray
                    img[:, :, 1] = gray
                    img[:, :, 2] = gray
                    GSV_prj = e2p(img, FOV, u_deg, v_deg, (1200, 1920))
                    cv2.imwrite(os.path.join(img_fd, GSV_name), GSV_prj)
                    img_fps.append(os.path.join(img_fd, GSV_name))
                    SV_labels.append(GSV_name)
                    xs.append(GSV_info_all.iloc[idx]['E'])
                    ys.append(GSV_info_all.iloc[idx]['N'])
                    zs.append(GSV_info_all.iloc[idx]['elevation'])
                    xacc.append(1)
                    yacc.append(1)
                    zacc.append(1)
                except:
                    continue

            SV_labels.append('template.jpg')
            xacc.append(9 * np.std(xs))
            yacc.append(9 * np.std(ys))
            zacc.append(9 * np.std(zs))
            xs.append(x0[0])
            ys.append(x0[1])
            zs.append(x0[2])
            EOP_df = pd.DataFrame({'label': SV_labels,
                                   'xs': xs,
                                   'ys': ys,
                                   'zs': zs,
                                   'xacc': xacc,
                                   'yacc': yacc,
                                   'zacc': zacc})
            EOP_df.to_csv(EOP_fp, index=False)
            shutil.copy(os.path.join(fd, 'template.jpg'), os.path.join(img_fd, 'template.jpg'))

            doc = ms.Document()
            doc.save(os.path.join(BA_fd, 'BA.psx'))
            chunk = doc.addChunk()
            chunk.crs = ms.CoordinateSystem(f'EPSG::{crs_prj}')
            chunk.addPhotos(img_fps)
            chunk.importReference(EOP_fp, ms.ReferenceFormatCSV, columns='nxyzXYZ',
                                  delimiter=",", skip_rows=1, crs=ms.CoordinateSystem(f'EPSG::{crs_prj}'))

            for camera in chunk.cameras:
                calib = ms.Calibration()
                calib.f = 1290.1988

                # # Split cameras
                # sensor = chunk.addSensor()
                # sensor.label = camera.label
                # sensor.type = camera.sensor.type
                # if 'template' in camera.label:
                #     calib.cx = 938.1260986328125 - 1920 / 2
                #     calib.cy = 616.1744384765625 - 1200 / 2
                #     calib.k1 = -0.21753168106079102
                #     calib.k2 = -0.39623507857322693
                #     calib.p1 = 0.00039196922443807
                #     calib.p2 = -0.00002578284329502
                #     calib.k3 = -0.01959949918091297
                #     calib.k4 = 0.21201662719249725
                # camera.sensor = sensor
                camera.sensor.user_calib = calib
            chunk.matchPhotos(downscale=0, keypoint_limit=40000, tiepoint_limit=10000, generic_preselection=False,
                              reference_preselection=True, filter_stationary_points=True, keep_keypoints=True)
            chunk.alignCameras()
            chunk.exportReference(rst_fp, columns='nuvwdef', delimiter=',')
            doc.save()
            c = 1
        else:
            print(f'Poor local matching, skip')

    ms_prj_fp = 'spherical_align\\temp.psx'
    query_fp = 'BA_temp\\Monroe_Luma_Rectified_Right_300\\IMG\\template.jpg'