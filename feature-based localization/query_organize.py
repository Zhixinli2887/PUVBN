import os
import pathlib
from PIL import Image


if __name__ == "__main__":
    in_fd = 'E:\\VPN\\RichmondCollect_filtered\\RichmondCollect_filtered'
    dst_fd = 'query'
    fps_all = list(pathlib.Path(in_fd).rglob("*.tiff"))

    if not os.path.exists('query\\s7'):
        os.mkdir('query\\s7')

    if not os.path.exists('query\\s27'):
        os.mkdir('query\\s27')

    for fp in fps_all:
        split = fp.__fspath__().split('\\')
        place, sensor = split[4].split('_')
        lf, deg = split[5], split[6]
        out_fd = os.path.join(dst_fd, sensor)
        out_fname = f'{place}_{lf}_{deg}.png'
        out_fp = os.path.join(out_fd, out_fname)

        img = Image.open(fp)
        png_image = img.convert("RGB")

        if os.path.exists(out_fp):
            out_fp = out_fp.replace('.png', '2.png')

        png_image.save(out_fp)

