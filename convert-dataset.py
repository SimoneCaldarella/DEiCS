import cv2 
import os
from pathlib import Path
from tqdm import tqdm

# final_size = (640, 192)

final_size = (720, 405)

# final_size = (1280, 720)

# final_size = (256, 144)

if __name__ == '__main__':

    if len(os.sys.argv) != 1:
        raise RuntimeError("please specify the relative path containing the dataset as first and unique argument")

    root_pth = 'dataset' if (len(os.sys.argv) <= 1) else os.sys.argv[1]
    new_root_pth = 'reduced_dataset' if (len(os.sys.argv) <= 1) else Path(*Path(os.sys.argv[1]).parts[:-1], 'reduced_dataset')

    pbar = tqdm(total=sum([len([el for el in l if (el.endswith(".png")) or (el.endswith('.bmp'))]) for _, _, l in os.walk(root_pth)]), desc=f'Converting dataset into size {final_size}')
    for root, d, fnames in os.walk(root_pth):
        for fname in fnames:
            if fname.endswith('.png') or fname.endswith('.bmp'):
                os.makedirs(Path(new_root_pth, *Path(root).parts[1:]), exist_ok=True)

                if not ("depth" in fname):
                    old_pth = os.path.join(root, fname)
                    new_pth = str(Path(new_root_pth, *Path(old_pth).parts[1:]))
                    img = cv2.imread(old_pth)
                    new_img = cv2.resize(img, final_size)
                    cv2.imwrite(new_pth, new_img)
                else:
                    old_pth = os.path.join(root, fname)
                    new_pth = str(Path(new_root_pth, *Path(old_pth).parts[1:]))
                    img = cv2.imread(old_pth)
                    cv2.imwrite(new_pth, img)
                
                pbar.update()
