import os, hashlib
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch

URL_MAP = {
    "cifar10": "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1",
    "ema_cifar10": "https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1",
    "lsun_bedroom": "https://heibox.uni-heidelberg.de/f/f179d4f21ebc4d43bbfe/?dl=1",
    "ema_lsun_bedroom": "https://heibox.uni-heidelberg.de/f/b95206528f384185889b/?dl=1",
    "lsun_cat": "https://heibox.uni-heidelberg.de/f/fac870bd988348eab88e/?dl=1",
    "ema_lsun_cat": "https://heibox.uni-heidelberg.de/f/0701aac3aa69457bbe34/?dl=1",
    "lsun_church": "https://heibox.uni-heidelberg.de/f/2711a6f712e34b06b9d8/?dl=1",
    "ema_lsun_church": "https://heibox.uni-heidelberg.de/f/44ccb50ef3c6436db52e/?dl=1",
}
CKPT_MAP = {
    "cifar10": "diffusion_cifar10_model/model-790000.ckpt",
    "ema_cifar10": "ema_diffusion_cifar10_model/model-790000.ckpt",
    "lsun_bedroom": "diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "ema_lsun_bedroom": "ema_diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "lsun_cat": "diffusion_lsun_cat_model/model-1761000.ckpt",
    "ema_lsun_cat": "ema_diffusion_lsun_cat_model/model-1761000.ckpt",
    "lsun_church": "diffusion_lsun_church_model/model-4432000.ckpt",
    "ema_lsun_church": "ema_diffusion_lsun_church_model/model-4432000.ckpt",
}
MD5_MAP = {
    "cifar10": "82ed3067fd1002f5cf4c339fb80c4669",
    "ema_cifar10": "1fa350b952534ae442b1d5235cce5cd3",
    "lsun_bedroom": "f70280ac0e08b8e696f42cb8e948ff1c",
    "ema_lsun_bedroom": "1921fa46b66a3665e450e42f36c2720f",
    "lsun_cat": "bbee0e7c3d7abfb6e2539eaf2fb9987b",
    "ema_lsun_cat": "646f23f4821f2459b8bafc57fd824558",
    "lsun_church": "eb619b8a5ab95ef80f94ce8a5488dae3",
    "ema_lsun_church": "fdc68a23938c2397caba4a260bc2445f",
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root=None, check=False):
    if 'church_outdoor' in name:
        name = name.replace('church_outdoor', 'church')
    assert name in URL_MAP
    # Modify the path when necessary
    cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("/atlas/u/tsong/.cache"))
    root = (
        root
        if root is not None
        else os.path.join(cachedir, "diffusion_models_converted")
    )
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def draw_collage(list_img_tensors, list_attri_dicts, save_path):
    """
    Draws a collage of images with their corresponding attributes.

    Args:
        list_img_tensors (list): List of (n_attri + 1) image tensors, each of shape (b, c, w, h).
        list_attri_dicts (list): List of attribute dictionaries, each corresponding to an image tensor.
                                 Each dictionary contains 'n_attri' attributes with tensor values.
        save_path (str): Path to save the resulting collage image.
    """
    # Number of attributes + 1 (original image)
    num_cols = len(list_img_tensors)
    # Number of images in each batch
    batch_size = list_img_tensors[0].shape[0]
    # Get the width and height of images
    _, _, img_w, img_h = list_img_tensors[0].shape

    # Create a figure for the collage
    fig, axes = plt.subplots(nrows=batch_size, ncols=num_cols, figsize=(num_cols * 3, batch_size * 3))
    if batch_size == 1:
        axes = [axes]  # Ensure axes is a list when batch_size is 1

    for b in range(batch_size):
        for col in range(num_cols):
            # Convert the tensor to a PIL image and plot it
            img_tensor = list_img_tensors[col][b]
            img_tensor = torch.clamp(img_tensor * 0.5 + 0.5, 0.0, 1.0)
            img_pil = to_pil_image(img_tensor.cpu())
            ax = axes[b][col] if batch_size > 1 else axes[col]
            ax.imshow(img_pil)
            ax.axis('off')  # Remove axis ticks

            # Prepare attribute labels for the current image
            attr_labels = list_attri_dicts[col].keys()
            attr_values = [list_attri_dicts[col][attr][b].item() for attr in attr_labels]
            label_text = ", ".join(f"{attr}: {int(value)}" for attr, value in zip(attr_labels, attr_values))

            # Set the title with the attribute information above each image
            ax.set_title(label_text, fontsize=8)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
