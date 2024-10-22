import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch import sqrt
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

from functions.denoising import compute_alpha
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        if config.model.type == "conditional":  # currently only support categorical conditioning
            conditioning_vars = {attr: {'type': 'categorical', 'num_classes': num} for attr, num in
                                 zip(config.data.attrs, config.data.attrs_cate_num)}
            model = Model(config, conditioning_vars)
        else:
            model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            print("Resuming training from checkpoint with START EPOCH: {}, STEP: {}".format(start_epoch, step))
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas
                # Send every values in dict y to device
                if isinstance(y, dict):
                    y = {k: v.to(self.device) for k, v in y.items()}

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b, y)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True, mid_num_timesteps=None, y=None):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                if mid_num_timesteps is not None and mid_num_timesteps < self.num_timesteps:
                    seq = range(0, mid_num_timesteps, skip)
                else:
                    seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                if mid_num_timesteps is not None:
                    raise NotImplementedError
                else:
                    seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            if y is not None:
                xs = generalized_steps(x, seq, model, self.betas, y=y, eta=self.args.eta)
            else:
                xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):

        ## WORKING CODE ##
        if self.config.model.type == "conditional":  # currently only support categorical conditioning
            conditioning_vars = {attr: {'type': 'categorical', 'num_classes': num} for attr, num in
                                 zip(self.config.data.attrs, self.config.data.attrs_cate_num)}
            model = Model(self.config, conditioning_vars)
        else:
            model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            raise NotImplementedError

        model.eval()
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 5
        config.sampling.batch_size = total_n_samples
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                    range(n_rounds), desc="Generating image samples for test."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                y = {
                    'Male': torch.tensor([0, 0, 0, 0, 0]).long().to(self.device),  # tensor of shape [batch_size], dtype=torch.long
                    'Young': torch.tensor([1, 1, 1, 1, 1]).long().to(self.device),
                    'Gray_Hair': torch.tensor([1, 1, 1, 1, 1]).long().to(self.device)
                }
                x = self.sample_image(x, model, y=y)
                # Clone x to keep the original data, detach x to avoid backpropagation
                x_t = x.clone().to(self.device).detach()
                x = inverse_data_transform(config, x)
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1
                exit(1)

                # Send it to half-noisy
                mid_num_timesteps = self.num_timesteps - self.num_timesteps // 2
                for t in tqdm.tqdm(range(0, mid_num_timesteps - 1)):
                    t_tensor = torch.ones(config.sampling.batch_size) * t
                    t_tensor = t_tensor.to(self.device)
                    alpha_tp1 = torch.tensor(compute_alpha(self.betas, (t_tensor + 1).long())).to(self.device)
                    alpha_t = torch.tensor(compute_alpha(self.betas, t_tensor.long())).to(self.device)
                    eplison_theta = model(x_t, t_tensor)
                    x_tp1 = torch.sqrt(alpha_tp1 / alpha_t) * (x_t - torch.sqrt(1 - alpha_t) * eplison_theta) + torch.sqrt(1 - alpha_tp1) * eplison_theta
                    x_t = x_tp1

                y = {
                    'Male': torch.tensor([0, 1, 1]).long().to(self.device),  # tensor of shape [batch_size], dtype=torch.long
                    'Young': torch.tensor([0, 1, 1]).long().to(self.device),
                    'Gray_Hair': torch.tensor([1, 0, 0]).long().to(self.device)
                }

                x = self.sample_image(x_t, model, mid_num_timesteps=mid_num_timesteps, y=y)
                x = inverse_data_transform(config, x)
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1


    def test2(self):
        ## NEW ##
        import os
        import torch
        from PIL import Image
        from torchvision import transforms

        # Define directories
        image_dir = "celebahq_subset/images/"
        attribute_file = "celebahq_subset/CelebAMask-HQ-attribute-anno.txt"

        # Define the attributes of interest
        attributes_of_interest = ["Male", "Young", "Gray_Hair"]

        # Define transformations for the images
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Parse the attribute file and filter only those that have corresponding images and selected attributes
        def parse_attributes(attribute_file, image_dir, attributes_of_interest):
            available_images = set(os.listdir(image_dir))
            with open(attribute_file, 'r') as f:
                lines = f.readlines()
            attri_names = lines[1].strip().split()

            # Map attribute names to indices
            attri_indices = {attr: attri_names.index(attr) for attr in attributes_of_interest}

            attri_data = {}
            for line in lines[2:]:
                parts = line.strip().split()
                image_name = parts[0]
                if image_name in available_images:  # Only include if the image is available
                    # Extract only the specified attributes
                    attributes = [int(parts[attri_indices[attr] + 1]) for attr in attributes_of_interest]
                    attributes = [0 if attr == -1 else 1 for attr in attributes]  # Change -1 to 0
                    attri_data[image_name] = dict(zip(attributes_of_interest, attributes))
            return attri_data

        # Create a batch of images and corresponding batched attribute dictionary
        def create_batch(image_dir, attribute_data):
            image_names = list(attribute_data.keys())
            batch_imgs = []
            batch_attri_dict = {key: [] for key in attribute_data[image_names[0]]}

            for img_name in image_names:
                # Load and transform the image
                img_path = os.path.join(image_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                batch_imgs.append(img_tensor)

                # Get the attributes for the image and append to corresponding list
                attri_dict = attribute_data[img_name]
                for key, value in attri_dict.items():
                    batch_attri_dict[key].append(value)

            # Stack the images into a tensor
            batch_imgs = torch.stack(batch_imgs)

            # Convert attribute lists into tensors
            for key in batch_attri_dict:
                batch_attri_dict[key] = torch.tensor(batch_attri_dict[key], dtype=torch.long)

            return batch_imgs, batch_attri_dict

        # Parse the attribute file with filtering
        attribute_data = parse_attributes(attribute_file, image_dir, attributes_of_interest)
        # Create a batch of images and corresponding batched attribute dictionary
        batch_imgs, batch_ys = create_batch(image_dir, attribute_data)
        # Move images and attributes to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_imgs = batch_imgs.to(device)
        list_img_tensors = [batch_imgs]
        batch_ys = {key: value.to(device) for key, value in batch_ys.items()}
        # Initialize the list with the original batch_ys
        list_attri_dicts = [batch_ys]
        # Iterate through each attribute in the batch_ys
        for attr in batch_ys.keys():
            # Create a new dictionary for the modified attributes
            modified_batch_ys = {key: value.clone() for key, value in batch_ys.items()}
            # Flip the value of the current attribute
            modified_batch_ys[attr] = 1 - modified_batch_ys[attr]
            # Append the modified attributes to the list
            list_attri_dicts.append(modified_batch_ys)

        if self.config.model.type == "conditional":  # currently only support categorical conditioning
            conditioning_vars = {attr: {'type': 'categorical', 'num_classes': num} for attr, num in
                                 zip(self.config.data.attrs, self.config.data.attrs_cate_num)}
            model = Model(self.config, conditioning_vars)
        else:
            model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            raise NotImplementedError

        model.eval()
        config = self.config
        config.sampling.batch_size = batch_imgs.size(0)
        for _i in range(1, len(list_attri_dicts)):
            with torch.no_grad():
                x_t = batch_imgs.clone().to(self.device)
                # Send it to half-noisy
                mid_num_timesteps = self.num_timesteps // 4
                for t in tqdm.tqdm(range(0, mid_num_timesteps - 1)):
                    t_tensor = torch.ones(config.sampling.batch_size) * t
                    t_tensor = t_tensor.to(self.device)
                    alpha_tp1 = torch.tensor(compute_alpha(self.betas, (t_tensor + 1).long())).to(self.device)
                    alpha_t = torch.tensor(compute_alpha(self.betas, t_tensor.long())).to(self.device)
                    eplison_theta = model(x_t, t_tensor)
                    x_tp1 = torch.sqrt(alpha_tp1 / alpha_t) * (x_t - torch.sqrt(1 - alpha_t) * eplison_theta) + torch.sqrt(
                        1 - alpha_tp1) * eplison_theta
                    x_t = x_tp1
                # Denoise the image
                x = self.sample_image(x_t, model, mid_num_timesteps=mid_num_timesteps, y=list_attri_dicts[_i])
                list_img_tensors.append(x.clone().to("cpu"))
        # Make sure list_img_tensors, list_attri_dicts are all on CPU
        list_img_tensors = [img_tensor.to("cpu") for img_tensor in list_img_tensors]
        list_attri_dicts = [{key: value.to("cpu") for key, value in attri_dict.items()} for attri_dict in list_attri_dicts]
        # Define a path to save the resulting collage
        save_path = "collage_demo.png"
        from functions.ckpt_util import draw_collage
        # Call the function to create the collage
        draw_collage(list_img_tensors, list_attri_dicts, save_path)
