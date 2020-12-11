import os
import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader
import utils
import time
from torchvision.utils import save_image
from datasets import preprocess, postprocess
import math

class Trainer(object):

    def __init__(self, model, optim, scheduler=None, training_set=None, valid_set=None, args=None, cuda=None):

        # set path and date
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")

        self.out_root = os.path.join(args.out_root, "log_" + date)
        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)

        self.checkpoints_dir = os.path.join(self.out_root, "save_models")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.save_dir = os.path.join(self.out_root, "generated_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # optim
        self.optim = optim
        self.lr_update_threshold = self.optim.param_groups[0]['lr'] / 100.0
        self.scheduler = scheduler


        # gradient bound
        self.max_grad_clip = args.max_grad_clip
        self.max_grad_norm = args.max_grad_norm

        # data
        self.batch_size = args.batch_size

        self.train_data_loader = None
        if training_set is not None:
            self.train_data_loader = DataLoader(training_set,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                drop_last=True)

        self.valid_data_loader = None
        if valid_set is not None:
            self.valid_data_loader = DataLoader(valid_set,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                drop_last=False)

        self.num_epochs = args.num_epochs
        self.global_step = args.steps
        self.n_bits = args.n_bits

        # infer parameters
        self.nll_gap = args.nll_gap
        self.inference_gap = args.inference_gap
        self.save_gap = args.save_gap
        self.valid_gap = args.valid_gap
        self.n_samples = args.n_samples
        self.sample_each_row = args.sample_each_row

        # device
        self.cuda = args.cuda
        self.device = args.device
        self.num_gpus = args.num_gpus
        self.data_parallel = args.data_parallel


        # model
        if self.data_parallel and self.num_gpus > 1:
            print("Do data Parallel")
            model = torch.nn.DataParallel(model)

        self.model = model.to(self.device)



    def valid(self):
        print ("start valid")
        start = time.time()
        self.model.eval()
        loss_list = []
        with torch.no_grad():
            for i_batch, batch in enumerate(self.valid_data_loader):
                x = batch[0]
                x = x.to(self.device)

                x = preprocess(x, self.n_bits)
                # forward
                z, nll = self.model(x=x)

                # loss
                loss = torch.mean(nll)
                loss_list.append(loss.data.cpu().item())


        mean_loss = np.mean(loss_list)

        with open(os.path.join(self.out_root, "valid_nll.txt"), "a") as f:
            f.write("{} \t {:.5f}".format(self.global_step, mean_loss))
            f.write("\n")

        self.model.train()

        print ("end valid")
        print ("Valid elapsed time:{:.2f}".format(time.time() - start))


    def train(self):

        # set to training state
        self.model.train()

        # init glow
        starttime = time.time()

        # run
        num_batchs = len(self.train_data_loader)
        total_its = self.num_epochs * num_batchs
        for epoch in range(self.num_epochs):
            mean_nll = 0.0
            #print(self.optim.param_groups[0]['lr'])
            for i_batch, batch in enumerate(self.train_data_loader):

                x = batch[0]
                x = x.to(self.device)
                x = preprocess(x, self.n_bits)

                # forward
                z, nll = self.model(x=x)

                # loss
                loss = torch.mean(nll)
                mean_nll = mean_nll + loss.data

                currenttime = time.time()
                elapsed = currenttime - starttime
                print("Iteration: {}/{} \t Elapsed time: {:.2f} \t Loss:{:.5f}".format(self.global_step, total_its, elapsed, loss.data))
                if self.global_step % self.nll_gap == 0:
                    with open(os.path.join(self.out_root, "NLL.txt"), "a") as nll_file:
                        nll_file.write("{} \t {:.2f}\t {:.5f}".format(self.global_step, elapsed, loss.data) + "\n")

                # backward
                self.model.zero_grad()
                self.optim.zero_grad()
                loss.backward()

                # operate grad
                grad_norm = 0
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)


                # step
                if math.isnan(grad_norm):
                    continue
                else:
                    self.optim.step()


                # checkpoint
                if self.global_step % self.save_gap == 0 and self.global_step > 0:
                    utils.save_model(self.model, self.optim, self.scheduler, self.checkpoints_dir, self.global_step)


                if self.global_step % self.valid_gap == 0 and self.global_step > 0:
                    if self.valid_data_loader is not None:
                        self.valid()



                if self.global_step % self.inference_gap == 0 and self.global_step > 0:
                    self.save_inverse_images(x, z)
                    self.save_sample_images(self.n_samples,self.sample_each_row, 1.0)


                self.global_step = self.global_step + 1

            current_lr = self.optim.param_groups[0]['lr']
            if self.scheduler is not None and current_lr > self.lr_update_threshold:
                self.scheduler.step()


            mean_nll = float(mean_nll / float(num_batchs))
            with open(os.path.join(self.out_root, "Epoch_NLL.txt"), "a") as f:
                currenttime = time.time()
                elapsed = currenttime - starttime
                f.write("{} \t {:.2f}\t {:.5f}".format(epoch, elapsed, mean_nll) + "\n")




    def save_sample_images(self,n_samples=20, sample_each_row=5, eps_std=1.0):
        print ("Start sampling")
        start = time.time()
        assert n_samples % sample_each_row == 0, "cannot arrange the samples"
        samples = []

        for i in range(0, n_samples):

            s,_ = self.model(z=None, eps_std=eps_std, reverse=True)
            s = s.detach().cpu()
            s = postprocess(s, self.n_bits)
            samples.append(s)

        n_rows = int(n_samples / sample_each_row)
        i = 0
        output = None
        for r in range(0, n_rows):
            row = None
            for s in range(0, sample_each_row):
                if row is None:
                    row = samples[i]
                    i = i + 1
                    continue
                else:
                    row = torch.cat((row, samples[i]), dim=2)
                    i = i + 1

            if output is None:
                output = row
                continue
            else:
                output = torch.cat((output, row), dim=3)

        save_image(output, os.path.join(self.save_dir, "sample-{}.jpg".format(self.global_step)))
        print("End sampling")
        print("Elapsed time:{:.2f}".format(time.time() - start))


    def save_inverse_images(self, x, z):
        print ("Start sample inverse")
        start = time.time()
        assert x.size(0) == z.size(0), "sizes are not the consistent"

        x = postprocess(x, self.n_bits)
        img,_ = self.model(z=z, eps_std=1.0, reverse=True)
        img = img.detach().cpu()
        x = x.detach().cpu()
        img = postprocess(img, self.n_bits)

        output = None
        for i in range(0, min(self.batch_size, 10)):
            row = torch.cat((x[i], img[i]), dim=1)
            if output is None:
                output = row
            else:
                output = torch.cat((output, row), dim=2)

        save_image(output, os.path.join(self.save_dir, "img-{}.jpg".format(self.global_step)))
        print ("End sample inverse")
        print ("Elapsed time: {:.5f}".format(time.time() - start))



class Inferencer(object):

    def __init__(self, model, dataset, args):

        self.out_root = args.out_root
        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)

        self.sample_root = os.path.join(self.out_root, "samples")
        if not os.path.exists(self.sample_root):
            os.makedirs(self.sample_root)

        # cuda
        self.cuda = args.cuda

        # model
        self.model = model



        # data
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.data_loader = None
        if dataset is not None:
            self.data_loader = DataLoader(dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          drop_last=False)

        self.n_bits = args.n_bits


    def Inference(self):
        start = time.time()
        loss_list = []
        num_batchs = len(self.data_loader)
        for i_batch, batch in enumerate(self.data_loader):

            x = batch[0]

            if self.cuda:
                x = x.cuda()

            x = preprocess(x, n_bits=self.n_bits)


            # forward
            z, nll = self.model(x=x)

            # loss
            loss = torch.mean(nll)
            loss_list.append(loss.data.cpu().item())
            print("batch: {}/{}, elapsed time:{:.2f}, loss:{:.5f}".format(i_batch, num_batchs, time.time()-start, loss.data.cpu().item()))


        mean_loss = np.mean(loss_list)

        with open(os.path.join(self.out_root, "test_nll.txt"), "w") as f:
            f.write("NLL: {:.5f}".format(mean_loss))


    def Sample(self,n_samples=64, sample_each_row=8, eps_std=1.0):

        assert n_samples % sample_each_row == 0, "cannot arrange the samples"

        i = 0
        row_id = 0
        while i < n_samples:
            print ("sample: {}\{}".format(i, n_samples))
            row = None
            for r in range(0, sample_each_row):
                s,_ = self.model(z=None, eps_std=eps_std, reverse=True)
                s = postprocess(s, n_bits=self.n_bits)
                i = i + 1
                if row is None:
                    row = s
                else:
                    row = torch.cat((row, s), dim=3)
            save_image(row, os.path.join(self.sample_root, "sample-{}.png".format(row_id)))
            row_id = row_id + 1
