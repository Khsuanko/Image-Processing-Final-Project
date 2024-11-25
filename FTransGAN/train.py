import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch


def main():
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    global_step = 0  # Total batches trained
    writer = SummaryWriter("./logs_tensorboard")  # Create a tensorboard visualizer
    print('To view training results and loss plots, run `tensorboard --logdir=./logs_tensorboard`.')

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        print(f'Epoch: {epoch}')
        epoch_start_time = time.time()  # timer for entire epoch
        with tqdm(total=len(dataset.dataloader), desc=f'Epoch {epoch}/{opt.n_epochs}', unit='batch') as pbar:
            for i, data in enumerate(dataset.dataloader):  # inner loop within one epoch
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                total_iters += opt.batch_size
                global_step += 1

                # Log loss every batch
                losses = model.get_current_losses()
                writer.add_scalar('Batch Loss G_GAN', losses['G_GAN'], global_step)
                writer.add_scalar('Batch Loss G_L1', losses['G_L1'], global_step)
                writer.add_scalar('Batch Loss D_content', losses['D_content'], global_step)
                writer.add_scalar('Batch Loss D_style', losses['D_style'], global_step)

                # Log images every 100 batches
                if global_step % 1 == 0:
                    # print(f'Batch: {i + 1}/{len(dataset.dataloader)} | Losses: {losses} | Global Step: {global_step}')
                    model.compute_visuals()
                    visuals = model.get_current_visuals()

                    gt_image = visuals['gt_images'][0]
                    generated_image = torch.clamp(visuals['generated_images'][0], 0.0, 1.0)
                    image_pair = torch.cat((gt_image, generated_image), dim=2)
                    writer.add_image('GT/Generated Image Pair', image_pair, global_step, dataformats='CHW')

                    # for key, images in visuals.items():
                    #     image_pair = torch.cat((first_image, gt_image), dim=2)
                    #     print(type(key), type(values))
                    #     print(key, values.shape)
                    #     writer.add_images(key, values, global_step)

                # Save model every 500 step
                if global_step % 500 == 0:
                    print('saving the latest model (epoch %d, global_step %d)' % (epoch, global_step))
                    save_suffix = f'step_{global_step:09d}' if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                pbar.update()

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, global_step %d' % (epoch, global_step))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.


if __name__ == '__main__':
    main()
