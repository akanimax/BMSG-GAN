# BMSG-GAN 
## Official code repository for the paper "MSG-GAN: Multi-Scale Gradient GAN for Stable Image Synthesis" [[arXiv]](https://arxiv.org/abs/1903.06048)

## SageMaker
Training is now supported on AWS SageMaker. Please read https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html 

<p align="center">
<img alt="Flagship Diagram" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/flagship.png" />
<br>
</p>

### **MSG-GAN**: Multi-Scale Gradient GAN for Stable Image Synthesis

_Abstract:_ <br>
While Generative Adversarial Networks (GANs) have seen huge 
successes in image synthesis tasks, they are notoriously difficult 
to use, in part due to instability during training. One commonly 
accepted reason for this instability is that gradients passing from 
the discriminator to the generator can quickly become uninformative, 
due to a learning imbalance during training. In this work, we propose 
the Multi-Scale Gradient Generative Adversarial Network (MSG-GAN), 
a simple but effective technique for addressing this problem which 
allows the flow of gradients from the discriminator to the generator 
at multiple scales. This technique provides a stable approach for 
generating synchronized multi-scale images. We present a 
very intuitive implementation of the mathematical MSG-GAN 
framework which uses the concatenation operation in the 
discriminator computations. We empirically validate the effect 
of our MSG-GAN approach through experiments on the CIFAR10 and 
Oxford102 flowers datasets and compare it with other relevant 
techniques which perform multi-scale image synthesis. In addition, 
we also provide details of our experiment on CelebA-HQ dataset 
for synthesizing 1024 x 1024 high resolution images.


<p align="center">
<img alt="Training time-lapse gif" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/explanatory_video_2.gif" />
<br>
</p>

An explanatory training time-lapse video/gif for the MSG-GAN. The higher resolution layers initially display plain colour blocks but eventually (very soon) the training penetrates all layers and then they all work in unison to produce better samples. Please observe the first few secs of the training, where the face like blobs appear in a sequential order from the lowest resolution to the highest resolution. 

### Multi-Scale Gradients architecture
<p align="center">
<img alt="proposed MSG-GAN architecture" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/architecture.png"
width=90% />
</p>

<p>
The above figure describes the architecture of MSG-GAN for 
generating synchronized multi-scale images. Our method is 
based on the architecture proposed in proGAN, 
but instead of a progressively growing training scheme, 
includes connections from the intermediate
layers of the generator to the intermediate layers of the 
discriminator. The multi-scale images input to 
the discriminator are converted into spatial 
volumes which are concatenated with the corresponding 
activation volumes obtained from the main path of 
convolutional layers.
</p> <br>

<p>
For the discrimination process, appropriately downsampled 
versions of the real images are fed to corresponding layers 
of the discriminator as shown in the diagram (from above).
</p> <br>

<p align="center">
<img alt="synchronization explanation" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/synchronization.png"
     width=80% />
</p>
<br>

Above figure explains how, during training, all the layers 
in the MSG-GAN first synchronize colour-wise and subsequently 
improve the generated images at various scales. 
The brightness of the images across all layers (scales) 
synchronizes eventually

### Running the Code
**Please note to use value of `learning_rate=0.003` for 
both G and D for all experiments for best results**. The model 
is quite robust and converges to a very similar FID or IS 
very quickly even for different learning rate settings.
Please use the `relativistic-hinge` as the loss function 
(set as default) for training.

Start the training by running the `train.py` script in the `sourcecode/` 
directory. Refer to the following parameters for tweaking for your own use:

    -h, --help            show this help message and exit
      --generator_file GENERATOR_FILE
                            pretrained weights file for generator
      --generator_optim_file GENERATOR_OPTIM_FILE
                            saved state for generator optimizer
      --shadow_generator_file SHADOW_GENERATOR_FILE
                            pretrained weights file for the shadow generator
      --discriminator_file DISCRIMINATOR_FILE
                            pretrained_weights file for discriminator
      --discriminator_optim_file DISCRIMINATOR_OPTIM_FILE
                            saved state for discriminator optimizer
      --images_dir IMAGES_DIR
                            path for the images directory
      --folder_distributed FOLDER_DISTRIBUTED
                            whether the images directory contains folders or not
      --flip_augment FLIP_AUGMENT
                            whether to randomly mirror the images during training
      --sample_dir SAMPLE_DIR
                            path for the generated samples directory
      --model_dir MODEL_DIR
                            path for saved models directory
      --loss_function LOSS_FUNCTION
                            loss function to be used: standard-gan, wgan-gp,
                            lsgan,lsgan-sigmoid,hinge, relativistic-hinge
      --depth DEPTH         Depth of the GAN
      --latent_size LATENT_SIZE
                            latent size for the generator
      --batch_size BATCH_SIZE
                            batch_size for training
      --start START         starting epoch number
      --num_epochs NUM_EPOCHS
                            number of epochs for training
      --feedback_factor FEEDBACK_FACTOR
                            number of logs to generate per epoch
      --num_samples NUM_SAMPLES
                            number of samples to generate for creating the grid
                            should be a square number preferably
      --checkpoint_factor CHECKPOINT_FACTOR
                            save model per n epochs
      --g_lr G_LR           learning rate for generator
      --d_lr D_LR           learning rate for discriminator
      --adam_beta1 ADAM_BETA1
                            value of beta_1 for adam optimizer
      --adam_beta2 ADAM_BETA2
                            value of beta_2 for adam optimizer
      --use_eql USE_EQL     Whether to use equalized learning rate or not
      --use_ema USE_EMA     Whether to use exponential moving averages or not
      --ema_decay EMA_DECAY
                            decay value for the ema
      --data_percentage DATA_PERCENTAGE
                            percentage of data to use
      --num_workers NUM_WORKERS
                            number of parallel workers for reading files

##### Sample Training Run
For training a network at resolution `256 x 256`, 
use the following arguments:

    $ python train.py --depth=7 \ 
                      --latent_size=512 \
                      --images_dir=<path to images> \
                      --sample_dir=samples/exp_1 \
                      --model_dir=models/exp_1

Set the `batch_size`, `feedback_factor` and 
`checkpoint_factor` accordingly.
We used 2 Tesla V100 GPUs of the 
DGX-1 machine for our experimentation.

### Generated samples on different datasets

<p align="center">
     <b> <b> :star: [NEW] :star: </b> CelebA HQ [1024 x 1024] (30K dataset)</b> <br>
     <img alt="CelebA-HQ" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/HQ_faces_sheet.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> <b> :star: [NEW] :star: </b> Oxford Flowers (improved samples) [256 x 256] (8K dataset)</b> <br>
     <img alt="oxford_big" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/flowers_sheet.png"
          width=80% />
     <img alt="oxford_variety" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/variety_flowers_sheet.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> CelebA HQ [256 x 256] (30K dataset)</b> <br>
     <img alt="CelebA-HQ" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/CelebA-HQ_sheet.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> LSUN Bedrooms [128 x 128] (3M dataset) </b> <br>
     <img alt="lsun_bedrooms" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/Bedrooms_sheet_new.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> CelebA [128 x 128] (200K dataset) </b> <br>
     <img alt="CelebA" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/faces_sheet.png"
          width=80% />
</p>
<br>

### Synchronized all-res generated samples
<p align="center">
     <b> Cifar-10 [32 x 32] (50K dataset)</b> <br>
     <img alt="cifar_allres" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/CIFAR10_allres_sheet.png"
          width=80% />
</p>
<br>

<p align="center">
     <b> Oxford-102 Flowers [256 x 256] (8K dataset)</b> <br>
     <img alt="flowers_allres" src="https://github.com/akanimax/BMSG-GAN/blob/master/diagrams/FLowers_allres_sheet.png"
          width=80% />
</p>
<br>

### Cite our work
    @article{karnewar2019msg,
      title={MSG-GAN: Multi-Scale Gradient GAN for Stable Image Synthesis},
      author={Karnewar, Animesh and Wang, Oliver and Iyengar, Raghu Sesha},
      journal={arXiv preprint arXiv:1903.06048},
      year={2019}
    }

### Other Contributors :smile:

<p align="center">
     <b> Cartoon Set [128 x 128] (10K dataset) by <a href="https://github.com/huangzh13">@huangzh13</a> </b> <br>
     <img alt="Cartoon_Set" src="https://github.com/huangzh13/BMSG-GAN/blob/dev/diagrams/cartoonset_sheet.png"
          width=80% />
</p>
<br>

### Thanks
Please feel free to open PRs here if 
you train on other datasets using this architecture. 
<br>

Best regards, <br>
@akanimax :)
