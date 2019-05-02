from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse, random
import os, sys, subprocess

import helpers, utils, PostProcessing

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append("models")
from FC_DenseNet_Tiramisu import build_fc_densenet
from Encoder_Decoder import build_encoder_decoder
from RefineNet import build_refinenet
from FRRN import build_frrn
from MobileUNet import build_mobile_unet
from PSPNet import build_pspnet
from GCN import build_gcn
from DeepLabV3 import build_deeplabv3
from DeepLabV3_plus import build_deeplabv3_plus
from AdapNet import build_adaptnet

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train", "test", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--class_balancing', type=str2bool, default=False, help='Whether to use median frequency class weights to balance the classes in the loss')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--image_folder', type=str, default="Predict", help='The directory of images you want to predict on. Only valid in "predict_folder" mode.')
parser.add_argument('--ratio_lock', type=str2bool, default=False, help='During prediction, whether or not to preserve aspect ratio for output images')
parser.add_argument('--pred_center_crop', type=str2bool, default=True, help='During prediction, whether or not to center each crop')
parser.add_argument('--pred_downsample', type=str2bool, default=True, help='During prediction, whether to downsample or crop')
parser.add_argument('--post_processing', type=int, default=0, help='What kind of Post Processing to do. (0: None, 1:Rail, 2:MARS, 3:MARS with Horizon)')
parser.add_argument('--generate_images', type=str2bool, default=True, help='During prediction, whether or not to generate images')
parser.add_argument('--rail_sens', type=int, default=300, help='Threshold for eliminating noise when processing rail images.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=10, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1. For example, .1 represents a max brightness change of 10% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--learn_rate', type=float, default=0.0001, help='Specify the learning rate.')
parser.add_argument('--droplets_num', type=int, default=5, help='How many droplets should be added to images')
parser.add_argument('--droplets_size', type=int, default=40, help='How large should droplets be')
parser.add_argument('--removal', type=int, default=200, help='Whether or not the top of the image should be assumed sky')
parser.add_argument('--theta', type=float, default=1.5, help='Controls how flat the probabilities will be (for visualization only). Higher is more binary.')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. Currently supports:\
    FC-DenseNet56, FC-DenseNet67, FC-DenseNet103, Encoder-Decoder, Encoder-Decoder-Skip, RefineNet-Res50, RefineNet-Res101, RefineNet-Res152, \
    FRRN-A, FRRN-B, MobileUNet, MobileUNet-Skip, PSPNet-Res50, PSPNet-Res101, PSPNet-Res152, GCN-Res50, GCN-Res101, GCN-Res152, DeepLabV3-Res50 \
    DeepLabV3-Res101, DeepLabV3-Res152, DeepLabV3_plus-Res50, DeepLabV3_plus-Res101, DeepLabV3_plus-Res152, AdapNet, custom')
args = parser.parse_args()

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path,1), cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB) # TODO: This doesn't work with .tif for some reason
    #image = cv2.imread(path,-1)
    return image

def download_checkpoints(model_name):
    subprocess.check_output(["python", "get_pretrained_checkpoints.py", "--model=" + model_name])

# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
train_writer = tf.summary.FileWriter( './logs/1/train ', sess.graph)

# Get the selected model.
# Some of them require pre-trained ResNet

if "Res50" in args.model and not os.path.isfile("models/resnet_v2_50.ckpt"):
    download_checkpoints("Res50")
if "Res101" in args.model and not os.path.isfile("models/resnet_v2_101.ckpt"):
    download_checkpoints("Res101")
if "Res152" in args.model and not os.path.isfile("models/resnet_v2_152.ckpt"):
    download_checkpoints("Res152")

# Compute your softmax cross entropy loss
print("Preparing the model ...")
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network = None
init_fn = None

if args.model == "FC-DenseNet56" or args.model == "FC-DenseNet67" or args.model == "FC-DenseNet103":
    network = build_fc_densenet(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "RefineNet-Res50" or args.model == "RefineNet-Res101" or args.model == "RefineNet-Res152":
    # RefineNet requires pre-trained ResNet weights
    network, init_fn = build_refinenet(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "FRRN-A" or args.model == "FRRN-B":
    network = build_frrn(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "Encoder-Decoder" or args.model == "Encoder-Decoder-Skip":
    network = build_encoder_decoder(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "MobileUNet" or args.model == "MobileUNet-Skip":
    network = build_mobile_unet(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "PSPNet-Res50" or args.model == "PSPNet-Res101" or args.model == "PSPNet-Res152":
    # Image size is required for PSPNet
    # PSPNet requires pre-trained ResNet weights
    network, init_fn = build_pspnet(net_input, label_size=[args.crop_height, args.crop_width], preset_model = args.model, num_classes=num_classes)
elif args.model == "GCN-Res50" or args.model == "GCN-Res101" or args.model == "GCN-Res152":
    # GCN requires pre-trained ResNet weights
    network, init_fn = build_gcn(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "DeepLabV3-Res50" or args.model == "DeepLabV3-Res101" or args.model == "DeepLabV3-Res152":
    # DeepLabV requires pre-trained ResNet weights
    network, init_fn = build_deeplabv3(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "DeepLabV3_plus-Res50" or args.model == "DeepLabV3_plus-Res101" or args.model == "DeepLabV3_plus-Res152":
    # DeepLabV3+ requires pre-trained ResNet weights
    network, init_fn = build_deeplabv3_plus(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "AdapNet":
    network = build_adaptnet(net_input, num_classes=num_classes)
elif args.model == "custom":
    network = build_custom(net_input, num_classes)
else:
    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")


losses = None
if args.class_balancing:
    print("Computing class weights for", args.dataset, "...")
    class_weights = utils.compute_class_weights(labels_dir=args.dataset + "/train_labels", label_values=label_values)
    weights = tf.reduce_sum(class_weights * net_output, axis=-1)
    unweighted_loss = None
    unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
    losses = unweighted_loss * weights
    print("Printing class weights for", args.dataset, "...")
    print(class_weights)

else:
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)

loss = tf.reduce_mean(losses)
opt = tf.train.AdamOptimizer(args.learn_rate).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
model_checkpoint_name = "checkpoints_1/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training or not args.mode == "train":
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

avg_scores_per_epoch = []

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = helpers.prepare_data(args.dataset)

##-------------------------------------------------------------------------------------------------##
if args.mode == "train":

    print("\n***** Begin training *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Crop Height -->", args.crop_height)
    print("Crop Width -->", args.crop_width)
    print("Num Epochs -->", args.num_epochs)
    print("Batch Size -->", args.batch_size)
    print("Num Classes -->", num_classes)
    print("Class Balancing -->", args.class_balancing)
    print("Learning Rate -->", args.learn_rate)
    print("")

    print("Data Augmentation:")
    print("\tVertical Flip -->", args.v_flip)
    print("\tHorizontal Flip -->", args.h_flip)
    print("\tBrightness Alteration -->", args.brightness)
    print("\tRotation -->", args.rotation)
    print("\tDroplets # -->", args.droplets_num)
    print("\tDroplets Size -->", args.droplets_size)
    print("")

    avg_loss_per_epoch = []

    # Which validation images do we want
    val_indices = []
    num_vals = min(args.num_val_images, len(val_input_names))

    # Set random seed to make sure models are validated on the same validation images.
    # So you can compare the results of different models more intuitively.
    random.seed(16)
    val_indices=random.sample(range(0,len(val_input_names)),num_vals)

    # Do the training here
    for epoch in range(0, args.num_epochs):

        current_losses = []

        cnt=0

        # Equivalent to shuffling
        id_list = np.random.permutation(len(train_input_names))

        num_iters = int(np.floor(len(id_list) / args.batch_size))
        st = time.time()
        epoch_st=time.time()
        for i in range(num_iters):
            # st=time.time()

            input_image_batch = []
            output_image_batch = []

            # Collect a batch of images
            for j in range(args.batch_size):
                index = i*args.batch_size + j
                id = id_list[index]
                input_image = load_image(train_input_names[id])
                output_image = load_image(train_output_names[id])

                with tf.device('/cpu:0'):
                    input_image, output_image = helpers.data_augmentation(input_image, output_image, args.crop_height, args.crop_width, args.h_flip, args.v_flip, args.droplets_num, args.droplets_size, args.brightness, args.rotation)

                    # Prep the data. Make sure the labels are in one-hot format
                    input_image = np.float32(input_image) / 255.0
                    output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

                    input_image_batch.append(np.expand_dims(input_image, axis=0))
                    output_image_batch.append(np.expand_dims(output_image, axis=0))

            # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****
            # input_image = tf.image.crop_to_bounding_box(input_image, offset_height=0, offset_width=0,
            #                                               target_height=args.crop_height, target_width=args.crop_width).eval(session=sess)
            # output_image = tf.image.crop_to_bounding_box(output_image, offset_height=0, offset_width=0,
            #                                               target_height=args.crop_height, target_width=args.crop_width).eval(session=sess)
            # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****

            # memory()

            if args.batch_size == 1:
                input_image_batch = input_image_batch[0]
                output_image_batch = output_image_batch[0]
            else:
                input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

            # Do the training
            _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
            current_losses.append(current)

            cnt = cnt + args.batch_size
            if cnt % 20 == 0:
                string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
                utils.LOG(string_print)
                st = time.time()

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)

        # Create directories if needed
        if not os.path.isdir("%s/%04d"%("checkpoints_1",epoch)):
            os.makedirs("%s/%04d"%("checkpoints_1",epoch))

        # Save latest checkpoint to same file name
        print("Saving latest checkpoint")
        saver.save(sess,model_checkpoint_name)

        if val_indices != 0 and epoch % args.checkpoint_step == 0:
            print("Saving checkpoint for this epoch")
            saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints_1",epoch))


####################################################################################################

    print("\n***** Begin testing *****")

    ## Usage: python3 main.py --mode test --dataset dataSet --crop_height 515 --crop_width 915 --model DeepLabV3-Res152

    # Create directories if needed
    if not os.path.isdir("%s"%("Val")):
            os.makedirs("%s"%("Val"))

    target=open("%s/val_scores.csv"%("Val"),'a')
    target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
    scores_list = []
    class_scores_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []
    run_times_list = []

    # Run testing on ALL test images
    for ind in range(len(test_input_names)):
        sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
        sys.stdout.flush()

        input_image = np.expand_dims(np.float32(load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
        gt = load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
        gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

        st = time.time()
        output_image = sess.run(network,feed_dict={net_input:input_image})

        run_times_list.append(time.time()-st)

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)
        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

        accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

        file_name = utils.filepath_to_name(test_input_names[ind])
        target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
        for item in class_accuracies:
            target.write(", %f"%(item))
        target.write("\n")

        scores_list.append(accuracy)
        class_scores_list.append(class_accuracies)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        iou_list.append(iou)

        gt = helpers.colour_code_segmentation(gt, label_values)

        cv2.imwrite("%s/%s_pred.png"%("Val", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
        cv2.imwrite("%s/%s_gt.png"%("Val", file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

    target.close()

    avg_score = np.mean(scores_list)
    class_avg_scores = np.mean(class_scores_list, axis=0)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_iou = np.mean(iou_list)
    avg_time = np.mean(run_times_list)
    print("Average test accuracy = ", avg_score)
    print("Average per class test accuracies = \n")
    for index, item in enumerate(class_avg_scores):
        print("%s = %f" % (class_names_list[index], item))
    print("Average precision = ", avg_precision)
    print("Average recall = ", avg_recall)
    print("Average F1 score = ", avg_f1)
    print("Average mean IoU score = ", avg_iou)
    print("Average run time = ", avg_time)





####################################################################################################

        if epoch % args.validation_step == 0:
            print("Performing validation")
            target=open("%s/%04d/val_scores.csv"%("checkpoints_1",epoch),'a')
            target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))

            scores_list = []
            class_scores_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            iou_list = []

            # Do the validation on a small set of validation images
            for ind in val_indices:

                input_image = np.expand_dims(np.float32(load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
                gt = load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
                gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

                # st = time.time()

                output_image = sess.run(network,feed_dict={net_input:input_image})

                output_image = np.array(output_image[0,:,:,:])
                output_image = helpers.reverse_one_hot(output_image)
                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

                accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

                file_name = utils.filepath_to_name(val_input_names[ind])
                target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
                for item in class_accuracies:
                    target.write(", %f"%(item))
                target.write("\n")

                scores_list.append(accuracy)
                class_scores_list.append(class_accuracies)
                precision_list.append(prec)
                recall_list.append(rec)
                f1_list.append(f1)
                iou_list.append(iou)

                gt = helpers.colour_code_segmentation(gt, label_values)

                file_name = os.path.basename(val_input_names[ind])
                file_name = os.path.splitext(file_name)[0]
                cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints_1",epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints_1",epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


            target.close()

            avg_score = np.mean(scores_list)
            class_avg_scores = np.mean(class_scores_list, axis=0)
            avg_scores_per_epoch.append(avg_score)
            avg_precision = np.mean(precision_list)
            avg_recall = np.mean(recall_list)
            avg_f1 = np.mean(f1_list)
            avg_iou = np.mean(iou_list)

            print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
            print("Average per class validation accuracies for epoch # %04d:"% (epoch))
            for index, item in enumerate(class_avg_scores):
                print("%s = %f" % (class_names_list[index], item))
            print("Validation precision = ", avg_precision)
            print("Validation recall = ", avg_recall)
            print("Validation F1 score = ", avg_f1)
            print("Validation IoU score = ", avg_iou)

        epoch_time=time.time()-epoch_st
        remain_time=epoch_time*(args.num_epochs-1-epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        if s!=0:
            train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
        else:
            train_time="Remaining training time : Training completed.\n"
        utils.LOG(train_time)
        scores_list = []

        fig = plt.figure(figsize=(11,8))
        ax1 = fig.add_subplot(111)
        # ax1.plot(range(args.num_epochs), avg_scores_per_epoch)
        ax1.plot(avg_scores_per_epoch)
        ax1.set_title("Average validation accuracy vs epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Avg. val. accuracy")
        plt.savefig('accuracy_vs_epochs_2.png')
        plt.clf()

        ax1 = fig.add_subplot(111)
        # ax1.plot(range(args.num_epochs), avg_loss_per_epoch)
        ax1.plot(avg_loss_per_epoch)
        ax1.set_title("Average loss vs epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Current loss")
        plt.savefig('loss_vs_epochs_2.png')

##-------------------------------------------------------------------------------------------------##
elif args.mode == "test":
    print("\n***** Begin testing *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Crop Height -->", args.crop_height)
    print("Crop Width -->", args.crop_width)
    print("Num Classes -->", num_classes)
    print("")

    ## Usage: python3 main.py --mode test --dataset dataSet --crop_height 515 --crop_width 915 --model DeepLabV3-Res152

    # Create directories if needed
    if not os.path.isdir("%s"%("Val")):
            os.makedirs("%s"%("Val"))

    target=open("%s/val_scores.csv"%("Val"),'w')
    target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
    scores_list = []
    class_scores_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []
    run_times_list = []

    # Run testing on ALL test images
    for ind in range(len(test_input_names)):
        sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
        sys.stdout.flush()

        input_image = np.expand_dims(np.float32(load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
        gt = load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
        gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

        st = time.time()
        output_image = sess.run(network,feed_dict={net_input:input_image})

        run_times_list.append(time.time()-st)

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)
        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

        accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

        file_name = utils.filepath_to_name(test_input_names[ind])
        target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
        for item in class_accuracies:
            target.write(", %f"%(item))
        target.write("\n")

        scores_list.append(accuracy)
        class_scores_list.append(class_accuracies)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        iou_list.append(iou)

        gt = helpers.colour_code_segmentation(gt, label_values)

        cv2.imwrite("%s/%s_pred.png"%("Val", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
        cv2.imwrite("%s/%s_gt.png"%("Val", file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

    target.close()

    avg_score = np.mean(scores_list)
    class_avg_scores = np.mean(class_scores_list, axis=0)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_iou = np.mean(iou_list)
    avg_time = np.mean(run_times_list)
    print("Average test accuracy = ", avg_score)
    print("Average per class test accuracies = \n")
    for index, item in enumerate(class_avg_scores):
        print("%s = %f" % (class_names_list[index], item))
    print("Average precision = ", avg_precision)
    print("Average recall = ", avg_recall)
    print("Average F1 score = ", avg_f1)
    print("Average mean IoU score = ", avg_iou)
    print("Average run time = ", avg_time)

##-------------------------------------------------------------------------------------------------##
elif args.mode == "predict": # This method is not recommended for benchmarking speed
                             # The first image that TensorFlow reads will take ~5x longer than the rest of the images

    ## Usage: python3 main.py --mode predict --image test1.png --dataset datasetName --model DeepLabV3-Res152

    if args.image is None:
        ValueError("You must pass an image path when using prediction mode.")

    print("\n***** Begin prediction *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Crop Height -->", args.crop_height)
    print("Crop Width -->", args.crop_width)
    print("Num Classes -->", num_classes)
    print("Image -->", args.image)
    print("")

    # Create directories if needed
    if not os.path.isdir("%s"%("Test")):
            os.makedirs("%s"%("Test"))

    sys.stdout.write("Testing image " + args.image)
    sys.stdout.flush()

    # to get the right aspect ratio of the output
    loaded_image = load_image(args.image)
    height, width, channels = loaded_image.shape
    resize_height = int(height / (width / args.crop_width))

    resized_image =cv2.resize(loaded_image, (args.crop_width, resize_height))
    input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})
    run_time = time.time()-st

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)

    # this was generalized to accept any dataset
    class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    file_name = utils.filepath_to_name(args.image)
    cv2.imwrite("%s/%s_pred.png"%("Test", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

    print("")
    print("Finished!")
    print("Wrote image " + "%s/%s_pred.png"%("Test", file_name))

##-------------------------------------------------------------------------------------------------##

    ## Usage: python3 main.py --mode predict_folder --dataset dataset --crop_height 390 --crop_width 390 --model DeepLabV3-Res152


elif args.mode == "predict_folder":

    print("\n***** Begin prediction on folder *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Crop Height -->", args.crop_height)
    print("Crop Width -->", args.crop_width)
    print("Num Classes -->", num_classes)
    print("Image Folder -->", args.image_folder)
    print("Removal -->", args.removal)
    print("Aspect Ratio Lock -->", args.ratio_lock)
    print("Pred Center Crop -->", args.pred_center_crop)
    print("Pred Downsample -->", args.pred_downsample)
    print("Post Processing -->", args.post_processing)
    print("Rail sensitivity -->", args.rail_sens)
    print("Generate Images -->", args.generate_images)
    print("")

    if (args.post_processing > 0):
        if not os.path.isdir("%s_%s/%s"%("Test",args.image_folder,"Unprocessed")):
                os.makedirs("%s_%s/%s"%("Test",args.image_folder,"Unprocessed"))
        if not os.path.isdir("%s_%s/%s"%("Test",args.image_folder,"Processed")):
                os.makedirs("%s_%s/%s"%("Test",args.image_folder,"Processed"))
        if not os.path.isdir("%s_%s/%s"%("Test",args.image_folder,"Original")):
                os.makedirs("%s_%s/%s"%("Test",args.image_folder,"Original"))
        if not os.path.isdir("%s_%s/%s"%("Test",args.image_folder,"Combined_proc")):
                os.makedirs("%s_%s/%s"%("Test",args.image_folder,"Combined_proc"))
        if not os.path.isdir("%s_%s/%s"%("Test",args.image_folder,"Combined_unproc")):
                os.makedirs("%s_%s/%s"%("Test",args.image_folder,"Combined_unproc"))
        if not os.path.isdir("%s_%s/%s"%("Test",args.image_folder,"Probabilities")):
                os.makedirs("%s_%s/%s"%("Test",args.image_folder,"Probabilities"))
        if not os.path.isdir("%s_%s/%s"%("Test",args.image_folder,"Probabilities_First")):
                os.makedirs("%s_%s/%s"%("Test",args.image_folder,"Probabilities_First"))

    imageDir = args.image_folder #default is 'Predict'
    image_path_list = []
    valid_image_extensions = [".jpg", ".png", ".tif"] #specify your image extension here
    i = 0

    #this will loop through all files in imageDir
    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir, file))

    start = time.time()
    dur1 = 0 #timing first part of loop
    dur2 = 0 #timing second part of loop
    dur3 = 0
    dur4 = 0

    for imagePath in sorted(image_path_list):

        start1 = time.time()

        i = i + 1;

        #sys.stdout.write("Testing image " + imagePath)
        #sys.stdout.flush()

        # to get the right aspect ratio of the output
        loaded_image = load_image(imagePath)
        if loaded_image is None:
            continue

        height, width, channels = loaded_image.shape

        if(args.pred_downsample):
            if(args.ratio_lock):
                resize_height = int(height / (width / args.crop_width))
                resized_image = cv2.resize(loaded_image, (args.crop_width, resize_height))
            else:
                resized_image = cv2.resize(loaded_image, (args.crop_width, args.crop_height))
        else:
            if(args.pred_center_crop):
                x_pad = int((loaded_image.shape[1]-1-args.crop_width)/2)
                y_pad = int((loaded_image.shape[0]-1-args.crop_height)/2)
                resized_image = loaded_image[y_pad:args.crop_height+y_pad, x_pad:args.crop_width+x_pad]
                resized_image = resized_image[0:args.crop_height, 0:args.crop_width]
                #height, width, channels = resized_image.shape
                #print("Original H/W/C: " + str(height) + " " + str(width) + " " + str(channels))
            else:
                resized_image = loaded_image[0:args.crop_height, 0:args.crop_width]

        input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0
        if (args.post_processing > 0):
            resized_image_copy = resized_image
            resized_image_vis = cv2.cvtColor(np.uint8(resized_image_copy), cv2.COLOR_RGB2BGR)

        st = time.time()
        end1 = time.time()
        dur1 = dur1 + (end1-start1)

        start2 = time.time()
        output_image = sess.run(network,feed_dict={net_input:input_image}) # GENERATING INPUT
        end2 = time.time()
        dur2 = dur2 + (end2-start2)

        start3 = time.time()
        run_time = time.time()-st

        output_image = np.array(output_image[0,:,:,:])
        file_name = utils.filepath_to_name(imagePath)

        if (args.post_processing > 0):
            #find the probabilities for each pixel
            sf = utils.softmax(output_image, theta = args.theta, axis = 2)
            sf_first = sf[:,:,0]
            sf_m = np.amax(sf, 2)

            output_image = helpers.reverse_one_hot(output_image)
            # this was generalized to accept any dataset
            class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
            unprocessed_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)

            start4 = time.time()
            if (args.post_processing >= 2):
                processed_image = PostProcessing.ProcessImageMARS(cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR),args.removal, args.post_processing)
            elif (args.post_processing == 1):
                processed_image = PostProcessing.ProcessImageRail(cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR), args.post_processing, args.rail_sens)
            end4 = time.time()
            dur4 = dur4 + (end4-start4)

            unprocessed_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR) #needs to be re-generated
            unprocessed_image_sf = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_BGR2GRAY)
            unprocessed_image_sf_o = unprocessed_image_sf * sf_m

            if resized_image_vis is None:
                print("WARNING: image not found (1)")
                continue
            if processed_image is None:
                print("WARNING: image not found (2)")
                continue
            if unprocessed_image is None:
                print("WARNING: image not found (3)")
                continue

            if (args.generate_images):
                cv2.imwrite("%s_%s/%s/%s.png"%("Test",args.image_folder,"Original", file_name),resized_image_vis)
                cv2.imwrite("%s_%s/%s/%s_pred.png"%("Test",args.image_folder,"Unprocessed", file_name),unprocessed_image)
                cv2.imwrite("%s_%s/%s/%s_pred.png"%("Test",args.image_folder,"Processed", file_name),processed_image)
                cv2.imwrite("%s_%s/%s/%s_pred.png"%("Test",args.image_folder,"Probabilities", file_name),sf_m*255)
                cv2.imwrite("%s_%s/%s/%s_pred.png"%("Test",args.image_folder,"Probabilities_First", file_name),sf_first*255)

                combined_proc = cv2.addWeighted(resized_image_vis,0.6,processed_image,0.4,0)
                combined_unproc = cv2.addWeighted(resized_image_vis,0.6,unprocessed_image,0.4,0)

                cv2.imwrite("%s_%s/%s/%s_pred.png"%("Test",args.image_folder,"Combined_proc", file_name),combined_proc)
                cv2.imwrite("%s_%s/%s/%s_pred.png"%("Test",args.image_folder,"Combined_unproc", file_name),combined_unproc)

        print("Generated Predictions for: " + "%s_%s/%s/%s_pred.png"%("Test",args.image_folder,"Processed", file_name))
        end3 = time.time()
        dur3 = dur3 + (end3-start3)

    # Doing timings
    duration = time.time() - start
    avgSpeed = duration/i
    avgSpeed1 = dur1/i
    avgSpeed2 = dur2/i
    avgSpeed3 = dur3/i
    avgSpeed4 = dur4/i
    FPS = 1/avgSpeed
    FPS2 = round(1/avgSpeed2,2)
    FPS4 = round(1/avgSpeed4,2)
    print("")
    print("Finished!")
    print("")
    print("Model generated predictions for " + str(i) + " images in " + str(round(duration,3)) + " seconds.")
    print("Average inference speed: " + str(round(avgSpeed,3)) + " seconds per image. (" + str(round(FPS,2)) +  " FPS)")
    print("")
    print("Reading image average time: " + str(round(avgSpeed1,3)))
    print("Predict image average time: " + str(round(avgSpeed2,3)) + " (" + str(FPS2) +  " FPS)")
    print("Writing image average time: " + str(round(avgSpeed3,3)))
    print("Processing image average time: " + str(round(avgSpeed4,3))+ " (" + str(FPS4) +  " FPS)")


##-------------------------------------------------------------------------------------------------##
else:
    ValueError("Invalid mode selected.")
