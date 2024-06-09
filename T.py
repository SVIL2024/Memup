import torch.utils.data as data
import torchvision.transforms as transforms
from data import *
from torch.autograd import Variable
from utils import *
from updatemem import *
import random
import argparse
import time
import random

number_range = [90, 180, 270]
random_number = random.choice(number_range)


parser = argparse.ArgumentParser(description="EMF")
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--learning_rate_ped2', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--learning_rate_avenue', default=0.0000001, type=float, help='initial learning_rate')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=0, help='number of workers for the test loader')
parser.add_argument('--loss_m_weight', help='loss_m_weight', type=float, default=0.0002)

parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2', 'avenue', 'shanghai'],
                    help='type of dataset: ped2, avenue, shanghai')
# parser.add_argument('--path', type=str, default='./exp_up13', help='directory of data')
parser.add_argument('--path', type=str, default='exp_up', help='directory of data')
parser.add_argument('--path_num', type=int, default=0, help='number of path')
parser.add_argument('--mem_dim', type=int, default=2000, help='size of mem')
parser.add_argument('--ano_mem_dim', type=int, default=2000, help='size of mem_ano')
parser.add_argument('--sigma_noise', default='0.9', type=float, help='sigma of noise added to the iamges')


parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='basename of folder to save weights')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                    help='adam or sgd with momentum and cosine annealing lr')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--mem_usage', default=[False, False, False, True], type=str)
parser.add_argument('--skip_ops', default=["none", "concat", "none"], type=str)

parser.add_argument('--pseudo_anomaly_jump', type=float, default=0.000001,
                    help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--jump', nargs='+', type=int, default=[2], help='Jump for pseudo anomaly (hyperparameter s)')  # --jump 2 3


parser.add_argument('--model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')

#test
parser.add_argument('--th', type=float, default=0.02, help='threshold for test updating')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--print_score', action='store_true', help='print score')
parser.add_argument('--vid_dir', type=str, default=None, help='save video frames file')
parser.add_argument('--print_time', action='store_true', help='print forward time')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

#Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')



args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    device = torch.cuda.current_device()
channel_in = 1
if args.dataset_type == 'ped2':
    channel_in = 1
    learning_rate = args.learning_rate_ped2
    train_folder = os.path.join('UCSDped2', 'Train')
    test_folder = os.path.join('UCSDped2', 'Test')

else:
    channel_in = 3
    learning_rate = args.learning_rate_avenue
    train_folder = os.path.join('Avenue', 'Train')
    test_folder = os.path.join('Avenue', 'Test')
    args.epochs = args.epochs + 70

print(f'epochs:{args.epochs}')

exp_dir = args.exp_dir
exp_dir += '_lr' + str(learning_rate) + '_'
exp_dir += 'weight'
exp_dir += '_recon'



# print('exp_dir: ', exp_dir)


# torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance


# train_folder = os.path.join(args.dataset_path, args.dataset_type, 'training')
# train_folder = os.path.join('UCSDped2', 'Train')
# train_folder = os.path.join('Avenue', 'Train')

print(f'train_folder:{train_folder}')

trans_compose = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation(degrees=(90, 90),
                                                                                       expand=False, center=(128, 128)),
                                                                             transforms.ToTensor()])

# trans_compose = transforms.Compose([transforms.ToTensor()])

# trans_compose = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.8)])

# Loading dataset
img_extension = '.tif' if args.dataset_type == 'ped2' else '.jpg'
train_dataset = Reconstruction3DDataLoader(train_folder, trans_compose,
                                           resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                           img_extension=img_extension)
train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder,
                                                    # transforms.Compose([transforms.ToTensor()]),
                                                    trans_compose,
                                                resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                                jump=args.jump, img_extension=img_extension)

#train_dataset = traindataset(train_folder)


train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, drop_last=True)
train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers, drop_last=True)


print(f'len(train_batch):{len(train_batch)}')


# Report the training process


log_dir = os.path.join('./', args.path + str(args.path_num), args.dataset_type, exp_dir)
# while os.path.exists(log_dir):
#     args.path_num = args.path_num + 1
# log_dir = os.path.join('./', args.path + str(args.path_num), args.dataset_type, exp_dir)
# if not os.path.exists(args.path + str(args.path_num)):

print(f'log_dir:{log_dir}')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

orig_stdout = sys.stdout

f = open(os.path.join(log_dir, 'log.txt'), 'a')
sys.stdout = f

torch.set_printoptions(profile="full")

loss_func_mse = nn.MSELoss(reduction='none')

# entropy_loss_func = EntropyLossEncap().cuda()
# loss_m_weight = args.loss_m_weight

if args.start_epoch < args.epochs:
    model = ML_MemAE_SC(num_in_ch=channel_in, features_root=32,
                        mem_dim=args.mem_dim, ano_mem_dim=args.ano_mem_dim, shrink_thres=0.0005,
                        mem_usage=args.mem_usage, skip_ops=args.skip_ops, hard_shrink_opt=True)

    model.cuda()
    separateness_loss_ano = torch.zeros(1, 1).cuda()
    compactness_loss_ano = torch.zeros(1, 1).cuda()
    # separateness_loss = torch.zeros(1, 1).cuda()
    # compactness_loss = torch.zeros(1, 1).cuda()
    sigma = args.sigma_noise ** 2

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.8)
    tic = time.time()

    # model.eval()
    for epoch in range(args.start_epoch, args.epochs):
        pseudolossepoch = 0
        lossepoch = 0
        pseudolosscounter = 0
        losscounter = 0

        # for j, (imgs) in enumerate(zip(train_batch)):
        # for j, (imgs) in enumerate(train_batch):
        #     net_in = copy.deepcopy(imgs)
        #     net_in = net_in.cuda()
        for j, (imgs, imgsjump) in enumerate(zip(train_batch, train_batch_jump)):
            net_in = copy.deepcopy(imgs)
            net_in = net_in.cuda()
            imgsjump_gaus = gaussian(imgsjump, 1, 0, sigma)
            # noise = np.random.normal(0, 1, imgsjump.shape)
            jump_pseudo_stat = []
            cls_labels = []

            for b in range(args.batch_size):
                total_pseudo_prob = 0
                rand_number = np.random.rand()
                pseudo_bool = False

                # skip frame pseudo anomaly
                pseudo_anomaly_jump = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_jump
                total_pseudo_prob += args.pseudo_anomaly_jump
                if pseudo_anomaly_jump:
                    net_in[b] = imgsjump_gaus[b][0]
                    # net_in[b] = imgsjump[b][0]
                    jump_pseudo_stat.append(True)
                    pseudo_bool = True
                else:
                    jump_pseudo_stat.append(False)

                if pseudo_bool:
                    cls_labels.append(0)
                else:
                    cls_labels.append(1)
            for b in range(args.batch_size):
                if jump_pseudo_stat[b]:
                    out, feas, separateness_loss_ano, compactness_loss_ano = model.forward(net_in, mem=False, mem_ano=True, train=True)
                else:
                    out, feas, separateness_loss, compactness_loss = model.forward(net_in, mem=True, mem_ano=False, train=True)
            # out = model(net_in)
            loss_mem = torch.abs(out['mem'].cuda() - out['mem_ano'].cuda()) * -1.0
            loss_mse = loss_func_mse(out["recon"], net_in)
            # cross_entropy = entropy_loss_func(out["att"])
            # loss_sparsity = torch.mean(torch.sum(-out["att_weight3"] * torch.log(out["att_weight3"] + 1e-12), dim=1))
            # loss_sparsity_ano = -1.0 * torch.mean(
            #     torch.sum(-out["att_weight3_ano"] * torch.log(out["att_weight3_ano"] + 1e-12), dim=1))

            modified_loss_mse = []
            for b in range(args.batch_size):
                if jump_pseudo_stat[b]:
                    modified_loss_mse.append(torch.mean(-loss_mse[b]))
                    pseudolossepoch += modified_loss_mse[-1].cpu().detach().item()
                    pseudolosscounter += 1

                else:  # no pseudo anomaly
                    modified_loss_mse.append(torch.mean(loss_mse[b]))
                    lossepoch += modified_loss_mse[-1].cpu().detach().item()
                    losscounter += 1
            assert len(modified_loss_mse) == loss_mse.size(0)
            stacked_loss_mse = torch.stack(modified_loss_mse)
            loss_all = (stacked_loss_mse
                        + args.loss_separate * separateness_loss + args.loss_compact * compactness_loss
                        # + args.loss_m_weight * loss_sparsity_ano + args.loss_m_weight * loss_sparsity
                        + args.loss_separate * separateness_loss_ano + args.loss_compact * compactness_loss_ano
                        ).sum() + loss_mem.sum() * 0.00002
            # loss_all = loss_mse

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        scheduler.step()


        # Save the model and the memory items
        model_dict = {
            'model': model,
            # 'discriminator':discriminator,
            # 'optimizer_g': optimizer_g.state_dict(),
            # 'optimizer_d': optimizer_d.state_dict(),
        }

        torch.save(model_dict, os.path.join(log_dir, 'model_{:02d}.pth'.format(epoch)))
    print('Training is finished')
    toc = time.time()
    # print('time:' + str(1000 * (toc - tic)) + "ms")
    # print('mean time:' + str(1000 * (toc - tic) / args.epochs) + "ms")
    print('time:' + str(1000 * (toc - tic) / 60000 / 60) + "h")
    print('mean time:' + str(1000 * (toc - tic) / 60000 / args.epochs) + "min")
    sys.stdout = orig_stdout
    f.close()



#Test
loss_func_mse = nn.MSELoss(reduction='none')
labels = np.load('./frame_labels_'+args.dataset_type+'.npy', allow_pickle=True)
# img_extension = '.tif' if args.dataset_type == 'ped2' else '.jpg'
test_dataset = Reconstruction3DDataLoader(test_folder, trans_compose,
                                          resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                          img_extension=img_extension, train=False)
test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

model = ML_MemAE_SC(num_in_ch=channel_in, features_root=32,
                        mem_dim=args.mem_dim, ano_mem_dim=args.ano_mem_dim, shrink_thres=0.0005,
                        mem_usage=args.mem_usage, skip_ops=args.skip_ops, hard_shrink_opt=True)

print(f'len(test_batch):{len(test_batch)}')
videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*/')))
# videos_list = sorted(glob.glob('test_folder/*'))
# print(videos_list)
for video in videos_list:
    video_name = video.split('\\')[-2]
    #video_name = video.split('/')[-2]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + img_extension))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print(f'epochs:{args.epochs}')
print('Evaluation of', args.dataset_type)
# Setting for video anomaly detection
tic = time.time()

for epoch_num in range(args.epochs):
    if epoch_num < 10:
        model_dict = torch.load(log_dir + "\\" + f'model_0{epoch_num}.pth')
    else:
        model_dict = torch.load(log_dir + "\\" + f'model_{epoch_num}.pth')

    # model_weight = model_dict['model']
    model.load_state_dict(model_dict['model'].state_dict())

    model.cuda()
    for video in sorted(videos_list):
        video_name = video.split('\\')[-2]
        labels_list = np.append(labels_list,
                                labels[0][8 + label_length:videos[video_name]['length'] + label_length - 7])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0

    label_length += videos[videos_list[video_num].split('\\')[-2]]['length']

    model.eval()

    for k, (imgs) in enumerate(test_batch):

        if k == label_length - 15 * (video_num + 1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('\\')[-2]]['length']

        imgs = Variable(imgs).cuda()
        with torch.no_grad():
            out, feas, compactness_loss = model.forward(imgs, mem=True, train=False)
            outputs = out['recon']
            loss_mse = loss_func_mse(outputs[0, :, 8], imgs[0, :, 8])
            mse_feas = compactness_loss.item()

            # Calculating the threshold for updating at the test time
            # point_sc = point_score(outputs, imgs)

        loss_pixel = torch.mean(loss_mse)
        mse_imgs = loss_pixel.item()
        # if point_sc < args.th:
        #     query = F.normalize(feas, dim=1)
        #     query = query.permute(0, 2, 3, 4, 1)  # b X deep X h X w X d
        #     model.mem3 = model.update(query, out['mem'], False).cuda()

        psnr_list[videos_list[video_num].split('\\')[-2]].append(psnr(mse_imgs))
        # feature_distance_list[videos_list[video_num].split('\\')[-2]].append(mse_feas)

    # Measuring the abnormality score (S) and the AUC
    anomaly_score_total_list = []
    vid_idx = []
    for vi, video in enumerate(sorted(videos_list)):
        video_name = video.split('\\')[-2]
        score = anomaly_score_list(psnr_list[video_name])
        # anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]),
        #                                       feature_distance_list[video_name], args.alpha)
        anomaly_score_total_list += score
        vid_idx += [vi for _ in range(len(score))]

    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))

    # print('vididx,frame,anomaly_score,anomaly_label')
    # for a in range(len(anomaly_score_total_list)):
    #     print(str(vid_idx[a]), ',', str(a), ',', 1-anomaly_score_total_list[a], ',', labels_list[a])

    print('The result of ', args.dataset_type)
    if epoch_num < 10:
        print(f'model_0{epoch_num}_AUC: ', accuracy * 100, '%')
    else:
        print(f'model_{epoch_num}_AUC: ', accuracy * 100, '%')
    print('----------------------------------------')
toc = time.time()
# print('time:' + str(1000 * (toc - tic)) + "ms")
# print('mean time:' + str(1000 * (toc - tic) / args.epochs) + "ms")
print('time:' + str(1000 * (toc - tic) / 3600000) + "h")
print('mean time:' + str(1000 * (toc - tic) / 60000 / args.epochs) + "min")
print('Testing is finished')



