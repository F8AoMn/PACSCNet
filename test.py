from datetime import datetime
import torch.utils.data as data
import torch.optim.lr_scheduler
import torch.nn.init
from dataprocessing import *
from tqdm import tqdm

from PACSCNet.PACSCNet import FFNet

if DATASET == 'Potsdam':
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    # print(all_files)
    # all_ids = ["".join(f.split('')[5:7]) for f in all_files]
    all_ids = ["".join(f.split('/')[-1].split('_')[2] + '_' + f.split('/')[-1].split('_')[3]) for f in all_files]
    # print(all_ids)
elif DATASET == 'Vaihingen':
    # all_ids =
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]
# train_ids = random.sample(all_ids, 2 * len(all_ids) // 3 + 1)
# test_ids = list(set(all_ids) - set(train_ids))

# Exemple of a train/test split on Vaihingen :
if DATASET == 'Potsdam':
    train_ids = ['2_10', '3_10', '3_11', '3_12', '4_11', '4_12', '5_10', '5_12', '6_8', '6_9', '6_10', '6_11', '6_12', '7_7', '7_9', '7_11', '7_12']
    test_ids = ['2_11', '2_12', '4_10', '5_11']   # '2_11', '2_12', '4_10', '5_11', '6_7', '7_8', '7_10'
elif DATASET == 'Vaihingen':
    train_ids = ['1', '3', '5', '7', '13', '17', '21', '23', '26', '32', '37']
    test_ids = ['11', '15', '28', '30', '34']

print("Tiles for training: ", train_ids)
print("Tiles for testing: ", test_ids)


def test(net , test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_ndsm = (1 / 255 * np.asarray(io.imread(NDSM_FOLODER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        net.eval()

        for img, ndsm, gt, gt_e in tqdm(zip(test_images, test_ndsm, test_labels, eroded_labels), total=len(test_ids), leave=None):
            pred = np.zeros(img.shape[:2] + (N_CLASSES, ))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=None)):
                # Display in progress results
                # Build the tensor
                image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                ndsm_patches = [np.copy(ndsm[np.newaxis, x:x+w, y:y+h]) for x, y, w, h in coords]
                ndsm_patches = np.asarray(ndsm_patches)
                ndsm_patches = Variable(torch.from_numpy(ndsm_patches).cuda(), volatile=True)
                # ndsm_patches = torch.repeat_interleave(ndsm_patches, 3, dim=1)
                # Do the inference
                outs = net(image_patches, ndsm_patches)
                # outs = net(image_patches)
                outs = outs.data.cpu().numpy()
                # outimg = convert_to_color_(outs)
                # io.imsave('./result/inference_tile_{}.png'.format(i), outimg)

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x+w, y:y+h] += out
                del(outs)

            pred = np.argmax(pred, axis=-1)

            all_preds.append(pred)
            all_gts.append(gt_e)

            metrics(pred.ravel(), gt_e.ravel())
            accuary = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        # return accuary, all_preds, all_gts
        return accuary, all_preds, all_gts
    else:
        return accuary


if __name__ == "__main__":
    net = FFNet().cuda()
    net.load_state_dict(torch.load('./weight/PACSCNet-Vaihingen-loss.pth'))
    # net.load_state_dict(torch.load('./weight/PACSCNet-Potsdam-loss.pth'))

    _, all_preds, all_gts = test(net, test_ids, all=True, stride=48)
    # print(accuary)
    for p, id_ in zip(all_preds, test_ids):
        img = convert_to_color_(p)
        io.imsave('./results/Vaihingen/PACSCNet/top_mosaic_09cm_area{}.png'.format(id_), img)
        # io.imsave('./results/Potsdam/PACSCNet/top_mosaic_09cm_area{}.png'.format(id_), img)
