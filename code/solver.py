import torch
import network
import numpy as np
import cv2
from dataset import get_loader
from os.path import exists, join
from os import makedirs

class Solver(object):
    def __init__(self):
        self.net = network.Net().cuda()

    def threshold(self,i):
        if i>0.7:
            return 1
        return 0

    def image_multiply(self, original, mask):

        for i in range (223):
            for j in range(223):
                original[i][j][0] *= self.threshold(mask[i][j][0])
                original[i][j][1] *=self.threshold(mask[i][j][0])
                original[i][j][2] *=self.threshold(mask[i][j][0])
        return original

    def image_crop(self, img):
        left = 0
        right = 0
        top = 0
        bottom = 0
        flag_first = False
        for i in range (223):
            for j in range(223):
                if ((img[i][j][0] != 0) or (img[i][j][1] != 0) or (img[i][j][2] != 0)) and (flag_first == False) :
                    top = i
                    bottom = i
                    left = j
                    right = j
                    flag_first = True
                elif ((img[i][j][0] != 0) or (img[i][j][1] != 0) or (img[i][j][2] != 0)) :
                    if j < left:
                        left = j
                    if j > right:
                        right = j
                    if i > bottom:
                        bottom = i
        height = bottom - top
        width = right - left
        print(left)
        print(right)
        print(top)
        print(bottom)
        image_clip = img[int(top):(int(top) + int(height)), int(left):(int(left) + int(width))]
        return image_clip
                    
        
    def test(self, ckpt_path, test_roots, batch_size, test_thread_num, save_base):
        with torch.no_grad():
            self.net.eval()
            ckpt = torch.load(ckpt_path)
            state_dict = ckpt['state_dict']
            temp = {}
            for key, value in state_dict.items():
                if str(key).startswith('get_pred_d') == True:
                    continue
                elif str(key).startswith('get_pdepths.get_pdepth_6.') == True:
                    continue
                elif str(key).startswith('pool_k') == True:
                    key = 'cross_modal_feature_fusion.high_level_fusion.FAM_1' + key[len('pool_k'):]
                elif str(key).startswith('pool_r') == True:
                    key = 'cross_modal_feature_fusion.high_level_fusion.FAM_2' + key[len('pool_r'):]
                elif str(key).startswith('fc.att') == True:
                    key = 'get_fused_depth_features.get_fusion_weights.fc' + key[len('fc.att'):]
                elif str(key).startswith('fc.fc') == True:
                    key = 'get_fused_depth_features.get_fusion_weights.fc' + key[len('fc.fc'):]
                elif str(key).startswith('get_cmprs_r.') == True:
                    key = 'cross_modal_feature_fusion.compr_I.' + key[len('get_cmprs_r.'):]
                elif str(key).startswith('get_cmprs_d.') == True:
                    key = 'get_fused_depth_features.compr_Do.' + key[len('get_cmprs_d.'):]
                elif str(key).startswith('get_cmprs_dp.') == True:
                    key = 'get_fused_depth_features.compr_De.' + key[len('get_cmprs_dp.'):]
                elif str(key).startswith('get_pdepths.get_pd_6.') == True:
                    key = 'estimate_depth.get_feat.' + key[len('get_pdepths.get_pd_6.'):]
                elif str(key).startswith('get_pdepths.get_cmprs_pd.') == True:
                    key = 'estimate_depth.compr.' + key[len('get_pdepths.get_cmprs_pd.'):]
                elif str(key).startswith('get_pdepths.up_pd.') == True:
                    key = 'estimate_depth.u_decoder.' + key[len('get_pdepths.up_pd.'):]
                elif str(key).startswith('get_att.conv') == True:
                    key = 'cross_modal_feature_fusion.high_level_fusion.transform.' + key[len('get_att.conv.'):]
                elif str(key).startswith('up_kb.') == True:
                    key = 'cross_modal_feature_fusion.low_level_fusion.u_decoder_boundary.' + key[len('up_kb.'):]
                elif str(key).find('up.') != -1:
                    if str(key).find('bdry_conv.') != -1:
                        key = str(key).replace('up.', 'cross_modal_feature_fusion.low_level_fusion.boundary_enhance.')
                        key = str(key).replace('bdry_conv.', 'conv.')
                    else:
                        key = str(key).replace('up.', 'cross_modal_feature_fusion.low_level_fusion.u_decoder_saliency.')
                temp[key] = value
            state_dict = temp
            self.net.load_state_dict(state_dict)
            torch.save({'state_dict': self.net.state_dict()}, './new.pth')
            test_loader = get_loader(roots=test_roots, request=('img', 'depth', 'name'), batch_size=batch_size, num_thread=test_thread_num)
            
            if exists(save_base) == False:
                makedirs(save_base)
            
            for data_batch in test_loader:
                img, depth, names = data_batch['img'].cuda(), data_batch['depth'].cuda(), data_batch['name']
                predictions = self.net(img, depth)
                for i in range(len(predictions)):
                    batch_preds = predictions[i].permute(0, 2, 3, 1).cpu()  # [N, 1, 224, 224]
                    batch_path_list = [join(save_base, name +'.jpg'.format(i=i)) for name in names]
                    for j in range(len(batch_path_list)):
                        pred_path = batch_path_list[j]
                        
                        origin = pred_path.replace("result","img")
                        print(origin)
                        img_org = cv2.imread(origin,cv2.IMREAD_COLOR)

                        img_org = cv2.resize(img_org,(224,224))
                        preds = self.image_multiply(img_org, batch_preds[j])
                        preds = self.image_crop(preds)
                        preds = cv2.resize(preds,(224,224))
                        cv2.imwrite(filename=pred_path, img=np.array(preds))