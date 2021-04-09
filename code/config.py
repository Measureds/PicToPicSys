from os.path import join

test_thread_num = 4  # Thread-num used for the test
#device = '0'  # GPU index
device = ('cpu') #'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 10  # Batch-size used for the test

checkpoint_path = './CDNet.pth'  # The file path of pre-trained checkpoint, e.g., '/mnt/jwd/code/CDNet.pth'
img_base = './img'  # RGB-images directory path, e.g., '/mnt/jwd/data/images/'
depth_base = './depth'  # Depth-maps directory path, e.g., '/mnt/jwd/data/depths/'
save_base = './result'  #  The directory path used to save the predictions, e.g., './Predictions/'

test_roots = {'img': img_base,
              'depth': depth_base}