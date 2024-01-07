##### Import both ResNet and XGBoost to build the inference pipeline #####
import cv2
import numpy as np
from torch import as_tensor, load as torch_load
from torchvision.transforms import Normalize
from .resnet.ResNet import ResNet



normalizer = Normalize(0.212, 0.224)
resnet_data_config = {
    "input": { "image": { "Filename": {} } },
    "output": { "categorical": { "BMD": {
        "levels": [ "Normal", "Osteopenia", "Osteoporosis" ]
    } } }
}
resnet_model = ResNet({}, resnet_data_config)
resnet_model.eval()



def image2tensor(inimg):
    # Resize image
    resize_target = (512, int(512*inimg.shape[0]/inimg.shape[1]))
    inimg = cv2.resize(inimg, resize_target)
    if inimg.shape[0] < 768: # pad
        upad = (768-inimg.shape[0])//2
        dpad = 768-inimg.shape[0]-upad
        inimg = np.pad(inimg, ((upad, dpad), (0, 0), (0, 0)))
    elif inimg.shape[0] > 768: # crop
        ucrop = (inimg.shape[0]-768)//2
        dcrop = inimg.shape[0]-768-ucrop
        inimg = inimg[ucrop:-dcrop, :, :]
    else: pass
    
    # Normalize image
    tnsr_img = as_tensor(inimg).float().permute(-1, 0, 1)[None, ...]
    return normalizer(tnsr_img/255)
    
    
    
def resnet_inference(img_tnsr):
    input_dict = { "input": { "image": { "Filename": {
        'data': img_tnsr
    } } } }
    output_dict = resnet_model.infer( input_dict )
    return output_dict['output']['categorical']['BMD']
    
    
    
def load_resnet_weight(weight_path, cuda):
    if cuda:
        resnet_model.cuda()
        resnet_model.load_state_dict(torch_load(weight_path))
    else:
        resnet_model.load_state_dict(torch_load(
            weight_path, map_location='cpu'
        ))
    
    


