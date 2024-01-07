import fire, cv2
from pathlib import Path
from model_utils import image2tensor, resnet_inference, load_resnet_weight



def _expected_error(*msg):
    print(' '.join(msg))
    return None



def main(img_path, #ethnicity=None, gender=None,
        resnet_weights='resnet_weights.pt', cuda=False,
    ):
    ### Preprocessing
    # Encode image
    if not (img_path := Path(img_path)).exists():
        return _expected_error("Image does not exists:", str(img_path))
    img = cv2.imread(str(img_path))
    
    input_tensor = image2tensor(img)
    if cuda: input_tensor = input_tensor.cuda()
    
    ### Resnet prediction
    # Load model weights
    if not(resnet_weight_path := Path(resnet_weights)).exists():
        return _expected_error("Model weights missing:", str(resnet_weight_path))
    load_resnet_weight(resnet_weight_path, cuda)
    result_dict = resnet_inference(input_tensor)
    
    ### Show the results
    target_prob = result_dict['prob'][0].tolist()[result_dict['pred'].item()]*100
    target_label = result_dict['levels'][result_dict['pred'].item()]
    print( f"{target_prob:.2f}% chance of {target_label}" )
    
    return None



if __name__=="__main__":
    fire.Fire(main)

