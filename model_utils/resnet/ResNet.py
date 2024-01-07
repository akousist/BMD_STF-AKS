from torchvision.models import resnet50, ResNet50_Weights
from copy import deepcopy
from torch import nn, no_grad
import torch
from pathlib import Path
from .xai_utils.ig_funcs import integrated_gradient



class ResNet(nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.config = model_config
        self.data_config = data_config
        
        ### Check data signature
        _check_data_config(data_config)
        
        ### Update output dimension
        self.config.update({ 'output_dims': [
            len(params['levels']) for params in data_config['output']['categorical'].values()
        ] })
        
        ### Neural network
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        output_dim = sum(self.config['output_dims'])
        mid_layer = int((output_dim*2048)**0.5) # Geometrical mean
        self.classifier = nn.Sequential(
            nn.Linear(2048, mid_layer, bias=True),
            nn.Dropout(p=0.1),
            nn.Linear(mid_layer, output_dim, bias=True),
        )
        
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.init_weights()
        
        return None
        
        
        
    def forward(self, data_dict):
        ### Unpack data
        img_key = list(data_dict['input']['image'].keys())[0]
        inp_img = data_dict['input']['image'][img_key]['data']
        labels = data_dict['output']['categorical']
        
        ### Put through model
        logits = self.classifier(self.backbone(inp_img))
        logit_list = torch.split(logits, self.config['output_dims'], dim=-1)
        loss_list = [
            self.loss_func(logit, label['data'][:, 0])
            for logit, label in zip(logit_list, labels.values())
        ]
        
        data_dict['main_loss'] = sum(loss_list)
        data_dict['length'] = inp_img.size(0)

        ### Put prediction
        for key, logit in zip(labels, logit_list):
            labels[key]['pred'] = logit.detach().argmax(dim=-1, keepdim=True)
            labels[key]['prob'] = logit.detach().softmax(dim=-1)
        
        return data_dict
        
        
    @no_grad()
    def infer(self, data_dict):
        ### Unpack data
        img_key = list(data_dict['input']['image'].keys())[0]
        inp_img = data_dict['input']['image'][img_key]['data']
        data_dict.update({'output': self.data_config['output']})
        labels = data_dict['output']['categorical']
        
        ### Put through model
        embedding = self.backbone(inp_img)
        logit_list = torch.split(self.classifier(embedding), self.config['output_dims'], dim=-1)

        ### Put prediction
        for key, logit in zip(labels, logit_list):
            labels[key]['pred'] = logit.argmax(dim=-1, keepdim=True)
            labels[key]['prob'] = logit.softmax(dim=-1)
        if 'unsup-output' in data_dict:
            for key in data_dict['unsup-output']['vector']:
                data_dict['unsup-output']['vector'][key]['gen'] = embedding.detach()
                break
                
        return data_dict
        
        
        
    def explain(self, data_dict):
        ### Infer the data it has no grad
        if any(['pred' not in data_dict['output']['categorical'].values()]):
            self.eval()
            data_dict = self.infer(data_dict)
    
        ### Unpack data
        img_key = list(data_dict['input']['image'].keys())[0]
        inp_img = data_dict['input']['image'][img_key]['data']
        labels = data_dict['output']['categorical'].keys()
        
        ### xAI workflow on each output
        labels_buffer = deepcopy(data_dict['output']['categorical'])
        xai_maps = {}
        for key in labels:
            # eliminate unused labels
            for other_key in labels:
                if other_key == key:
                    data_dict['output']['categorical'][key]['data'] =\
                        data_dict['output']['categorical'][key]['pred']
                else: data_dict['output']['categorical'][key]['data'][:, :] = -100
            
            # put into xai function
            xai_maps[key] = integrated_gradient(self, data_dict)
            
            # restore label
            data_dict['output']['categorical'].update(deepcopy(labels_buffer))
            
        for key in labels:
            data_dict['output']['categorical'][key]['xai'] = xai_maps[key]
            
        return data_dict
        
        
        
        
        
        
    def init_weights(self):
        # self.backbone.load_state_dict( torch.load(
            # Path(__file__).parent / 'resnet50_pretrained.pt', map_location=map
        # ) )
        for name, param in self.classifier.named_parameters():
            if 'weight' in name: nn.init.xavier_normal_(param)
            elif 'bias' in name: nn.init.zeros_(param)
            else: raise NotImplementedError(f'Undefined init method for {name}')
        return None
    
    
    
def _check_data_config(data_config):
    pass
    
    
    
