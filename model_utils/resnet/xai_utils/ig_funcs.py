from copy import deepcopy
import torch
from torch import nn



def integrated_gradient(model, data_dict):
    # Basic setting
    baseline_method='gaussian'
    num_step = 50
    alphas = [1/num_step*(i+1) for i in range(num_step)]
    model.eval()
    
    # Buffer the input
    model_input = deepcopy(data_dict['input'])
    
    # Make baseline
    baseline = {}
    for data_type, data_type_set in data_dict['input'].items():
        for data_name, data_content in data_type_set.items():
            baseline[data_name] = {
                'gaussian': torch.randn_like,
                'zero': torch.zeros_like,
                'uniform': torch.rand_like,
            }.get(baseline_method)(data_content['data'])
    
    # Start iteration on alphas
    grad_dict = {}
    for alpha in alphas:
        # Interpolate data
        for data_type, data_type_set in data_dict['input'].items():
            if data_type not in grad_dict: grad_dict[data_type] = {}
            for data_name in data_type_set:
                # Create a list for the specific data
                if data_name not in grad_dict[data_type]:
                    grad_dict[data_type][data_name] = []
                # Replace input with a parameter that receives gradients
                data_dict['input'][data_type][data_name]['data'] = nn.Parameter(
                    model_input[data_type][data_name]['data'] * alpha +\
                    baseline[data_name] * (1-alpha),
                    requires_grad = True
                )
        
        # Compute gradients
        model.zero_grad()
        loss = model(data_dict)['main_loss']
        loss.backward()
        
        # Collect gradients
        for data_type, data_type_set in data_dict['input'].items():
            for data_name, data_content in data_type_set.items():
                grad_dict[data_type][data_name].append(
                    data_content['data'].grad.detach()
                )
                
    # Summarize gradients
    output_xai_map = {}
    for data_type, data_type_set in data_dict['input'].items():
        output_xai_map[data_type] = {}
        for data_name, data_content in data_type_set.items():
            # Stack the gradients
            total_gradients = torch.stack(grad_dict[data_type][data_name], dim=0)
            # Average of neighboring gradients (like in integral)
            avg_grads_epsilon = (total_gradients[:-1] + total_gradients[1:]) / 2
            # Average across the gradients, making integral calculation easy
            avg_grads = avg_grads_epsilon.mean(dim=0)
            # Multiply it with difference between baseline and original image
            output_xai_map[data_type][data_name] = avg_grads * (
                model_input[data_type][data_name]['data'] - baseline[data_name]
            )
            
    # Restore the data_dict
    data_dict['input'] = model_input

    return output_xai_map