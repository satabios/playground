#Meta Pruning


import torch
from torchvision.models import resnet18
import torch_pruning as tp


import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
import torch.optim as optim
import re
from torchsummary import summary
import torch_pruning as tp

from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
import torch.optim as optim




def reformat_layer_name(str_data):

    split_data = str_data.split('.')

    for ind in range(len(split_data)):
        data = split_data[ind]
        if (data.isdigit()):
            split_data[ind] = "[" + data + "]"
    final_string = '.'.join(split_data)

    iters_a = re.finditer(r'[a-zA-Z]\.\[', final_string)
    indices = [m.start(0) + 1 for m in iters_a]
    iters = re.finditer(r'\]\.\[', final_string)
    indices.extend([m.start(0) + 1 for m in iters])

    final_string = list(final_string)
    final_string = [final_string[i] for i in range(len(final_string)) if i not in indices]

    str_data = ''.join(final_string)

    

    return str_data

import random

def generate_random_array(n):
    random_array = []
    
    # Generate a random number of values between 0 and n
    num_values = random.randint(0, n)
    
    # Generate the random array with unique values
    row = random.sample(range(0, n+1), num_values)
    random_array.append(row)
    
    return random_array[0]







fusing_layers = [
    'Conv2d',
    'BatchNorm2d',
    'ReLU',
    'Linear',
    'BatchNorm1d',
]


def summary_string_fixed(model, all_layers, input_size, model_name=None, batch_size=-1, dtypes=None):
    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)
        
    def name_fixer(names):

        return_list = []
        for string in names:
            matches = re.finditer(r'\.\[(\d+)\]', string)
            pop_list = [m.start(0) for m in matches]
            pop_list.sort(reverse=True)
            if len(pop_list) > 0:
                string = list(string)
                for pop_id in pop_list:
                    string.pop(pop_id)
                string = ''.join(string)
            return_list.append(string)
        return return_list

    def register_hook(module, module_idx):
        def hook(module, input, output):
            nonlocal module_idx
            m_key = all_layers[module_idx][0]
            m_key = model_name + "." + m_key

            try:
                eval(m_key)
            except:
                m_key = name_fixer([m_key])[0]

            summary[m_key] = OrderedDict()
            summary[m_key]["type"] = str(type(module)).split('.')[-1][:-2]
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
                summary[m_key]["weight_shape"] = module.weight.shape
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    if isinstance(input_size, tuple):
        input_size = [input_size]

    x = [torch.rand(2, *in_size).type(dtype)
         for in_size, dtype in zip(input_size, dtypes)]

    summary = OrderedDict()
    hooks = []

    for module_idx, (layer_name, module) in enumerate(all_layers):
        register_hook(module, module_idx)

    model(*x)

    for h in hooks:
        h.remove()

    return summary

def layer_mapping(model):
    
    def get_all_layers(model, parent_name=''):
        layers = []

        def reformat_layer_name(str_data):
            try:
                split_data = str_data.split('.')
                for ind in range(len(split_data)):
                    data = split_data[ind]
                    if (data.isdigit()):
                        split_data[ind] = "[" + data + "]"
                final_string = '.'.join(split_data)

                iters_a = re.finditer(r'[a-zA-Z]\.\[', final_string)
                indices = [m.start(0) + 1 for m in iters_a]
                iters = re.finditer(r'\]\.\[', final_string)
                indices.extend([m.start(0) + 1 for m in iters])

                final_string = list(final_string)
                final_string = [final_string[i] for i in range(len(final_string)) if i not in indices]

                str_data = ''.join(final_string)

            except:
                pass

            return str_data

        for name, module in model.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            test_name = "model." + full_name
            try:
                eval(test_name)
                layers.append((full_name, module))
            except:
                layers.append((reformat_layer_name(full_name), module))
            if isinstance(module, nn.Module):
                layers.extend(get_all_layers(module, parent_name=full_name))
        return layers
    all_layers = get_all_layers(model)
    model_summary = summary_string_fixed(model, all_layers, (3, 64, 64), model_name='model')  # , device="cuda")

    name_type_shape = []
    for key in model_summary.keys():
        data = model_summary[key]
        if ("weight_shape" in data.keys()):
            name_type_shape.append([key, data['type'], data['weight_shape'][0]])
        #     else:
    #         name_type_shape.append([key, data['type'], 0 ])
    name_type_shape = np.asarray(name_type_shape)

    name_list = name_type_shape[:, 0]

    r_name_list = np.asarray(name_list)
    random_picks = np.random.randint(0, len(r_name_list), 10)
    test_name_list = r_name_list[random_picks]
    eval_hit = False
    for layer in test_name_list:
        try:
            eval(layer)

        except:
            eval_hit = True
            break
    if (eval_hit):
        fixed_name_list = name_fixer(r_name_list)
        name_type_shape[:, 0] = fixed_name_list

    layer_types = name_type_shape[:, 1]
    layer_shapes = name_type_shape[:, 2]
    mapped_layers = {'model_layer': [], 'Conv2d_BatchNorm2d_ReLU': [], 'Conv2d_BatchNorm2d': [], 'Linear_ReLU': [],
                     'Linear_BatchNorm1d': []}

    def detect_sequences(lst):
        i = 0
        while i < len(lst):

            if i + 2 < len(lst) and [l for l in lst[i: i + 3]] == [
                fusing_layers[0],
                fusing_layers[1],
                fusing_layers[2],
            ]:
                test_layer = layer_shapes[i: i + 2]
                if (np.all(test_layer == test_layer[0])):
                    mapped_layers['Conv2d_BatchNorm2d_ReLU'].append(
                        np.take(name_list, [i for i in range(i, i + 3)]).tolist()
                    )
                    i += 3

            elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                fusing_layers[0],
                fusing_layers[1],
            ]:
                test_layer = layer_shapes[i: i + 2]
                if (np.all(test_layer == test_layer[0])):
                    mapped_layers['Conv2d_BatchNorm2d'].append(
                        np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                    )
                    i += 2
            # if i + 1 < len(lst) and [ type(l) for l in lst[i:i+2]] == [fusing_layers[0], fusing_layers[2]]:
            #     detected_sequences.append(np.take(name_list,[i for i in range(i,i+2)]).tolist())
            #     i += 2
            # elif i + 1 < len(lst) and [ type(l) for l in lst[i:i+2]] == [fusing_layers[1], fusing_layers[2]]:
            #     detected_sequences.append(np.take(name_list,[i for i in range(i,i+2)]).tolist())
            #     i += 2
            elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                fusing_layers[3],
                fusing_layers[2],
            ]:
                mapped_layers['Linear_ReLU'].append(
                    np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                )
                i += 2
            elif i + 1 < len(lst) and [l for l in lst[i: i + 2]] == [
                fusing_layers[3],
                fusing_layers[4],
            ]:
                mapped_layers['Linear_BatchNorm1d'].append(
                    np.take(name_list, [i for i in range(i, i + 2)]).tolist()
                )
                i += 2
            else:
                i += 1

    detect_sequences(layer_types)

    for keys, value in mapped_layers.items():
        mapped_layers[keys] = np.asarray(mapped_layers[keys])

    mapped_layers['name_type_shape'] = name_type_shape
    # self.mapped_layers = mapped_layers

    # CWP
    keys_to_lookout = ['Conv2d_BatchNorm2d_ReLU', 'Conv2d_BatchNorm2d']
    pruning_layer_of_interest, qat_layer_of_interest = [], []

    # CWP or QAT Fusion Layers
    for keys in keys_to_lookout:
        data = mapped_layers[keys]
        if (len(data) != 0):
            qat_layer_of_interest.append(data)
    mapped_layers['qat_layers'] = np.asarray(qat_layer_of_interest)

    return mapped_layers

def cwp_possible_layers(layer_name_list):
    possible_indices = []
    # for idx in range(len(layer_shape_list)):
    idx = 0
    while idx < len(layer_name_list):
        
        current_value = layer_name_list[idx]
        layer_shape = eval(current_value).weight.shape
        curr_merge_list = []
        curr_merge_list.append([current_value, 0])
        hit_catch = False
        for internal_idx in range(idx + 1, len(layer_name_list) - 1):


            new_layer = layer_name_list[internal_idx]
            new_layer_shape = eval(new_layer).weight.shape

            if (len(new_layer_shape) == 4):
                curr_merge_list.append([new_layer, 0])
            if (layer_shape[0] == new_layer_shape[1]):
                hit_catch = True
                break

            elif (len(new_layer_shape) == 1):
            #                 ipdb.set_trace()
                curr_merge_list[len(curr_merge_list) - 1][1] = new_layer

        possible_indices.append(curr_merge_list)
        if (hit_catch == True):

            idx = internal_idx
        else:
            idx += 1

    return possible_indices

##########################


mapped_layers = layer_mapping(pruned_model)
name_list = mapped_layers['name_type_shape'][:, 0]
conv_list = name_list[mapped_layers['name_type_shape'][:, 1]=='Conv2d']



##########################
model = VGG()
checkpoint = torch.load('./vgg.cifar.pretrained.pth',map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

example_inputs = torch.randn(1, 3, 32, 32)

# 1. Importance criterion
imp  = tp.importance.MagnitudeImportance(p=2, target_types=[torch.nn.Conv2d])

# 2. Initialize a pruner with the model and the importance criterion
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) :
        ignored_layers.append(m) # DO NOT prune the final classifier!
        
pruning_ratio_dict= { model.backbone.conv0 : 0.40000000000000013, model.backbone.conv1 : 0.15000000000000002, model.backbone.conv2: 0.1, model.backbone.conv3: 0.15000000000000002, model.backbone.conv4 : 0.1, model.backbone.conv5 : 0.1, model.backbone.conv6 : 0.20000000000000004} 


pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
    model,
    example_inputs,
    importance=imp,
    pruning_ratio_dict = pruning_ratio_dict,
    ignored_layers=ignored_layers,
)

pruner.step()
model.zero_grad()

# finetune the pruned model here
# finetune(model)
# ...
