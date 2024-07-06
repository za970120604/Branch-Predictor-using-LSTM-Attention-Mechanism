from prettytable import PrettyTable
import random
import torch
from torch import nn
from collections import deque

class Container:
    def __init__(self, max_size):
        self.max_size = max_size
        self.container = deque(maxlen=max_size)

    def push(self, item):
        self.container.append(item)

    def pop(self):
        return self.container.pop()

    def __len__(self):
        return len(self.container)
    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def generate_random_address(system_bits):
    address = random.randint(0, 2**system_bits-1)
    # address = 2**64-1
    first_16_bits = (address >> 48) & 0xFFFF
    seocnd_16_bits = (address >> 32) & 0xFFFF
    third_16_bits = (address >> 16) & 0xFFFF
    last_16_bits = address & 0xFFFF
    return first_16_bits, seocnd_16_bits, third_16_bits, last_16_bits

def generate_PC_in_chunks(PC):
    first_16_bits = (PC >> 48) & 0xFFFF
    seocnd_16_bits = (PC >> 32) & 0xFFFF
    third_16_bits = (PC >> 16) & 0xFFFF
    last_16_bits = PC & 0xFFFF
    return first_16_bits, seocnd_16_bits, third_16_bits, last_16_bits

# def generate_PC_set(system_bits, PC_workingset_size = 500, batch_size = 10):
#     batch_PC_sets = []
#     for _ in range(batch_size) :
#         previous_pc_records = []
#         current_branch_PC_chunks = generate_random_address(system_bits)
#         specualtive_target_PC_chunks = generate_random_address(system_bits) ## needs to be modified into current_branch_PC.target_branch_PC

#         for _ in range(PC_workingset_size-2):
#             address_chunks = generate_random_address(system_bits)
#             previous_pc_records.append(address_chunks)

#         previous_pc_records.append(current_branch_PC_chunks)
#         previous_pc_records.append(specualtive_target_PC_chunks)
#         batch_PC_sets.append(previous_pc_records)

#    return torch.tensor(batch_PC_sets)

def create_PCs_chunks(PC, history_table, Speculative_PC, PC_workingset_size):
    previous_pc_records = []
    current_branch_PC_chunks = generate_PC_in_chunks(PC)
    specualtive_target_PC_chunks = generate_PC_in_chunks(Speculative_PC)
    for i in range(0, PC_workingset_size-2):
        previous_PC_chunks = generate_PC_in_chunks(history_table[int(-1-i)])
        previous_pc_records.append(previous_PC_chunks)
    
    previous_pc_records.reverse()
    previous_pc_records.append(current_branch_PC_chunks)
    previous_pc_records.append(specualtive_target_PC_chunks)
    return torch.tensor(previous_pc_records).unsqueeze(0) # create a dummy demnsion on the 0th dim so that we have shape 
                                                          # [1,working set size - 2 + 1 current + 1 speculative ,4(chunks)]

def create_batch_tensor(prediction_in_batch, groundtruth_in_batch, device):
    return torch.cat(prediction_in_batch, dim=0).to(device), torch.cat(groundtruth_in_batch, dim=0).to(device)

if __name__ == "__main__":
    history_table = Container(100)
    for i in range(100):
        history_table.push(i)
    print(create_PCs_chunks(1000, history_table.container, 1024, 10), create_PCs_chunks(1000, history_table.container, 1024, 10)[:,:, 3].shape)


