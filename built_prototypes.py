
import os
import sys
import numpy as np
import random
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from   torch import nn
from tqdm import tqdm
# import matplotlib.pyplot as plt
#
# PArse user arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperspherical Multi-prototypes")
    parser.add_argument('-c', dest="classes", default=66, type=int)
    parser.add_argument('-d', dest="dims", default=1024, type=int)
    parser.add_argument('-l', dest="learning_rate", default=0.1, type=float)
    parser.add_argument('-m', dest="momentum", default=0.9, type=float)
    parser.add_argument('-e', dest="epochs", default=10000, type=int,)
    parser.add_argument('-s', dest="seed", default=300, type=int)
    parser.add_argument('-w', dest="wtvfile", default="Semantic_vectors/rams_bert_sem.npy", type=str) #wiki_bert_sem.npy
    parser.add_argument('-per_num',dest="num_proto_per_type", default=3, type=int)
    parser.add_argument('-hd',dest="wtv_dims", default=768, type=int)
    args = parser.parse_args()
    return args


def prototype_loss(prototypes,mask):
    
    intra_class_similarities = torch.matmul(prototypes, prototypes.transpose(1, 2)) # [c,n,h ] [c,h,n]= [c,n,n]
    intra_class_min_similarities = intra_class_similarities.min(dim=-1)[0]
    intra_class_similarity_loss = (1 - intra_class_min_similarities).mean() 
   
    product = torch.matmul(prototypes.view(-1,args.dims),prototypes.view(-1,args.dims).t()) + 1
    product -= 2. * torch.diag(torch.diag(product))
    product[mask.bool()] = -2.0 
    inter_class_similarity = product.max(dim=1)[0].mean()

    loss = intra_class_similarity_loss + inter_class_similarity
    return loss

# #
# Compute the semantic relation loss.
#
def prototype_loss_sem(wtvv,triplets,prototypes):
    
    cos_sem_wtvv = torch.cosine_similarity(wtvv.unsqueeze(1), wtvv.unsqueeze(0), dim=2)
    cos_porto_wtvv = torch.cosine_similarity(prototypes.view(-1,args.dims).unsqueeze(1), prototypes.view(-1,args.dims).unsqueeze(0), dim=-1)
   
    cos_sem_ij = cos_sem_wtvv[triplets[:, 0], triplets[:, 1]]
    cos_sem_ik = cos_sem_wtvv[triplets[:, 0], triplets[:, 3]]

    cos_proto_ij = cos_porto_wtvv[triplets[:, 0], triplets[:, 1]]
    cos_proto_ik = cos_porto_wtvv[triplets[:, 0], triplets[:, 3]]

    oijk = cos_proto_ij - cos_proto_ik
    Sijk_true = torch.where(cos_sem_ij >= cos_sem_ik, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
    Sijk = torch.exp(oijk) / (1 + torch.exp(oijk))
    loss = -Sijk_true * torch.log(Sijk) - (1 - Sijk_true) * torch.log(1 - Sijk)
    
       
    return loss.mean() 


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")


    # Set seed.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize prototypes.
    prototypes = torch.randn(args.classes,args.num_proto_per_type, args.dims).to(device)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=-1))
    # Initialize prototypes and optimizer.
    if os.path.exists(args.wtvfile):
        use_wtv = True
        wtvv = np.load(args.wtvfile)
        wtvv = torch.from_numpy(wtvv).to(device)
        wtvv = wtvv.unsqueeze(1).repeat(1, args.num_proto_per_type, 1).view(-1,args.wtv_dims)
        triplets = [[i,j,i,k] for i in tqdm(range(args.classes)) for j in range(args.classes) for k in range(args.classes) if i != j != k != i]
        triplets = np.array(triplets).astype(int)
    else:
        use_wtv = False

    # Initialize prototypes.
    optimizer = optim.SGD([prototypes], lr=args.learning_rate, \
            momentum=args.momentum)
    mask = torch.zeros(args.classes *args.num_proto_per_type, args.classes*args.num_proto_per_type)
    for i in range(0, args.classes*args.num_proto_per_type, args.num_proto_per_type):
           mask[i:i+args.num_proto_per_type,i:i+args.num_proto_per_type] = 1
    losses = []
    # Optimize for separation.
    for i in tqdm(range(args.epochs)):
        # Compute loss.
        loss1 = prototype_loss(prototypes,mask)
        if use_wtv:
            loss2 = prototype_loss_sem(wtvv,triplets,prototypes)
            loss = loss1 + loss2
        else:
            loss = loss1
        # Update.
        loss.backward()
        optimizer.step()
        # Renormalize prototypes.
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=-1))
        optimizer = optim.SGD([prototypes], lr=args.learning_rate, \
                momentum=args.momentum)
        
        print(str(args.epochs) +''+ str(loss)),
        sys.stdout.flush()
    
    # Store result.
    np.save("prototypes-{}d-{}c_mutil{}_proto_sem_test".format(args.dims, args.classes,args.num_proto_per_type), \
            prototypes.data.cpu().numpy())
