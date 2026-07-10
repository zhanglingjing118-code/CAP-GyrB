import os
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import * #ZLJ
from model import * #ZLJ
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score, r2_score

from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

def load_best_result(model):
    best_ckpt_path = "E:/LingjingZhang/dnatopo/mtlhignn_test/test/mtl7.0/seed_3412/checkpoints/best_ckpt.pth"
    ckpt = torch.load(best_ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])

    return model

model = HiGNN(in_channels=46,
              hidden_channels=64,
              out_channels=1,
              edge_dim=10,
              num_layers=4,
              dropout=0.0,
              slices=2,
              f_att=True,
              r=1,
              brics=True,
              cl=False)

model = load_best_result(model)

path = "E:/LingjingZhang/dnatopo/brics_b/data/mtl/"
task_type = 'regression'
tasks = 'pIC50(M)'
aic50 = MolDataset(root=path, dataset="a_input_plus_random", task_type=task_type, tasks=tasks)
bic50 = MolDataset(root=path, dataset="b_input_plus_random", task_type=task_type, tasks=tasks)


loader_b = DataLoader(bic50, batch_size=150)


iter_ = iter(loader_b)
batch = next(iter_)

model.eval()  # 关闭dropout
output = model(batch, 1)

pred = output[0].detach().cpu().numpy()
print(pred[-10:])
print(len(pred))
att = output[1][0].detach().numpy()


cross = output[1][1].detach().numpy()
#print(cross)

idx = cross[1]



for i in range():
    num = np.where(idx==i)[0]

    smiles = df_test['smiles'].iloc[i]
    mol = Chem.MolFromSmiles(smiles)
    print(smiles)
    
    try:
        results = np.array(sorted(list(FindBRICSBonds(mol))), dtype=np.long)
        bond_to_break = results[:, 0, :]
        bond_to_break = bond_to_break.tolist()
        with Chem.RWMol(mol) as rwmol:
            for s in bond_to_break:
                rwmol.RemoveBond(*s)
        rwmol = rwmol.GetMol()


        #results

        cluster_idx = []
        Chem.rdmolops.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False, frags=cluster_idx)
        #cluster_idx

        atoms = mol.GetAtoms()

        hit_ats = list(range(0, len(atoms)))

        weight_atom = att[num][cluster_idx]
        weight_atom = (weight_atom - weight_atom.min())/(weight_atom.max() - weight_atom.min())
        #weight_atom

        hit_bonds = []
        weight_bond = []
        for bond in mol.GetBonds():
            aid1 = hit_ats[bond.GetBeginAtomIdx()]
            aid2 = hit_ats[bond.GetEndAtomIdx()]
            hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
            # 不是很严格
            if weight_atom[aid1] == weight_atom[aid2]:
                weight_bond.append(weight_atom[aid1])
            else:
                weight_bond.append(0)
        #weight_bond

        norm = matplotlib.colors.Normalize(vmin=0,vmax=1.5)
        cmap = cm.get_cmap('Oranges')
        plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)

        atom_cols = {}
        for at in hit_ats:
            atom_cols[at] = plt_colors.to_rgba(float(weight_atom[at]))
    
        bond_cols = {}
        for bd in hit_bonds:
            bond_cols[bd] = plt_colors.to_rgba(float(weight_bond[bd]))

        d = Draw.MolDraw2DCairo(1000, 1000)
        #d.SetDPI(1000)
        Draw.PrepareAndDrawMolecule(d, mol, highlightAtoms=hit_ats,
                                   highlightBonds=hit_bonds,
                                  highlightAtomColors=atom_cols,
                                  highlightBondColors=bond_cols)

        d.FinishDrawing()
        # 保存加权分子图的 SVG 文件
        pathway=f'E:/LingjingZhang/dnatopo/brics_b/result/{str(i)}.png'
        with open(pathway, 'wb') as f:
            f.write(d.GetDrawingText())

    except:
        continue
