import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.SaltRemover import SaltRemover
# from molvs.normalize import Normalizer, Normalization
from molvs import Standardizer
from molvs.fragment import LargestFragmentChooser
from molvs.charge import Reionizer, Uncharger
from molvs.tautomer import TAUTOMER_TRANSFORMS, TAUTOMER_SCORES, MAX_TAUTOMERS, TautomerCanonicalizer, TautomerEnumerator, TautomerTransform
import pandas as pd

data2 =pd.read_csv("./a_input_pro.csv",index_col=None,header=0)

def Standardize(x):
	testsmi = str(x)
	remover=SaltRemover()
	try:
		mol = Chem.MolFromSmiles(testsmi)
		# mol = Normalizer(mol)
		mol = Standardizer().standardize(mol)
		mol = LargestFragmentChooser()(mol)
		mol = Uncharger()(mol)
		mol = TautomerCanonicalizer().canonicalize(mol)
		mol2=remover.StripMol(mol)
		smiles = Chem.MolToSmiles(mol2)
		if mol2.GetNumAtoms()==0:
			return "None"
		else:
			return smiles
	except:
		return "None"



data2['SMILES']=data2['SMILES'].apply(Standardize)

data2 = data2.drop(data2[data2["SMILES"]=="None"].index)

data2 = data2.drop_duplicates(subset=['SMILES'])

smiles1=data2['SMILES'].tolist()

smiles1=list(set(smiles1))


for i in smiles1:
	a=str(i)
	try:
		mol = Chem.MolFromSmiles(a)
		bond_num = mol.GetNumBonds()
		if bond_num==0:
			data2 = data2.drop(data2[data2["SMILES"]==i].index)
	except:
		data2 = data2.drop(data2[data2["SMILES"]==i].index)
		continue



data2.to_csv("./a_input_pro.csv",index=False,header=True)