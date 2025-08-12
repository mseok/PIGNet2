from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType

_periodic_table = """\
H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,HE
LI,BE,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,NE
NA,MG,1,1,1,1,1,1,1,1,1,1,AL,SI,P,S,CL,AR
K,CA,SC,TI,V,CR,MN,FE,CO,NI,CU,ZN,GA,GE,AS,SE,BR,KR
RB,SR,Y,ZR,NB,MO,TC,RU,RH,PD,AG,CD,IN,SN,SB,TE,I,XE
CS,BA,1,HF,TA,W,RE,OS,IR,PT,AU,HG,TL,PB,BI,PO,AT,RN
"""
PERIODIC_TABLE = dict()
for i, row in enumerate(_periodic_table.split()):
    for j, symbol in enumerate(row.split(",")):
        PERIODIC_TABLE[symbol] = (i, j)
_lanthanides = "LA,CE,PR,ND,PM,SM,EU,GD,TB,DY,HO,ER,TM,YB,LU"
for symbol in _lanthanides.split(","):
    PERIODIC_TABLE[symbol] = (5, 2)
del _lanthanides, _periodic_table, i, row, j, symbol

PERIODS = list(range(6))
GROUPS = list(range(18))

VDW_RADII = {
    6: 1.90,
    7: 1.8,
    8: 1.7,
    9: 1.5,
    12: 1.2,
    15: 2.1,
    16: 2.0,
    17: 1.8,
    20: 1.2,
    25: 1.2,
    26: 1.2,
    27: 1.2,
    28: 1.2,
    29: 1.2,
    30: 1.2,
    35: 2.0,
    53: 2.2,
}

_metal_atomic_numbers = [
    *range(3, 5),
    *range(11, 14),
    *range(19, 32),
    *range(37, 51),
    *range(55, 84),
    *range(87, 117),
]
_rd_p_table = Chem.GetPeriodicTable()
METALS = {_rd_p_table.GetElementSymbol(num) for num in _metal_atomic_numbers}
del _metal_atomic_numbers, _rd_p_table

H_DONOR_SMARTS = "[!$([#6,H0,-,-2,-3])]"

H_ACCEPTOR_SMARTS = "[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]"

HYDROPHOBES = {"C", "S", "F", "CL", "BR", "I"}

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "X"]

ATOM_DEGREES = list(range(6))

HYBRIDIZATIONS = [
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    None,
]
