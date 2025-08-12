"""Detailed explanation and analysis functions for PIGNet predictions."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem

ENERGY_NAMES = {0: "vdW", 1: "H-bond", 2: "Metal-ligand", 3: "Hydrophobic", 4: "Ionic"}


def generate_detailed_explanation(
    core,
    protein_path: str,
    ligand_path: str,
    energy_breakdown: bool = False,
    fragment_analysis: bool = False,
    atom_analysis: bool = False,
    format: str = "text",
) -> str | dict | ET.Element:
    """Generate detailed explanation of predictions with various analysis options.

    Args:
        core: PIGNetCore instance
        protein_path: Path to protein PDB file
        ligand_path: Path to ligand file
        energy_breakdown: Include energy component breakdown
        fragment_analysis: Include fragment-residue interaction analysis
        atom_analysis: Include atom-atom interaction analysis
        format: Output format ('text', 'json', or 'xml')

    Returns:
        Formatted explanation in requested format
    """
    # Get basic prediction with residue tracking
    result = core.predict_complex(
        protein_pdb=protein_path,
        ligand_sdf=ligand_path,
        write_output=False,
        detailed_interactions=energy_breakdown,
        track_residues=fragment_analysis or atom_analysis,
    )

    # Prepare explanation data structure
    explanation_data = {
        "protein": str(protein_path),
        "ligand": str(ligand_path),
        "device": str(core.device),
        "ensemble": core.ensemble,
        "models": [str(Path(p).name) for p in core.model_paths],
        "prediction": {
            "affinity": result["affinity"],
            "unit": result["affinity_unit"],
            "conformations": result["conformations"],
        },
    }

    # Add energy breakdown if requested
    if energy_breakdown and "interactions" in result:
        energy_data = {}
        for interaction in result["interactions"]:
            if "energy_components" in interaction:
                for comp_name, value in interaction["energy_components"].items():
                    if comp_name not in energy_data:
                        energy_data[comp_name] = []
                    energy_data[comp_name].append(value)

        # Average across models
        explanation_data["energy_breakdown"] = {
            name: np.mean(values) for name, values in energy_data.items()
        }

    # Add fragment analysis if requested
    if fragment_analysis and "residue_info" in result:
        residue_info = result["residue_info"]
        explanation_data["fragment_analysis"] = {
            "pocket_residues": residue_info["pocket_residues"],
            "ligand_fragments": residue_info["ligand_fragments"],
            "num_pocket_residues": len(residue_info["pocket_residues"]),
            "num_fragments": len(residue_info["ligand_fragments"]),
        }

    # Add atom-level analysis if requested
    if atom_analysis:
        # This would require more detailed implementation in the model
        explanation_data["atom_analysis"] = {
            "note": "Atom-level analysis requires model modifications for full implementation"
        }

    # Format output based on requested format
    if format == "json":
        return explanation_data
    elif format == "xml":
        return _dict_to_xml(explanation_data, "explanation")
    else:  # text format
        return _format_text_explanation(explanation_data)


def _format_text_explanation(data: dict) -> str:
    """Format explanation data as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("PIGNet Prediction Analysis")
    lines.append("=" * 60)

    # Basic information
    lines.append(f"\nProtein: {data['protein']}")
    lines.append(f"Ligand: {data['ligand']}")
    lines.append(f"Device: {data['device']}")
    lines.append(f"Ensemble: {data['ensemble']}")
    lines.append(f"Models: {', '.join(data['models'])}")

    # Prediction results
    lines.append("\n" + "-" * 40)
    lines.append("Prediction Results:")
    lines.append("-" * 40)
    pred = data["prediction"]
    lines.append(f"Binding Affinity: {pred['affinity']:.3f} {pred['unit']}")
    lines.append(f"Conformations analyzed: {pred['conformations']}")

    # Energy breakdown if available
    if "energy_breakdown" in data:
        lines.append("\n" + "-" * 40)
        lines.append("Energy Component Breakdown:")
        lines.append("-" * 40)
        for comp, value in data["energy_breakdown"].items():
            lines.append(f"  {comp}: {value:.3f} kcal/mol")

    # Fragment analysis if available
    if "fragment_analysis" in data:
        lines.append("\n" + "-" * 40)
        lines.append("Fragment Analysis:")
        lines.append("-" * 40)
        frag = data["fragment_analysis"]
        lines.append(f"Number of pocket residues: {frag['num_pocket_residues']}")
        lines.append(f"Number of ligand fragments: {frag['num_fragments']}")

        if frag["pocket_residues"]:
            lines.append("\nKey pocket residues:")
            for i, res in enumerate(frag["pocket_residues"][:10]):  # Show first 10
                lines.append(f"  {i + 1}. {res}")
            if len(frag["pocket_residues"]) > 10:
                lines.append(f"  ... and {len(frag['pocket_residues']) - 10} more")

    # Atom analysis if available
    if "atom_analysis" in data:
        lines.append("\n" + "-" * 40)
        lines.append("Atom-level Analysis:")
        lines.append("-" * 40)
        if "note" in data["atom_analysis"]:
            lines.append(f"  {data['atom_analysis']['note']}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def _dict_to_xml(data: dict, root_name: str) -> ET.Element:
    """Convert dictionary to XML element."""
    root = ET.Element(root_name)

    def add_element(parent, key, value):
        if isinstance(value, dict):
            child = ET.SubElement(parent, key)
            for k, v in value.items():
                add_element(child, k, v)
        elif isinstance(value, list):
            child = ET.SubElement(parent, key)
            for i, item in enumerate(value):
                add_element(child, f"item_{i}", item)
        else:
            child = ET.SubElement(parent, key)
            child.text = str(value)

    for key, value in data.items():
        add_element(root, key, value)

    return root


def analyze_interactions(
    protein_mol: Chem.Mol,
    ligand_mol: Chem.Mol,
    distance_cutoff: float = 5.0,
) -> dict[str, list[tuple[int, int, float]]]:
    """Analyze interactions between protein and ligand.

    Args:
        protein_mol: RDKit Mol object for protein
        ligand_mol: RDKit Mol object for ligand
        distance_cutoff: Distance cutoff for interactions (Angstroms)

    Returns:
        Dictionary mapping interaction types to lists of (protein_atom, ligand_atom, distance)
    """
    interactions = {
        "hydrogen_bonds": [],
        "hydrophobic": [],
        "pi_stacking": [],
        "salt_bridges": [],
    }

    # Get conformers
    protein_conf = protein_mol.GetConformer()
    ligand_conf = ligand_mol.GetConformer()

    # Analyze each protein-ligand atom pair
    for p_idx in range(protein_mol.GetNumAtoms()):
        p_atom = protein_mol.GetAtomWithIdx(p_idx)
        p_pos = protein_conf.GetAtomPosition(p_idx)

        for l_idx in range(ligand_mol.GetNumAtoms()):
            l_atom = ligand_mol.GetAtomWithIdx(l_idx)
            l_pos = ligand_conf.GetAtomPosition(l_idx)

            # Calculate distance
            distance = p_pos.Distance(l_pos)

            if distance > distance_cutoff:
                continue

            # Check for hydrogen bonds
            if _is_hbond_donor(p_atom) and _is_hbond_acceptor(l_atom):
                interactions["hydrogen_bonds"].append((p_idx, l_idx, distance))
            elif _is_hbond_acceptor(p_atom) and _is_hbond_donor(l_atom):
                interactions["hydrogen_bonds"].append((p_idx, l_idx, distance))

            # Check for hydrophobic interactions
            if _is_hydrophobic(p_atom) and _is_hydrophobic(l_atom):
                interactions["hydrophobic"].append((p_idx, l_idx, distance))

            # Check for salt bridges
            if _is_charged(p_atom) and _is_charged(l_atom):
                p_charge = p_atom.GetFormalCharge()
                l_charge = l_atom.GetFormalCharge()
                if p_charge * l_charge < 0:  # Opposite charges
                    interactions["salt_bridges"].append((p_idx, l_idx, distance))

    return interactions


def _is_hbond_donor(atom: Chem.Atom) -> bool:
    """Check if atom can be hydrogen bond donor."""
    if atom.GetSymbol() in ["N", "O", "S"]:
        # Check if it has hydrogen attached
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == "H":
                return True
    return False


def _is_hbond_acceptor(atom: Chem.Atom) -> bool:
    """Check if atom can be hydrogen bond acceptor."""
    return atom.GetSymbol() in ["N", "O", "S", "F"]


def _is_hydrophobic(atom: Chem.Atom) -> bool:
    """Check if atom is hydrophobic."""
    symbol = atom.GetSymbol()
    if symbol == "C":
        # Carbon is hydrophobic if all neighbors are C or H
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() not in ["C", "H"]:
                return False
        return True
    return False


def _is_charged(atom: Chem.Atom) -> bool:
    """Check if atom is charged."""
    return atom.GetFormalCharge() != 0


def extract_pocket_residues(
    protein_mol: Chem.Mol,
    ligand_mol: Chem.Mol,
    distance_cutoff: float = 5.0,
) -> list[str]:
    """Extract residues in the binding pocket.

    Args:
        protein_mol: RDKit Mol object for protein
        ligand_mol: RDKit Mol object for ligand
        distance_cutoff: Distance cutoff for pocket definition (Angstroms)

    Returns:
        List of residue identifiers in the pocket
    """
    pocket_residues = set()

    protein_conf = protein_mol.GetConformer()
    ligand_conf = ligand_mol.GetConformer()

    for p_idx in range(protein_mol.GetNumAtoms()):
        p_pos = protein_conf.GetAtomPosition(p_idx)

        for l_idx in range(ligand_mol.GetNumAtoms()):
            l_pos = ligand_conf.GetAtomPosition(l_idx)

            if p_pos.Distance(l_pos) <= distance_cutoff:
                atom = protein_mol.GetAtomWithIdx(p_idx)
                res_info = atom.GetPDBResidueInfo()
                if res_info:
                    residue_id = (
                        f"{res_info.GetResidueName()}"
                        f"{res_info.GetResidueNumber()}"
                        f"{res_info.GetChainId()}"
                    )
                    pocket_residues.add(residue_id)
                break  # This protein atom is in the pocket

    return sorted(list(pocket_residues))


def fragment_ligand(
    ligand_mol: Chem.Mol,
    method: str = "recap",
) -> list[Chem.Mol]:
    """Fragment ligand into smaller pieces.

    Args:
        ligand_mol: RDKit Mol object for ligand
        method: Fragmentation method ('recap', 'brics', or 'simple')

    Returns:
        List of fragment molecules
    """
    if method == "recap":
        from rdkit.Chem import Recap

        hierarchy = Recap.RecapDecompose(ligand_mol)
        fragments = [node.mol for node in hierarchy.GetAllChildren().values()]
    elif method == "brics":
        from rdkit.Chem import BRICS

        fragments = list(BRICS.BRICSDecompose(ligand_mol))
        fragments = [Chem.MolFromSmiles(smiles) for smiles in fragments]
    else:  # simple fragmentation
        # Break at rotatable bonds
        fragments = []
        rotatable_bonds = ligand_mol.GetSubstructMatches(
            Chem.MolFromSmarts("[!#1]~[!#1]")
        )

        if len(rotatable_bonds) > 0:
            # For simplicity, just break at the first rotatable bond
            bond_idx = ligand_mol.GetBondBetweenAtoms(
                rotatable_bonds[0][0], rotatable_bonds[0][1]
            ).GetIdx()

            # Fragment the molecule
            frags = Chem.FragmentOnBonds(ligand_mol, [bond_idx])
            fragments = Chem.GetMolFrags(frags, asMols=True)
        else:
            fragments = [ligand_mol]

    return fragments if fragments else [ligand_mol]


def get_energy_breakdown(
    model,
    batch: torch.Tensor,
    detailed: bool = False,
) -> dict[str, float]:
    """Extract energy component breakdown from model prediction.

    Args:
        model: PIGNet model
        batch: Data batch
        detailed: Whether to include detailed breakdown

    Returns:
        Dictionary with energy components
    """
    energy_components = {}

    # This would need to be implemented based on the actual model architecture
    # For now, return a placeholder structure
    with torch.no_grad():
        # Run model forward pass if needed
        if not hasattr(model, "last_energy_components"):
            model.predict_step(batch)

        # Extract energy components (placeholder implementation)
        energy_components = {
            "vdw": 0.0,
            "hbond": 0.0,
            "metal_ligand": 0.0,
            "hydrophobic": 0.0,
            "pi_stacking": 0.0,
            "salt_bridge": 0.0,
            "total": 0.0,
        }

        if detailed:
            # Add per-residue or per-atom breakdown
            energy_components["per_residue"] = {}
            energy_components["per_atom"] = {}

    return energy_components
