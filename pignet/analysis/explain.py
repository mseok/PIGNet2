"""Detailed explanation and analysis functions for PIGNet predictions."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch

ENERGY_NAMES = {0: "vdW", 1: "H-bond", 2: "Metal-ligand", 3: "Hydrophobic", 4: "Ionic"}


def generate_detailed_explanation(
    core,
    protein_path: str,
    ligand_path: str,
    energy_breakdown: bool = False,
    fragment_analysis: bool = False,
    atom_analysis: bool = False,
    format: str = "text",
) -> str:
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
    # Get basic prediction
    result = core.predict_complex(
        protein_pdb=protein_path, ligand_sdf=ligand_path, write_output=False
    )

    # Prepare explanation data structure
    explanation_data = {
        "protein": str(protein_path),
        "ligand": str(ligand_path),
        "device": str(core.device),
        "ensemble": core.ensemble,
        "models": [str(Path(p).name) for p in core.model_paths],
        "affinity": {
            "value": float(result["affinity"]),
            "unit": result["affinity_unit"],
            "kcal_mol": float(result["affinity"]),
        },
        "conformations": result.get("conformations", 1),
    }

    # Add per-conformation predictions if available
    predictions_key = (
        "predictions_per_conf"
        if "predictions_per_conf" in result
        else "all_predictions"
    )
    if predictions_key in result:
        explanation_data["conformation_predictions"] = [
            {"conformation": i + 1, "affinity": float(pred)}
            for i, pred in enumerate(result[predictions_key])
        ]

        # Add statistics
        preds = result[predictions_key]
        explanation_data["statistics"] = {
            "mean": float(np.mean(preds)),
            "min": float(np.min(preds)),
            "max": float(np.max(preds)),
            "std": float(np.std(preds)),
        }

    # Energy component breakdown if requested
    if energy_breakdown:
        explanation_data["energy_components"] = get_energy_breakdown(
            core, protein_path, ligand_path
        )

    # Fragment-residue analysis if requested
    if fragment_analysis:
        explanation_data["fragment_residue_interactions"] = (
            get_fragment_residue_analysis(core, protein_path, ligand_path)
        )

    # Atom-atom analysis if requested
    if atom_analysis:
        explanation_data["atom_interactions"] = get_atom_atom_analysis(
            core, protein_path, ligand_path
        )

    # Format output
    if format == "json":
        return json.dumps(explanation_data, indent=2, sort_keys=False)
    elif format == "xml":
        return format_pretty_xml(explanation_data, "PIGNet_Explanation")
    else:  # text format
        return format_text_explanation(explanation_data)


def get_energy_breakdown(core, protein_path: str, ligand_path: str) -> dict[str, Any]:
    """Get energy component breakdown for the prediction."""
    # Use first model's config for data preparation
    conv_range = core.configs[0].model.get("conv_range", (0.0, 10.0))

    # Prepare data
    data_list = core._prepare_data(protein_path, ligand_path, conv_range)

    if not data_list:
        return {"error": "Failed to process complex"}

    energy_breakdown = {
        "total": float(
            core.predict_complex(protein_path, ligand_path, write_output=False)[
                "affinity"
            ]
        ),
        "components": [],
    }

    # Process with each model and average
    all_model_energies = []

    for model in core.models:
        model.eval()

        from torch_geometric.data import Batch

        model_energies = []
        for data, name, _ in data_list:  # Unpack tuple with residue_info
            batch = Batch.from_data_list([data])
            batch = batch.to(core.device)

            with torch.no_grad():
                energies, _ = model(batch)
                model_energies.append(energies)

        # Average across conformations for this model
        if model_energies:
            avg_energies = torch.stack(model_energies).mean(0)
            all_model_energies.append(avg_energies)

    # Average across all models if ensemble
    if all_model_energies:
        if core.ensemble and len(all_model_energies) > 1:
            final_energies = torch.stack(all_model_energies).mean(0)
        else:
            final_energies = all_model_energies[0]

        # Get individual energy components
        for i in range(final_energies.shape[1]):
            if i < len(ENERGY_NAMES):
                energy_breakdown["components"].append(
                    {
                        "type": ENERGY_NAMES[i],
                        "value": float(final_energies[0, i].item()),
                        "unit": "kcal/mol",
                    }
                )

    return energy_breakdown


def get_fragment_residue_analysis(
    core, protein_path: str, ligand_path: str
) -> dict[str, Any]:
    """Get fragment-residue interaction analysis with ensemble averaging."""
    conv_range = core.configs[0].model.get("conv_range", (0.0, 10.0))
    data_list = core._prepare_data(protein_path, ligand_path, conv_range)

    if not data_list:
        return {"error": "Failed to process complex"}

    from torch_geometric.data import Batch

    # Process first conformation
    data, name, _ = data_list[0]  # Unpack tuple with residue_info
    batch = Batch.from_data_list([data])
    batch = batch.to(core.device)

    # Check if we have fragment and residue information
    if not hasattr(batch, "fragment_map") or batch.fragment_map is None:
        return {"error": "No fragment information available in the data"}

    if not hasattr(batch, "residue_map") or batch.residue_map is None:
        return {
            "error": "No residue information available. Ensure protein PDB file contains residue information."
        }

    # Collect fragment-residue energies from all models
    all_fragment_energies = []
    fragment_ids = None
    residue_ids = None

    for model in core.models:
        model.eval()

        # Check if model has the analysis method
        if hasattr(model, "predict_residue_fragment_interactions"):
            with torch.no_grad():
                interactions = model.predict_residue_fragment_interactions(batch)

            # Get results for this conformation
            if name in interactions:
                interaction_data = interactions[name]

                if interaction_data.get("error"):
                    continue

                if interaction_data["fragment_energies"] is not None:
                    all_fragment_energies.append(interaction_data["fragment_energies"])

                    # Store IDs from first successful model
                    if fragment_ids is None:
                        fragment_ids = interaction_data["fragment_ids"]
                        residue_ids = interaction_data["residue_ids"]

    if not all_fragment_energies:
        return {"error": "No models could generate fragment-residue analysis"}

    # Average energies across all models (ensemble)
    if core.ensemble and len(all_fragment_energies) > 1:
        fragment_energies = np.mean(all_fragment_energies, axis=0)
    else:
        fragment_energies = all_fragment_energies[0]

    # Format results
    formatted_interactions = {
        "fragments": int(fragment_ids.shape[0]) if fragment_ids is not None else 0,
        "residues": int(residue_ids.shape[0]) if residue_ids is not None else 0,
        "interactions": [],
        "ensemble_averaged": core.ensemble and len(all_fragment_energies) > 1,
        "num_models_used": len(all_fragment_energies),
    }

    # Get residue names if available
    residue_names_map = {}
    if hasattr(batch, "residue_names") and batch.residue_names is not None:
        residue_list = batch.residue_names
        # Handle case where residue_names might be nested list
        if isinstance(residue_list, list) and len(residue_list) > 0:
            # If it's a list of lists (batch), take the first one
            if isinstance(residue_list[0], list):
                residue_list = residue_list[0]
            for idx, res_name in enumerate(residue_list):
                residue_names_map[idx + 1] = res_name

    # Get fragment SMILES if available
    fragment_smiles_map = {}
    if hasattr(batch, "fragment_smiles") and batch.fragment_smiles is not None:
        smiles_list = batch.fragment_smiles
        if isinstance(smiles_list, list):
            # If it's a list of lists (batch), take the first one
            if len(smiles_list) > 0 and isinstance(smiles_list[0], list):
                smiles_list = smiles_list[0]
            for idx, smiles in enumerate(smiles_list):
                fragment_smiles_map[idx] = smiles

    formatted_interactions["fragment_smiles"] = fragment_smiles_map

    # Get all interactions
    for frag_id in range(fragment_energies.shape[0]):
        for res_id in range(fragment_energies.shape[1]):
            total_energy = float(fragment_energies[frag_id, res_id].sum())

            # Skip interactions with zero or very small energy
            if abs(total_energy) < 0.001:
                continue

            # Skip Res0 (dummy residue for ligand atoms)
            if res_id == 0:
                continue

            interaction = {
                "fragment": int(frag_id),
                "residue": int(res_id),
                "residue_name": residue_names_map.get(res_id, f"Res{res_id}"),
                "total_energy": total_energy,
                "energy_breakdown": {},
            }

            for energy_idx in range(fragment_energies.shape[2]):
                if energy_idx < len(ENERGY_NAMES):
                    energy_val = float(fragment_energies[frag_id, res_id, energy_idx])
                    interaction["energy_breakdown"][ENERGY_NAMES[energy_idx]] = (
                        energy_val
                    )

            formatted_interactions["interactions"].append(interaction)

    # Sort by total energy (most favorable first)
    formatted_interactions["interactions"].sort(
        key=lambda x: x["total_energy"], reverse=False
    )

    return formatted_interactions


def get_atom_atom_analysis(core, protein_path: str, ligand_path: str) -> dict[str, Any]:
    """Get atom-atom interaction analysis."""
    # Similar implementation to fragment analysis but at atom level
    # For brevity, returning placeholder
    return {
        "note": "Atom-atom analysis implementation pending",
        "ligand_atoms": 0,
        "pocket_atoms": 0,
        "significant_interactions": [],
    }


def format_text_explanation(data: dict[str, Any]) -> str:
    """Format explanation data as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("PIGNet Prediction Explanation")
    lines.append("=" * 80)

    lines.append(f"\nProtein: {data['protein']}")
    lines.append(f"Ligand: {data['ligand']}")
    lines.append(f"Device: {data['device']}")

    if data["ensemble"]:
        lines.append(f"\nEnsemble prediction using {len(data['models'])} models:")
        for i, model in enumerate(data["models"]):
            lines.append(f"  {i + 1}. {model}")
    else:
        lines.append(f"\nModel: {data['models'][0]}")

    lines.append("\n" + "=" * 40)
    lines.append("RESULTS")
    lines.append("=" * 40)

    lines.append(
        f"\nBinding Affinity: {data['affinity']['value']:.3f} {data['affinity']['unit']}"
    )
    lines.append(f"                  ({data['affinity']['kcal_mol']:.2f} kcal/mol)")
    lines.append(f"Number of conformations: {data['conformations']}")

    # Conformation predictions
    if "conformation_predictions" in data and len(data["conformation_predictions"]) > 1:
        lines.append("\nPer-conformation predictions:")
        for conf in data["conformation_predictions"]:
            lines.append(
                f"  Conformation {conf['conformation']}: {conf['affinity']:.3f} kcal/mol"
            )

        if "statistics" in data:
            lines.append("\nStatistics:")
            lines.append(f"  Mean: {data['statistics']['mean']:.3f} kcal/mol")
            lines.append(f"  Min: {data['statistics']['min']:.3f} kcal/mol")
            lines.append(f"  Max: {data['statistics']['max']:.3f} kcal/mol")
            lines.append(f"  Std: {data['statistics']['std']:.3f}")

    # Energy component breakdown
    if "energy_components" in data:
        lines.append("\n" + "-" * 40)
        lines.append("ENERGY COMPONENT BREAKDOWN")
        lines.append("-" * 40)

        if "error" in data["energy_components"]:
            lines.append(f"Error: {data['energy_components']['error']}")
        else:
            lines.append(f"Total energy: {data['energy_components']['total']:.3f} kcal/mol")
            lines.append("\nComponents:")
            for comp in data["energy_components"]["components"]:
                lines.append(
                    f"  {comp['type']:15s}: {comp['value']:7.3f} {comp['unit']}"
                )

    # Fragment-residue interactions (tabular format)
    if "fragment_residue_interactions" in data:
        lines.append("\n" + "-" * 40)
        lines.append("FRAGMENT-RESIDUE INTERACTIONS")
        lines.append("-" * 40)

        frag_data = data["fragment_residue_interactions"]
        if "error" in frag_data:
            lines.append(f"Error: {frag_data['error']}")
        else:
            lines.append(f"Fragments: {frag_data['fragments']}")
            lines.append(f"Residues: {frag_data['residues']}")

            if frag_data.get("ensemble_averaged"):
                lines.append(
                    f"\nEnsemble averaging: Yes ({frag_data['num_models_used']} models)"
                )
            else:
                lines.append("\nEnsemble averaging: No (single model)")

            # Display fragment SMILES mapping
            if "fragment_smiles" in frag_data and frag_data["fragment_smiles"]:
                lines.append("\nFragment SMILES:")
                for frag_id, smiles in sorted(frag_data["fragment_smiles"].items()):
                    lines.append(f"  Frag{frag_id}: {smiles}")

            if frag_data["interactions"]:
                lines.append("\nInteraction Table:")
                # Header
                header = f"{'Residue-Fragment':20s} {'vdW':>8s} {'H-bond':>8s} {'Metal':>8s} {'Hydro':>8s} {'Ionic':>8s} {'Total':>8s}"
                lines.append(header)
                lines.append("-" * len(header))

                # Sort interactions by residue and fragment
                sorted_interactions = sorted(
                    frag_data["interactions"],
                    key=lambda x: (x["residue"], x["fragment"]),
                )

                for inter in sorted_interactions:
                    res_name = inter.get("residue_name", f"Res{inter['residue']}")
                    res_frag = f"{res_name}-Frag{inter['fragment']}"
                    breakdown = inter["energy_breakdown"]

                    # Get values for each energy type
                    vdw = breakdown.get("vdW", 0.0)
                    hbond = breakdown.get("H-bond", 0.0)
                    metal = breakdown.get("Metal-ligand", 0.0)
                    hydro = breakdown.get("Hydrophobic", 0.0)
                    ionic = breakdown.get("Ionic", 0.0)
                    total = inter["total_energy"]

                    line = f"{res_frag:20s} {vdw:8.3f} {hbond:8.3f} {metal:8.3f} {hydro:8.3f} {ionic:8.3f} {total:8.3f}"
                    lines.append(line)

    # Atom-atom interactions
    if "atom_interactions" in data:
        lines.append("\n" + "-" * 40)
        lines.append("ATOM-ATOM INTERACTIONS")
        lines.append("-" * 40)

        atom_data = data["atom_interactions"]
        if "note" in atom_data:
            lines.append(f"  {atom_data['note']}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def format_pretty_xml(data: dict[str, Any], root_name: str = "root") -> str:
    """Convert dictionary to pretty-printed XML string."""
    root = ET.Element(root_name)
    _dict_to_xml_elem(data, root)

    # Pretty print the XML
    import xml.dom.minidom

    rough_string = ET.tostring(root, encoding="unicode")
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def _dict_to_xml_elem(data: Any, parent: ET.Element):
    """Recursively convert dictionary to XML elements."""
    if isinstance(data, dict):
        for key, value in data.items():
            # Make XML-safe key names
            safe_key = key.replace(" ", "_").replace("-", "_")

            if isinstance(value, list):
                list_elem = ET.SubElement(parent, safe_key)
                for item in value:
                    item_elem = ET.SubElement(list_elem, "item")
                    _dict_to_xml_elem(item, item_elem)
            else:
                elem = ET.SubElement(parent, safe_key)
                _dict_to_xml_elem(value, elem)
    else:
        parent.text = str(data)
