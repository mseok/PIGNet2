"""Core PIGNet model wrapper with direct model loading for optimal performance."""

import logging
import sys
from pathlib import Path

import torch
import torch_geometric.data as pyg_data
from omegaconf import open_dict
from rdkit import Chem

from pignet2.data.data import complex_to_data
from pignet2.exe.generate_data import extract_binding_pocket, read_mols
from pignet2.exe.protonate import protonate_ligand, protonate_pdb
from pignet2.exe import utils

logger = logging.getLogger(__name__)


class PIGNetCore:
    """Direct model interface for PIGNet without subprocess overhead."""

    def __init__(
        self,
        model_paths: str | list[str],
        device: str = "cuda",
        ensemble: bool = True,
        output_path: str | None = None,
        protonate_ligand: bool = True,
        protonate_protein: bool = True,
    ):
        """Initialize PIGNet with direct model loading.

        Args:
            model_paths: Path(s) to model checkpoint(s)
            device: Device to run model on ('cuda' or 'cpu')
            ensemble: Whether to use ensemble predictions
            output_path: Path to save predictions (optional)
            protonate_ligand: Whether to protonate ligands
            protonate_protein: Whether to protonate proteins
        """
        self.model_paths = (
            model_paths if isinstance(model_paths, list) else [model_paths]
        )
        self.device_name = device
        self.ensemble = ensemble
        self.output_path = output_path
        self.protonate_ligand_flag = protonate_ligand
        self.protonate_protein_flag = protonate_protein

        # Set up device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            if device == "cuda":
                logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")

        # Load models
        self.models = []
        self.configs = []
        self._load_models()

    def _load_models(self):
        """Load all models into memory."""
        for model_path in self.model_paths:
            logger.info(f"Loading model from {model_path}")

            # Load checkpoint
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as _:
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )
            config = utils.merge_configs(checkpoint.get("config", {}), {})

            # Update old model targets to new package structure
            with open_dict(config):
                if "model" in config and "_target_" in config.model:
                    old_target = config.model._target_
                    if old_target.startswith("models."):
                        config.model._target_ = old_target.replace(
                            "models.", "pignet2.models."
                        )

            # Ensure data config exists for prediction
            with open_dict(config):
                if "data" not in config:
                    config.data = {}

                # Add a default test task if no tasks exist
                if not config.data:
                    config.data["test"] = {
                        "test_result_path": (
                            str(self.output_path)
                            if self.output_path
                            else "predictions.txt"
                        )
                    }

                # Set output path in config
                if self.output_path:
                    # Update the first task's output path
                    task_name = next(iter(config.data))
                    config.data[task_name]["test_result_path"] = str(self.output_path)

            # Initialize model
            model = utils.initialize_state(self.device, checkpoint, config)[0]
            model.eval()

            self.models.append(model)
            self.configs.append(config)

        logger.info(f"Loaded {len(self.models)} models")

    def _prepare_data(
        self,
        protein_path: str | Path,
        ligand_path: str | Path,
        conv_range: tuple[float, float] = (0.0, 10.0),
        track_residues: bool = False,
    ) -> list[tuple[pyg_data.Data | None, str, dict | None]]:
        """Prepare protein-ligand complex data.

        Args:
            protein_path: Path to protein PDB file
            ligand_path: Path to ligand file (SDF/MOL2/PDB)
            conv_range: Convolution range for the model
            track_residues: Whether to track residue information

        Returns:
            List of (data, name, residue_info) tuples
        """
        protein_path = Path(protein_path)
        ligand_path = Path(ligand_path)
        name_root = f"{protein_path.stem}_{ligand_path.stem}"

        # Read ligand molecules
        mol_ligands = read_mols(ligand_path)
        if self.protonate_ligand_flag:
            mol_ligands = [protonate_ligand(lig) for lig in mol_ligands]

        # Protonate protein if needed
        if self.protonate_protein_flag:
            protein_protonated = protonate_pdb(protein_path)
        else:
            protein_protonated = protein_path

        # Process each ligand conformation
        data_list = []
        for idx, ligand in enumerate(mol_ligands):
            # Extract binding pocket
            mol_target = extract_binding_pocket(ligand, protein_protonated)
            name = f"{name_root}_{idx}"

            # Convert to data
            if ligand is None or mol_target is None:
                datum = None
                residue_info = None
            else:
                # Enhanced data conversion with residue tracking
                datum = complex_to_data(
                    ligand,
                    mol_target,
                    key=name,
                    conv_range=conv_range,
                )

                # Extract residue information if requested
                residue_info = None
                if track_residues and datum is not None:
                    residue_info = self._extract_residue_info(mol_target, ligand)

            if datum is not None:
                data_list.append((datum, name, residue_info))

        # Clean up protonated protein file
        if self.protonate_protein_flag and isinstance(protein_protonated, Path):
            protein_protonated.unlink()

        return data_list

    def _extract_residue_info(
        self, protein_mol: Chem.Mol, ligand_mol: Chem.Mol
    ) -> dict:
        """Extract residue-level information from protein-ligand complex.

        Args:
            protein_mol: RDKit Mol object for protein pocket
            ligand_mol: RDKit Mol object for ligand

        Returns:
            Dictionary containing residue information
        """
        residue_info = {
            "pocket_residues": [],
            "ligand_fragments": [],
            "interactions": [],
        }

        # Extract pocket residues
        for atom in protein_mol.GetAtoms():
            res_info = atom.GetPDBResidueInfo()
            if res_info:
                residue_id = f"{res_info.GetResidueName()}{res_info.GetResidueNumber()}{res_info.GetChainId()}"
                if residue_id not in residue_info["pocket_residues"]:
                    residue_info["pocket_residues"].append(residue_id)

        # Fragment ligand (simplified version)
        # In a full implementation, this would use more sophisticated fragmentation
        try:
            fragments = Chem.GetMolFrags(ligand_mol, asMols=True, sanitizeFrags=False)
            residue_info["ligand_fragments"] = [
                Chem.MolToSmiles(frag) for frag in fragments
            ]
        except:
            residue_info["ligand_fragments"] = [Chem.MolToSmiles(ligand_mol)]

        return residue_info

    def predict_complex(
        self,
        protein_pdb: str | Path,
        ligand_sdf: str | Path,
        complex_name: str | None = None,
        write_output: bool = True,
        detailed_interactions: bool = False,
        track_residues: bool = False,
    ) -> dict[str, float | list[float] | str]:
        """Predict binding affinity for a protein-ligand complex.

        Args:
            protein_pdb: Path to protein PDB file
            ligand_sdf: Path to ligand SDF file
            complex_name: Optional name for the complex
            write_output: Whether to write results to output file
            detailed_interactions: Whether to include detailed interaction analysis
            track_residues: Whether to track residue information

        Returns:
            Dictionary with prediction results
        """
        # Generate complex name if not provided
        if complex_name is None:
            protein_name = Path(protein_pdb).stem
            ligand_name = Path(ligand_sdf).stem
            complex_name = f"{protein_name}_{ligand_name}"

        # Use first model's config for data preparation
        conv_range = self.configs[0].model.get("conv_range", (0.0, 10.0))

        # Prepare data
        data_list = self._prepare_data(
            protein_pdb, ligand_sdf, conv_range, track_residues=track_residues
        )

        if not data_list:
            raise ValueError(f"Failed to process complex {complex_name}")

        # Get predictions from all models
        all_predictions = []
        all_interactions = [] if detailed_interactions else None
        all_residue_info = []

        for model, config in zip(self.models, self.configs):
            model.reset_log()

            # Process each conformation
            for data, name, residue_info in data_list:
                batch = pyg_data.Batch.from_data_list([data])
                batch = batch.to(self.device)

                with torch.no_grad():
                    model.predict_step(batch)

                    # Get detailed interactions if requested
                    if detailed_interactions:
                        try:
                            # Get energy breakdown
                            energy_components = self._get_energy_breakdown(model, batch)

                            if all_interactions is not None:
                                all_interactions.append(
                                    {
                                        "energy_components": energy_components,
                                        "model_idx": len(all_predictions),
                                    }
                                )
                        except Exception as e:
                            logger.warning(f"Failed to get detailed interactions: {e}")

                # Store residue info
                if residue_info and track_residues:
                    all_residue_info.append(residue_info)

            # Write predictions if requested
            if write_output and self.output_path:
                utils.write_predictions(model, config, False)

            # Extract predictions from model predictions dict
            predictions = []
            task = next(iter(config.data))  # Get the task name

            # Get predictions for all keys (conformations)
            for datum, name, _ in data_list:
                if task in model.predictions and name in model.predictions[task]:
                    # Get energy components (vdw, hbond, ml, hydro)
                    energy_components = model.predictions[task][name]
                    total_energy = sum(energy_components)
                    predictions.append(total_energy)

            all_predictions.append(predictions)

        # Aggregate predictions
        if self.ensemble and len(self.models) > 1:
            # Average across models for each conformation
            ensemble_predictions = []
            for conf_idx in range(len(data_list)):
                conf_preds = [
                    preds[conf_idx]
                    for preds in all_predictions
                    if conf_idx < len(preds)
                ]
                if conf_preds:
                    ensemble_predictions.append(sum(conf_preds) / len(conf_preds))

            # Return mean of all conformations
            mean_affinity = (
                sum(ensemble_predictions) / len(ensemble_predictions)
                if ensemble_predictions
                else 0.0
            )

            result = {
                "complex_name": complex_name,
                "affinity": mean_affinity,
                "affinity_unit": "kcal/mol",  # PIGNet predicts kcal/mol values
                "conformations": len(data_list),
                "ensemble_size": len(self.models),
                "all_predictions": ensemble_predictions,
                "device": str(self.device),
            }

            # Add interaction data if available
            if detailed_interactions and all_interactions:
                result["interactions"] = all_interactions

            # Add residue information if available
            if track_residues and all_residue_info:
                result["residue_info"] = all_residue_info[0]  # Use first conformation
        else:
            # Single model predictions
            predictions = all_predictions[0] if all_predictions else []
            mean_affinity = sum(predictions) / len(predictions) if predictions else 0.0

            result = {
                "complex_name": complex_name,
                "affinity": mean_affinity,
                "affinity_unit": "kcal/mol",
                "conformations": len(data_list),
                "predictions_per_conf": predictions,
                "device": str(self.device),
            }

            # Add interaction data if available
            if detailed_interactions and all_interactions:
                result["interactions"] = all_interactions

            # Add residue information if available
            if track_residues and all_residue_info:
                result["residue_info"] = all_residue_info[0]

        return result

    def _get_energy_breakdown(self, model, batch) -> dict[str, float]:
        """Extract energy component breakdown from model.

        Args:
            model: PIGNet model
            batch: Data batch

        Returns:
            Dictionary with energy components
        """
        # TODO: This is a simplified version - would need model modifications for full implementation
        energy_breakdown = {
            "vdw": 0.0,
            "hbond": 0.0,
            "metal_ligand": 0.0,
            "hydrophobic": 0.0,
            "total": 0.0,
        }

        # In a full implementation, this would extract the actual energy components
        # from the model's internal calculations

        return energy_breakdown

    def predict_batch(
        self,
        protein_ligand_pairs: list[tuple[str | Path, str | Path]],
        complex_names: list[str] | None = None,
        write_output: bool = True,
        track_residues: bool = False,
    ) -> list[dict[str, float | list[float] | str]]:
        """Predict binding affinities for multiple complexes.

        Args:
            protein_ligand_pairs: List of (protein_pdb, ligand_sdf) tuples
            complex_names: Optional list of names for complexes
            write_output: Whether to write results to output file
            track_residues: Whether to track residue information

        Returns:
            List of prediction dictionaries
        """
        results = []

        if complex_names is None:
            complex_names = [None] * len(protein_ligand_pairs)

        for (protein, ligand), name in zip(protein_ligand_pairs, complex_names):
            try:
                result = self.predict_complex(
                    protein,
                    ligand,
                    name,
                    write_output=write_output,
                    track_residues=track_residues,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {name or 'complex'}: {e}")
                results.append(
                    {
                        "complex_name": name
                        or f"{Path(protein).stem}_{Path(ligand).stem}",
                        "error": str(e),
                        "affinity": None,
                    }
                )

        return results

    def __del__(self):
        """Clean up models from memory."""
        if hasattr(self, "models"):
            del self.models
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
