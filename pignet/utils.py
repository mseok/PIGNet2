"""Utility functions for PIGNet."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

from .core import PIGNetCore

logger = logging.getLogger(__name__)


def quick_predict(
    protein_pdb: str | Path,
    ligand_sdf: str | Path,
    device: str = "cuda",
    verbose: bool = False,
) -> float:
    """Quick prediction using ensemble of all 4 pre-trained models.

    Args:
        protein_pdb: Path to protein PDB file
        ligand_sdf: Path to ligand SDF file
        device: Device to use ('cuda' or 'cpu')
        verbose: Whether to print progress

    Returns:
        Predicted binding affinity (kcal/mol)
    """
    # Use all 4 pre-trained models
    model_dir = Path(__file__).parent / "ckpt"
    model_paths = [
        model_dir / "pda_0.pt",
        model_dir / "pda_1.pt",
        model_dir / "pda_2.pt",
        model_dir / "pda_3.pt",
    ]

    # Check if models exist
    missing_models = [p for p in model_paths if not p.exists()]
    if missing_models:
        raise FileNotFoundError(f"Missing model files: {missing_models}")

    if verbose:
        logger.info("Loading ensemble of 4 pre-trained models")

    # Initialize ensemble model
    core = PIGNetCore(
        model_paths=[str(p) for p in model_paths],
        device=device,
        ensemble=True,
    )

    # Run prediction
    result = core.predict_complex(protein_pdb, ligand_sdf, write_output=False)

    return result["affinity"]


def batch_predict_from_dataframe(
    df: pd.DataFrame,
    model_path: str | Path,
    protein_col: str = "protein_path",
    ligand_col: str = "ligand_path",
    name_col: str | None = None,
    device: str = "cuda",
    track_residues: bool = False,
) -> pd.DataFrame:
    """Batch predict binding affinities from a DataFrame.

    Args:
        df: DataFrame containing protein and ligand paths
        model_path: Path to model checkpoint
        protein_col: Column name containing protein paths
        ligand_col: Column name containing ligand paths
        name_col: Optional column name for complex names
        device: Device to use ('cuda' or 'cpu')
        track_residues: Whether to track residue information

    Returns:
        DataFrame with added prediction columns
    """
    # Initialize model
    core = PIGNetCore(
        model_paths=model_path,
        device=device,
        ensemble=False,
    )

    # Prepare pairs
    pairs = list(zip(df[protein_col], df[ligand_col]))

    # Get names if available
    if name_col and name_col in df.columns:
        names = df[name_col].tolist()
    else:
        names = None

    # Run predictions
    logger.info(f"Running batch predictions for {len(pairs)} complexes")
    results = core.predict_batch(
        pairs, complex_names=names, track_residues=track_residues
    )

    # Add results to dataframe
    df = df.copy()
    df["affinity"] = [r.get("affinity", np.nan) for r in results]
    df["affinity_unit"] = [r.get("affinity_unit", "") for r in results]
    df["error"] = [r.get("error", "") for r in results]

    if track_residues:
        df["pocket_residues"] = [
            r.get("residue_info", {}).get("pocket_residues", []) for r in results
        ]
        df["ligand_fragments"] = [
            r.get("residue_info", {}).get("ligand_fragments", []) for r in results
        ]

    return df


def screen_compound_library(
    protein_pdb: str | Path,
    ligand_library: str | Path | list[str],
    model_path: str | Path,
    top_k: int = 10,
    device: str = "cuda",
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Screen a compound library against a protein target.

    Args:
        protein_pdb: Path to protein PDB file
        ligand_library: Path to SDF file with multiple compounds or list of paths
        model_path: Path to model checkpoint
        top_k: Number of top compounds to return
        device: Device to use
        output_csv: Optional path to save results

    Returns:
        DataFrame with top compounds sorted by affinity
    """
    # Initialize model
    core = PIGNetCore(
        model_paths=model_path,
        device=device,
        ensemble=False,
    )

    # Prepare ligand list
    if isinstance(ligand_library, (str, Path)):
        ligand_library = Path(ligand_library)
        if ligand_library.suffix == ".sdf":
            # Read multi-mol SDF
            supplier = Chem.SDMolSupplier(str(ligand_library))
            ligand_files = []
            ligand_names = []

            # Save individual molecules temporarily
            temp_dir = Path("/tmp/pignet_screening")
            temp_dir.mkdir(exist_ok=True)

            for i, mol in enumerate(supplier):
                if mol is not None:
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
                    temp_file = temp_dir / f"{name}.sdf"
                    writer = Chem.SDWriter(str(temp_file))
                    writer.write(mol)
                    writer.close()
                    ligand_files.append(str(temp_file))
                    ligand_names.append(name)
        else:
            # Single file
            ligand_files = [str(ligand_library)]
            ligand_names = [ligand_library.stem]
    else:
        # List of files
        ligand_files = [str(f) for f in ligand_library]
        ligand_names = [Path(f).stem for f in ligand_library]

    # Create pairs
    pairs = [(protein_pdb, lig) for lig in ligand_files]

    # Run screening
    logger.info(f"Screening {len(pairs)} compounds")
    results = core.predict_batch(pairs, complex_names=ligand_names)

    # Create results dataframe
    df = pd.DataFrame(
        {
            "compound": ligand_names,
            "ligand_path": ligand_files,
            "affinity": [r.get("affinity", np.nan) for r in results],
            "error": [r.get("error", "") for r in results],
        }
    )

    # Filter out errors and sort by affinity
    df_valid = df[df["error"] == ""].copy()
    df_valid = df_valid.sort_values("affinity", ascending=False).head(top_k)

    # Clean up temp files if created
    if "temp_dir" in locals():
        import shutil

        shutil.rmtree(temp_dir)

    # Save if requested
    if output_csv:
        df_valid.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")

    return df_valid


def calculate_enrichment_factor(
    predictions: list[float],
    labels: list[bool],
    top_fraction: float = 0.1,
) -> float:
    """Calculate enrichment factor for virtual screening.

    Args:
        predictions: List of predicted affinities
        labels: List of boolean labels (True for active compounds)
        top_fraction: Fraction of compounds to consider as "top"

    Returns:
        Enrichment factor
    """
    # Sort by predictions (higher is better for kcal/mol)
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_labels = np.array(labels)[sorted_indices]

    # Calculate enrichment
    n_top = int(len(predictions) * top_fraction)
    n_actives_total = sum(labels)
    n_actives_top = sum(sorted_labels[:n_top])

    if n_actives_total == 0:
        return 0.0

    expected_actives = n_top * (n_actives_total / len(predictions))
    if expected_actives == 0:
        return 0.0

    enrichment_factor = n_actives_top / expected_actives

    return enrichment_factor


def compare_models(
    test_data: str | pd.DataFrame,
    model_paths: list[str | Path],
    protein_col: str = "protein_path",
    ligand_col: str = "ligand_path",
    label_col: str | None = "affinity",
    device: str = "cuda",
) -> pd.DataFrame:
    """Compare multiple models on the same test set.

    Args:
        test_data: Path to CSV or DataFrame with test data
        model_paths: List of model checkpoint paths
        protein_col: Column name for protein paths
        ligand_col: Column name for ligand paths
        label_col: Optional column name for true affinities
        device: Device to use

    Returns:
        DataFrame with comparison results
    """
    # Load test data
    if isinstance(test_data, str):
        df = pd.read_csv(test_data)
    else:
        df = test_data.copy()

    # Prepare pairs
    pairs = list(zip(df[protein_col], df[ligand_col]))

    results = {}

    # Test each model
    for model_path in model_paths:
        model_name = Path(model_path).stem
        logger.info(f"Testing model: {model_name}")

        # Initialize model
        core = PIGNetCore(
            model_paths=model_path,
            device=device,
            ensemble=False,
        )

        # Run predictions
        predictions = core.predict_batch(pairs)

        # Extract affinities
        affinities = [p.get("affinity", np.nan) for p in predictions]
        results[model_name] = affinities

        # Calculate metrics if labels available
        if label_col and label_col in df.columns:
            from scipy import stats
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            labels = df[label_col].values
            valid_idx = ~(np.isnan(affinities) | np.isnan(labels))

            if valid_idx.sum() > 0:
                valid_preds = np.array(affinities)[valid_idx]
                valid_labels = labels[valid_idx]

                pearson_r, _ = stats.pearsonr(valid_preds, valid_labels)
                spearman_r, _ = stats.spearmanr(valid_preds, valid_labels)
                mae = mean_absolute_error(valid_labels, valid_preds)
                rmse = np.sqrt(mean_squared_error(valid_labels, valid_preds))

                logger.info(f"  Pearson R: {pearson_r:.3f}")
                logger.info(f"  Spearman R: {spearman_r:.3f}")
                logger.info(f"  MAE: {mae:.3f}")
                logger.info(f"  RMSE: {rmse:.3f}")

    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    comparison_df["protein"] = df[protein_col]
    comparison_df["ligand"] = df[ligand_col]

    if label_col and label_col in df.columns:
        comparison_df["true_affinity"] = df[label_col]

    return comparison_df


def analyze_binding_pocket(
    protein_pdb: str | Path,
    ligand_sdf: str | Path,
    model_path: str | Path,
    distance_cutoff: float = 5.0,
    device: str = "cuda",
) -> dict:
    """Analyze the binding pocket and interactions.

    Args:
        protein_pdb: Path to protein PDB file
        ligand_sdf: Path to ligand SDF file
        model_path: Path to model checkpoint
        distance_cutoff: Distance cutoff for pocket definition
        device: Device to use

    Returns:
        Dictionary with pocket analysis results
    """
    from .analysis import extract_pocket_residues, analyze_interactions
    from .exe.generate_data import extract_binding_pocket, read_mols
    from .exe.protonate import protonate_ligand, protonate_pdb

    # Initialize model
    core = PIGNetCore(
        model_paths=model_path,
        device=device,
        ensemble=False,
    )

    # Run prediction with residue tracking
    result = core.predict_complex(
        protein_pdb,
        ligand_sdf,
        track_residues=True,
        detailed_interactions=True,
    )

    # Load molecules for detailed analysis
    ligand_mol = read_mols(ligand_sdf)[0]
    ligand_mol = protonate_ligand(ligand_mol)
    protein_protonated = protonate_pdb(protein_pdb)
    protein_mol = extract_binding_pocket(ligand_mol, protein_protonated)

    # Analyze interactions
    interactions = analyze_interactions(
        protein_mol,
        ligand_mol,
        distance_cutoff=distance_cutoff,
    )

    # Extract pocket residues
    pocket_residues = extract_pocket_residues(
        protein_mol,
        ligand_mol,
        distance_cutoff=distance_cutoff,
    )

    # Clean up temp files
    if isinstance(protein_protonated, Path):
        protein_protonated.unlink()

    # Compile analysis results
    analysis = {
        "affinity": result["affinity"],
        "pocket_residues": pocket_residues,
        "num_pocket_residues": len(pocket_residues),
        "interactions": {
            "hydrogen_bonds": len(interactions["hydrogen_bonds"]),
            "hydrophobic": len(interactions["hydrophobic"]),
            "pi_stacking": len(interactions["pi_stacking"]),
            "salt_bridges": len(interactions["salt_bridges"]),
        },
        "interaction_details": interactions,
    }

    if "residue_info" in result:
        analysis["ligand_fragments"] = result["residue_info"]["ligand_fragments"]

    return analysis
