"""Command-line interface for PIGNet using Click."""

import json
import logging
import subprocess
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from .core import PIGNetCore
from .analysis import generate_detailed_explanation

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="2.0.0", prog_name="pignet")
def cli():
    """PIGNet: Deep learning for protein-ligand binding affinity prediction.

    A versatile tool for predicting protein-ligand interactions using
    graph neural networks with physics-based features.
    """
    pass


# ==============================================================================
# Prediction Commands
# ==============================================================================


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "-p",
    "--protein",
    "protein_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Protein PDB file(s)",
)
@click.option(
    "-l",
    "--ligand",
    "ligand_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Ligand file(s) (SDF/MOL2/PDB)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="predictions.txt",
    help="Output file path",
)
@click.option("--no-protonate-ligand", is_flag=True, help="Don't protonate ligands")
@click.option("--no-protonate-protein", is_flag=True, help="Don't protonate proteins")
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Device to use for computation",
)
@click.option("--json-output", is_flag=True, help="Output results as JSON")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def predict(
    checkpoint,
    protein_files,
    ligand_files,
    output,
    no_protonate_ligand,
    no_protonate_protein,
    device,
    json_output,
    verbose,
):
    """Predict binding affinity for protein-ligand complex(es).

    Example:
        pignet predict model.pt -p protein.pdb -l ligand.sdf
        pignet predict model.pt -p *.pdb -l *.sdf -o results.json --json-output
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize core model
    logger.info(f"Loading model from {checkpoint}")
    core = PIGNetCore(
        model_paths=checkpoint,
        device=device,
        ensemble=False,
        output_path=output if not json_output else None,
        protonate_ligand=not no_protonate_ligand,
        protonate_protein=not no_protonate_protein,
    )

    # Prepare protein-ligand pairs
    if len(protein_files) == 1 and len(ligand_files) > 1:
        # One protein, multiple ligands
        pairs = [(protein_files[0], lig) for lig in ligand_files]
    elif len(ligand_files) == 1 and len(protein_files) > 1:
        # Multiple proteins, one ligand
        pairs = [(prot, ligand_files[0]) for prot in protein_files]
    elif len(protein_files) == len(ligand_files):
        # Paired proteins and ligands
        pairs = list(zip(protein_files, ligand_files))
    else:
        raise click.BadParameter(
            "Number of proteins and ligands must match, or one must be singular"
        )

    # Run predictions
    logger.info(f"Running predictions for {len(pairs)} complex(es)")
    results = core.predict_batch(pairs, track_residues=True)

    # Output results
    if json_output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output}")
    else:
        # Text output
        if output:
            with open(output, "w") as f:
                for result in results:
                    if "error" in result:
                        f.write(
                            f"{result['complex_name']}: ERROR - {result['error']}\n"
                        )
                    else:
                        f.write(
                            f"{result['complex_name']}: {result['affinity']:.3f} {result['affinity_unit']}\n"
                        )
            logger.info(f"Results saved to {output}")

        # Also print to console
        for result in results:
            if "error" in result:
                click.echo(f"{result['complex_name']}: ERROR - {result['error']}")
            else:
                click.echo(
                    f"{result['complex_name']}: {result['affinity']:.3f} {result['affinity_unit']}"
                )


@cli.command()
@click.option(
    "-p",
    "--protein",
    required=True,
    type=click.Path(exists=True),
    help="Protein PDB file",
)
@click.option(
    "-l",
    "--ligand",
    required=True,
    type=click.Path(exists=True),
    help="Ligand file (SDF/MOL2/PDB)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Device to use",
)
def quick(protein, ligand, output, device):
    """Quick prediction using ensemble of all 4 pre-trained models.

    Example:
        pignet quick -p protein.pdb -l ligand.sdf
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
        click.echo(f"Error: Missing model files: {missing_models}", err=True)
        click.echo("Please ensure all 4 pre-trained models are in pignet2/ckpt", err=True)
        sys.exit(1)

    logger.info("Loading ensemble of 4 pre-trained models")
    core = PIGNetCore(
        model_paths=[str(p) for p in model_paths],
        device=device,
        ensemble=True,
        output_path=output,
    )

    # Run prediction
    result = core.predict_complex(protein, ligand, track_residues=True)

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("PIGNet Quick Prediction (Ensemble)")
    click.echo("=" * 60)
    click.echo(f"Protein: {protein}")
    click.echo(f"Ligand: {ligand}")
    click.echo(f"Device: {device}")
    click.echo("-" * 40)
    click.echo(f"Binding Affinity: {result['affinity']:.3f} {result['affinity_unit']}")
    click.echo(f"Ensemble Size: {result['ensemble_size']} models")
    click.echo(f"Conformations: {result['conformations']}")

    if "residue_info" in result:
        residue_info = result["residue_info"]
        click.echo("-" * 40)
        click.echo(f"Pocket Residues: {len(residue_info['pocket_residues'])}")
        click.echo(f"Ligand Fragments: {len(residue_info['ligand_fragments'])}")

    click.echo("=" * 60)

    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {output}")


@cli.command()
@click.option(
    "-c",
    "--checkpoint",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Path to model checkpoint",
)
@click.option(
    "-p",
    "--protein",
    required=True,
    type=click.Path(exists=True),
    help="Protein PDB file",
)
@click.option(
    "-l",
    "--ligand",
    required=True,
    type=click.Path(exists=True),
    help="Ligand file",
)
@click.option(
    "--energy-breakdown",
    is_flag=True,
    help="Include energy component breakdown",
)
@click.option(
    "--fragment-analysis",
    is_flag=True,
    help="Include fragment-residue analysis",
)
@click.option(
    "--atom-analysis",
    is_flag=True,
    help="Include atom-level analysis",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json", "xml"]),
    default="text",
    help="Output format",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Save output to file",
)
def explain(
    checkpoint,
    protein,
    ligand,
    energy_breakdown,
    fragment_analysis,
    atom_analysis,
    format,
    output,
):
    """Generate detailed explanation of prediction.

    Example:
        pignet explain -p protein.pdb -l ligand.sdf --energy-breakdown
        pignet explain -c model.pt -p protein.pdb -l ligand.sdf --energy-breakdown
        pignet explain -c model.pt -p protein.pdb -l ligand.sdf --fragment-analysis --format json
    """
    # Use ensemble of all 4 models by default
    model_dir = Path(__file__).parent / "ckpt"
    model_paths = [
        model_dir / "pda_0.pt",
        model_dir / "pda_1.pt",
        model_dir / "pda_2.pt",
        model_dir / "pda_3.pt",
    ]

    # Check if all models exist, otherwise use provided checkpoint
    missing_models = [p for p in model_paths if not p.exists()]
    if missing_models or checkpoint:
        logger.info(f"Using single model: {checkpoint}")
        model_paths = checkpoint
        use_ensemble = False
    else:
        logger.info("Using ensemble of 4 pre-trained models")
        use_ensemble = True

    # Initialize core model
    core = PIGNetCore(
        model_paths=model_paths,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ensemble=use_ensemble,
    )

    # Generate explanation
    explanation = generate_detailed_explanation(
        core,
        protein,
        ligand,
        energy_breakdown=energy_breakdown,
        fragment_analysis=fragment_analysis,
        atom_analysis=atom_analysis,
        format=format,
    )

    # Output results
    if output:
        with open(output, "w") as f:
            f.write(explanation)
        logger.info(f"Explanation saved to {output}")
    else:
        click.echo(output_str)


# ==============================================================================
# Training Commands
# ==============================================================================


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to training configuration file",
)
@click.option(
    "--config-name",
    default="baseline",
    help="Name of configuration to use from config directory",
)
@click.option("--epochs", type=int, help="Number of training epochs")
@click.option("--batch-size", type=int, help="Batch size")
@click.option("--lr", type=float, help="Learning rate")
@click.option("--device", type=click.Choice(["cuda", "cpu"]), help="Device to use")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def train(config, config_name, epochs, batch_size, lr, device, verbose):
    """Train a PIGNet model.

    Example:
        pignet train --config-name baseline --epochs 300
        pignet train --config my_config.yaml --epochs 100 --lr 0.001
    """
    # Import training module
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "exe"))

    # Construct command
    cmd = ["python", str(Path(__file__).parent / "exe" / "train.py")]

    if config:
        cmd.extend(["--config-path", str(Path(config).parent)])
        cmd.extend(["--config-name", Path(config).stem])
    else:
        cmd.extend(["--config-name", config_name])

    # Add overrides
    overrides = []
    if epochs is not None:
        overrides.append(f"run.n_epochs={epochs}")
    if batch_size is not None:
        overrides.append(f"run.batch_size={batch_size}")
    if lr is not None:
        overrides.append(f"run.lr={lr}")
    if device is not None:
        overrides.append(f"run.device={device}")

    if overrides:
        cmd.extend(overrides)

    # Run training
    logger.info(f"Starting training with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to test configuration file",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    help="Directory containing test data",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file for test results",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def test(checkpoint, config, data_dir, output, verbose):
    """Test a trained PIGNet model.

    Example:
        pignet test model.pt --data-dir test_data/
        pignet test model.pt --config test_config.yaml -o results.txt
    """
    # Import test module
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "exe"))

    # Construct command
    cmd = ["python", str(Path(__file__).parent / "exe" / "test.py"), checkpoint]

    if config:
        cmd.extend(["--config-path", str(Path(config).parent)])
        cmd.extend(["--config-name", Path(config).stem])

    if data_dir:
        cmd.extend(["data.test.data_dir=" + str(data_dir)])

    if output:
        cmd.extend([f"data.test.test_result_path={output}"])

    # Run testing
    logger.info(f"Starting testing with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ==============================================================================
# Utility Commands
# ==============================================================================


@cli.command()
@click.argument("csv_file", type=click.Path(exists=True))
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--protein-col",
    default="protein_path",
    help="Column name for protein paths",
)
@click.option(
    "--ligand-col",
    default="ligand_path",
    help="Column name for ligand paths",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output CSV file",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Device to use",
)
def batch(csv_file, checkpoint, protein_col, ligand_col, output, device):
    """Process batch predictions from CSV file.

    The CSV file should contain columns with paths to protein and ligand files.

    Example:
        pignet batch compounds.csv model.pt -o results.csv
        pignet batch data.csv model.pt --protein-col prot --ligand-col lig
    """
    # Read CSV
    df = pd.read_csv(csv_file)

    if protein_col not in df.columns or ligand_col not in df.columns:
        click.echo(
            f"Error: CSV must contain columns '{protein_col}' and '{ligand_col}'",
            err=True,
        )
        sys.exit(1)

    # Initialize model
    core = PIGNetCore(
        model_paths=checkpoint,
        device=device,
        ensemble=False,
    )

    # Prepare pairs
    pairs = list(zip(df[protein_col], df[ligand_col]))

    # Run predictions
    logger.info(f"Running batch predictions for {len(pairs)} complexes")
    results = core.predict_batch(pairs)

    # Add results to dataframe
    df["affinity"] = [r.get("affinity", np.nan) for r in results]
    df["affinity_unit"] = [r.get("affinity_unit", "") for r in results]
    df["error"] = [r.get("error", "") for r in results]

    # Save results
    if output:
        df.to_csv(output, index=False)
        logger.info(f"Results saved to {output}")
    else:
        click.echo(df.to_string())


@cli.command()
@click.pass_context
def help(ctx):
    """Show help information."""
    click.echo(ctx.parent.get_help())


if __name__ == "__main__":
    cli()
