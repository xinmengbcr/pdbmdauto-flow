import json  # noqa: E402
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import gromacs
import numpy as np
import pandas as pd

from bocoflow_core.logger import log_message
from bocoflow_core.node import Node, NodeException, NodeResult
from bocoflow_core.parameters import *

# use gromacs wrapper
# conda install -c conda-forge gromacswrapper
# 0.9.1


class GmxSolvIon(Node):
    """GmxSolvIon
    This Node prepares the system for GROMACS MD simulation by adding solvent (water)
    and ions. The process includes:

    1. Setting up the simulation box with specified dimensions
    2. Adding water molecules using the SPC216 water model
    3. Adding ions (Na+ and Cl-) to neutralize the system and achieve desired concentration
    4. Generating updated index groups for the solvated system

    [ Parameters ]
    - case_name: Name identifier for the simulation case (optional)
    - run_label: Label for the MD simulation (e.g., md, nvt, md_nvt)
    - box_size_string: Box dimensions in nm (e.g., "20 20 20", set "0 0 0" to ignore)
    - ion_conc: Ion concentration in mol/L (set 0 to skip ion addition)
    - scale_fill: Scale factor for Van der Waals radii (default: 0.57)
    - force_to_run: Force execution regardless of database record

    [ Input Files ]
    - TOP file (.top): Topology file containing molecular structure and force field
    - GRO file (.gro): Structure file in GROMACS format
    - MDP file (.mdp): Parameters file for molecular dynamics
    - NDX file (.ndx): Index file defining atom groups

    [ Output Files ]
    - Solvated structure file (*_box.gro): System with defined box
    - Solvated structure file (*_solv.gro): System with added water
    - Final structure file (*_ion.gro): System with water and ions
    - Updated topology file (*_solvion.top)
    - Updated index file (.ndx)

    [ Dependencies ]
    Requires GROMACS and gromacswrapper (v0.9.1+) to be installed:
    conda install -c conda-forge gromacswrapper

    [ Note ]
    The scale_fill parameter controls water density, with the default value of 0.57
    yielding density close to 1000 g/l. Ion concentration can be adjusted through the
    ion_conc parameter (mol/L).
    """

    # Basic configuration
    name = "GmxSolvIon"
    node_key = "GmxSolvIon"
    color = "green"
    num_in = 1
    num_out = 1

    # Define the parameters for GUI
    OPTIONS = {
        "case_name": StringParameter(
            label="Case Name",
            default="",
            docstring="Name of the case/protein (leave empty to use predecessor data)",
            optional=True,
        ),
        "run_label": StringParameter(
            label="give the md simulation label",
            default="md",
            docstring="label the md run, e.g. md, nvt, md_nvt, md_npt",
            optional=False,
        ),
        "output_gro_suffix": StringParameter(
            label="Output gro Filename",
            default="_box",
            docstring="Output Gro file suffix, will be saved in the same folder as the input file",
            optional=True,
        ),
        "box_size_string": StringParameter(
            label="Box Size",
            default="20 20 20",
            docstring="Box size in nm, e.g. 20 20 20; set 0 to ignore this parameter",
            optional=True,
        ),
        "ion_conc": FloatParameter(
            label="Ion Concentration",
            default=0.15,
            docstring="Ion concentration in mol/L, if the value is 0, no ions will be added",
            optional=True,
        ),
        "scale_fill": FloatParameter(
            label="Scale Fill",
            default=0.57,
            docstring="Scale factor to multiply Van der Waals radii from the database.\
            The default value of 0.57 yields density close to 1000 g/l.",
            optional=True,
        ),
        "input_top_file": FileParameterEdit(
            label="Input Top File",
            docstring="Input top file to be processed",
            optional=False,
        ),
        "input_gro_file": FileParameterEdit(
            label="Input Gro File",
            docstring="Input Gro file to be processed",
            optional=False,
        ),
        "input_mdp_file": FileParameterEdit(
            label="Input mdp File",
            docstring="Input mdp file to be processed",
            optional=False,
        ),
        "input_ndx_file": FileParameterEdit(
            label="Input ndx File",
            docstring="Input NDX file to be processed",
            optional=False,
        ),
        "force_to_run": BooleanParameter(
            "Force to Run",
            default=False,
            docstring="If true, the node will be executed regardless of the database record",
            optional=True,
        ),
    }

    def execute(self, predecessor_data, flow_vars):
        try:
            # Initialize standard result
            result = NodeResult()
            result.metadata.update({"execution_time": datetime.now().isoformat()})

            # Collect input data
            input_data = predecessor_data[0] if predecessor_data else {}
            case_name = flow_vars["case_name"].get_value() or input_data.get(
                "case_name"
            )
            run_label = flow_vars["run_label"].get_value()

            # Get critical data from predecessor
            box_size_string = flow_vars["box_size_string"].get_value()
            ion_conc = flow_vars["ion_conc"].get_value()
            scale_fill = flow_vars["scale_fill"].get_value()

            # Get parameters and resolve paths
            input_top_file = flow_vars["input_top_file"].get_value()
            resolved_input_top_file = (
                self.resolve_path(input_top_file)
                if hasattr(self, "resolve_path")
                else input_top_file
            )

            input_gro_file = flow_vars["input_gro_file"].get_value()
            resolved_input_gro_file = (
                self.resolve_path(input_gro_file)
                if hasattr(self, "resolve_path")
                else input_gro_file
            )

            input_mdp_file = flow_vars["input_mdp_file"].get_value()
            resolved_input_mdp_file = (
                self.resolve_path(input_mdp_file)
                if hasattr(self, "resolve_path")
                else input_mdp_file
            )

            input_ndx_file = flow_vars["input_ndx_file"].get_value()
            resolved_input_ndx_file = (
                self.resolve_path(input_ndx_file)
                if hasattr(self, "resolve_path")
                else input_ndx_file
            )

            working_path = os.path.dirname(input_gro_file)

            # Set box array from string
            box_size = np.fromstring(box_size_string, dtype=float, sep=" ")

            log_message(f"Starting {self.node_key} execution for case: {case_name}")
            log_message(f"Run label: {run_label}")
            log_message(f"Resolved input TOP file: {resolved_input_top_file}")
            log_message(f"Resolved input GRO file: {resolved_input_gro_file}")
            log_message(f"Resolved input MDP file: {resolved_input_mdp_file}")
            log_message(f"Resolved input NDX file: {resolved_input_ndx_file}")
            log_message(f"Working directory: {working_path}")
            log_message(f"Box size: {box_size}")
            log_message(f"Ion concentration: {ion_conc}")
            log_message(f"Scale fill: {scale_fill}")

            # Store input files in result with URI prefixes
            result.files["input"].update(
                {
                    "input_top_file": (
                        self.format_output_path(resolved_input_top_file)
                        if hasattr(self, "format_output_path")
                        else resolved_input_top_file
                    ),
                    "input_gro_file": (
                        self.format_output_path(resolved_input_gro_file)
                        if hasattr(self, "format_output_path")
                        else resolved_input_gro_file
                    ),
                    "input_mdp_file": (
                        self.format_output_path(resolved_input_mdp_file)
                        if hasattr(self, "format_output_path")
                        else resolved_input_mdp_file
                    ),
                    "input_ndx_file": (
                        self.format_output_path(resolved_input_ndx_file)
                        if hasattr(self, "format_output_path")
                        else resolved_input_ndx_file
                    ),
                }
            )

            # Call the processing function with resolved paths
            processing_result = prepare_md_run(
                run_label,
                resolved_input_top_file,
                resolved_input_gro_file,
                resolved_input_mdp_file,
                resolved_input_ndx_file,
                working_path,
                scale_fill,
                box_size,
                ion_conc,
            )

            # Add working path to result with URI prefix
            formatted_working_path = (
                self.format_output_path(working_path)
                if hasattr(self, "format_output_path")
                else working_path
            )

            # Get output file paths
            out_gro_file = processing_result.get("out_gro")
            out_ndx_file = processing_result.get("out_ndx")
            box_gro_file = os.path.join(
                working_path, Path(resolved_input_gro_file).stem + "_box.gro"
            )
            solv_gro_file = os.path.join(
                working_path, Path(resolved_input_gro_file).stem + "_solv.gro"
            )
            solvion_top_file = os.path.join(
                working_path, Path(resolved_input_top_file).stem + "_solvion.top"
            )
            ion_tpr_file = os.path.join(working_path, "ion.tpr")

            # Format output paths
            formatted_out_gro_file = (
                self.format_output_path(out_gro_file)
                if hasattr(self, "format_output_path")
                else out_gro_file
            )
            formatted_out_ndx_file = (
                self.format_output_path(out_ndx_file)
                if hasattr(self, "format_output_path")
                else out_ndx_file
            )
            formatted_box_gro_file = (
                self.format_output_path(box_gro_file)
                if hasattr(self, "format_output_path")
                else box_gro_file
            )
            formatted_solv_gro_file = (
                self.format_output_path(solv_gro_file)
                if hasattr(self, "format_output_path")
                else solv_gro_file
            )
            formatted_solvion_top_file = (
                self.format_output_path(solvion_top_file)
                if hasattr(self, "format_output_path")
                else solvion_top_file
            )
            formatted_ion_tpr_file = (
                self.format_output_path(ion_tpr_file)
                if hasattr(self, "format_output_path")
                else ion_tpr_file
            )

            # Store data with formatted paths
            result.data.update(
                {
                    "case_name": case_name,
                    "run_label": run_label,
                    "working_path": formatted_working_path,
                    "success": processing_result.get("success", False),
                    "message": processing_result.get("message", ""),
                }
            )

            # Store output files with URI prefixes
            result.files["output"].update(
                {
                    "out_gro": formatted_out_gro_file,
                    "out_ndx": formatted_out_ndx_file,
                    "box_gro": formatted_box_gro_file,
                    "solv_gro": formatted_solv_gro_file,
                    "solvion_top": formatted_solvion_top_file,
                    "ion_tpr": formatted_ion_tpr_file,
                }
            )

            # Update data with essential information only
            result.data.update(
                {
                    "case_name": case_name,
                    "working_path": formatted_working_path,
                    "run_label": run_label,
                    "solvated_gro_file": formatted_out_gro_file,
                    "solvated_top_file": formatted_solvion_top_file,
                    "ndx_file": formatted_out_ndx_file,
                }
            )

            result.success = processing_result.get("success", False)
            result.message = (
                "Successfully prepared system with solvent and ions"
                if result.success
                else "Failed to prepare system with solvent and ions"
            )

            log_message(
                f"Successfully completed {self.node_key} execution for case: {case_name}"
            )
            return result.to_json()

        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            log_message(error_msg, "error")
            if isinstance(e, NodeException):
                raise
            raise NodeException(self.name, error_msg)


def prepare_md_run(
    run_label,
    top_file,
    gro_file,
    mdp_file,
    ndx_file,
    working_folder,
    scale_fill=0.57,
    box=None,
    ion_conc=0.15,
):
    """Prepare the MD system by adding box, solvent, and ions.

    This function works with resolved filesystem paths (no URI prefixes).
    All paths passed in should already be resolved by the execute function.
    All paths returned will be unformatted filesystem paths.

    Args:
        run_label: Label for the MD run (e.g., md, nvt)
        top_file: Path to the GROMACS topology file (.top)
        gro_file: Path to the GROMACS structure file (.gro)
        mdp_file: Path to the GROMACS parameter file (.mdp)
        ndx_file: Path to the GROMACS index file (.ndx)
        working_folder: Working directory path where outputs will be saved
        scale_fill: Scale factor for Van der Waals radii (default: 0.57)
        box: Box dimensions in nm as a list [x, y, z]
        ion_conc: Ion concentration in mol/L (default: 0.15)

    Returns:
        Dictionary with processing results and output file paths
    """
    output_filename = Path(gro_file).stem + "_box.gro"
    box_gro_file = os.path.join(working_folder, output_filename)

    box = list(box)

    # https://github.com/Becksteinlab/GromacsWrapper/blob/0052c5b6ef3ee323f611ddbc476024ae45b0d14c/gromacs/setup.py#L56
    if box[0] > 0:
        try:
            gromacs.editconf(f=gro_file, o=box_gro_file, box=box)
        except Exception as e:
            raise e
    else:
        gromacs.editconf(f=gro_file, o=box_gro_file, bt="triclinic", d=2.0)
    time.sleep(1)

    solv_gro_file = os.path.join(working_folder, Path(gro_file).stem + "_solv.gro")

    # Get a new solvion top file
    source_top_file = top_file
    in_top_file = os.path.join(working_folder, Path(top_file).stem + "_solvion.top")
    # Remove the new top file if exists
    if os.path.exists(in_top_file):
        os.remove(in_top_file)
    # Copy the source top file to the new top file
    shutil.copy2(source_top_file, in_top_file)
    time.sleep(0.5)

    # Using command line directly for solvate
    cmd = (
        "echo SOL | gmx solvate -cp "
        + box_gro_file
        + " -cs spc216.gro -p "
        + in_top_file
        + " -o "
        + solv_gro_file
        + " -scale "
        + str(scale_fill)
    )

    subprocess.call(cmd, shell=True, cwd=working_folder)
    time.sleep(0.5)

    out_tpr = os.path.join(working_folder, "ion.tpr")

    # Using command line directly for grompp
    cmd = (
        "gmx grompp -f "
        + mdp_file
        + " -c "
        + solv_gro_file
        + " -p "
        + in_top_file
        + " -o "
        + out_tpr
        + " -po "
        + os.path.join(working_folder, f"mdout_{run_label}.mdp")
        + " -maxwarn 10"
    )
    subprocess.call(cmd, shell=True, cwd=working_folder)
    time.sleep(0.5)

    ion_gro_file = os.path.join(working_folder, Path(gro_file).stem + "_ion.gro")

    # Add ions
    cmd = (
        " echo SOL | gmx "
        + " genion "
        + " -s "
        + out_tpr
        + " -p "
        + in_top_file
        + " -o "
        + ion_gro_file
        + " -neutral "
        + " -nname "
        + " CL "
        + " -pname "
        + " NA "
        + " -conc "
        + str(ion_conc)
    )
    subprocess.call(cmd, shell=True, cwd=working_folder)
    time.sleep(0.5)

    # Create index groups
    cmd = " echo q | " + " gmx make_ndx -f " + ion_gro_file + " -o " + ndx_file
    subprocess.call(cmd, shell=True, cwd=working_folder)
    time.sleep(0.5)

    # Process index file
    ndx = gromacs.fileformats.ndx.NDX()
    ndx.read(ndx_file)
    group_list = ndx.ndxlist
    dna_nr = None
    protein_nr = None
    for group in group_list:
        if group["name"] == "Protein":
            protein_nr = group["nr"] - 1
        if group["name"] == "DNA":
            dna_nr = group["nr"] - 1

    # Add combined group if both Protein and DNA are present
    if protein_nr and dna_nr:
        cmd = (
            ' echo "'
            + str(protein_nr)
            + " | "
            + str(dna_nr)
            + ' \n q "'
            + " | gmx make_ndx -f "
            + ion_gro_file
            + " -o "
            + ndx_file
        )
    subprocess.call(cmd, shell=True, cwd=working_folder)
    time.sleep(0.5)

    # Clean up temporary files
    CLEAN_HASH_GMX_FILES = 1
    if CLEAN_HASH_GMX_FILES:
        purge(working_folder, "^#...")
        purge(working_folder, "^step....pdb")

    # Check if output files exist
    ion_gro_success = os.path.exists(ion_gro_file)
    ndx_success = os.path.exists(ndx_file)

    return {
        "success": ion_gro_success and ndx_success,
        "message": (
            "Successfully prepared system with solvent and ions"
            if (ion_gro_success and ndx_success)
            else "Failed to prepare system"
        ),
        "out_gro": ion_gro_file,
        "out_ndx": ndx_file,
        "box_gro": box_gro_file,
        "solv_gro": solv_gro_file,
        "top_file": in_top_file,
    }


# Utils ---------------------------------------------------------------------
def purge(dir, pattern):
    """Remove files matching a pattern in the specified directory.

    Args:
        dir: Directory path to clean
        pattern: Regex pattern to match files for deletion
    """
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))
