import json  # noqa: E402
import os
import re
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


class InitGmxRelax(Node):
    """InitGmxRelax
    This Node performs structure relaxation using GROMACS through a multi-step
    relaxation protocol. The protocol consists of four recommended steps:

    1. nvt_fixOri: NVT ensemble simulation with fixed original residues
    2. nvt_fixOriBackbone: NVT ensemble simulation with fixed backbone in original residues
    3. mm1: First energy minimization step
    4. mm2: Second energy minimization step

    [ Input ]
    - TOP file (.top): Topology file containing molecular structure and force field
    - GRO file (.gro): Structure file in GROMACS format
    - MDP file (.mdp): Parameters file for molecular dynamics
    - NDX file (.ndx): Index file defining atom groups

    [ Output ]
    - Relaxed structure files (.gro)
    - Trajectory files (.trr)
    - Energy files (.edr)
    - Log files (.log)

    [ Dependencies ]
    Requires GROMACS and gromacswrapper (v0.9.1+) to be installed:
    conda install -c conda-forge gromacswrapper

    """

    # Basic configuration
    name = "InitGmxRelax"
    node_key = "InitGmxRelax"
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
        "run_label": SelectParameter(
            "Run steps with the labels",
            options=["nvt_fixOri", "nvt_fixOriBackbone", "mm1", "mm2"],
            default="nvt_fixOri",
            docstring="run label",
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
        "input_ndx_file": FileParameterEdit(
            label="Input ndx File",
            docstring="Input NDX file to be processed",
            optional=False,
        ),
        "input_mdp_file": FileParameterEdit(
            label="Input mdp File",
            docstring="Input mdp file to be processed",
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

            working_path = os.path.dirname(resolved_input_gro_file)

            log_message(f"Starting {self.node_key} execution for case: {case_name}")
            log_message(f"Run label: {run_label}")
            log_message(f"Resolved input TOP file: {resolved_input_top_file}")
            log_message(f"Resolved input GRO file: {resolved_input_gro_file}")
            log_message(f"Resolved input MDP file: {resolved_input_mdp_file}")
            log_message(f"Resolved input NDX file: {resolved_input_ndx_file}")
            log_message(f"Working directory: {working_path}")

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
            processing_result = init_md(
                run_label,
                resolved_input_top_file,
                resolved_input_gro_file,
                resolved_input_mdp_file,
                resolved_input_ndx_file,
                working_path,
            )

            # Add working path to result with URI prefix
            formatted_working_path = (
                self.format_output_path(working_path)
                if hasattr(self, "format_output_path")
                else working_path
            )

            # Define the output file paths
            out_gro_path = os.path.join(working_path, f"{run_label}.gro")
            out_tpr_path = os.path.join(working_path, f"{run_label}.tpr")
            out_trr_path = os.path.join(working_path, f"{run_label}.trr")
            out_edr_path = os.path.join(working_path, f"{run_label}.edr")
            out_log_path = os.path.join(working_path, f"{run_label}.log")
            out_mdp_path = os.path.join(working_path, f"mdout_{run_label}.mdp")

            # Format output paths
            formatted_out_gro_path = (
                self.format_output_path(out_gro_path)
                if hasattr(self, "format_output_path")
                else out_gro_path
            )
            formatted_out_tpr_path = (
                self.format_output_path(out_tpr_path)
                if hasattr(self, "format_output_path")
                else out_tpr_path
            )
            formatted_out_trr_path = (
                self.format_output_path(out_trr_path)
                if hasattr(self, "format_output_path")
                else out_trr_path
            )
            formatted_out_edr_path = (
                self.format_output_path(out_edr_path)
                if hasattr(self, "format_output_path")
                else out_edr_path
            )
            formatted_out_log_path = (
                self.format_output_path(out_log_path)
                if hasattr(self, "format_output_path")
                else out_log_path
            )
            formatted_out_mdp_path = (
                self.format_output_path(out_mdp_path)
                if hasattr(self, "format_output_path")
                else out_mdp_path
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
                    "out_gro": formatted_out_gro_path,
                    "out_tpr": formatted_out_tpr_path,
                    "out_trr": formatted_out_trr_path,
                    "out_edr": formatted_out_edr_path,
                    "out_log": formatted_out_log_path,
                    "out_mdp": formatted_out_mdp_path,
                }
            )

            # Update data with essential information only
            result.data.update(
                {
                    "case_name": case_name,
                    "working_path": formatted_working_path,
                    "run_label": run_label,
                    "md_gro_file": formatted_out_gro_path,
                }
            )

            result.success = processing_result.get("success", False)
            result.message = (
                processing_result.get(
                    "message", f"Successfully completed {run_label} relaxation step"
                )
                if result.success
                else f"Failed to complete {run_label} relaxation step"
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
            raise NodeException("execute", error_msg)


def init_md(run_label, top_file, gro_file, mdp_file, ndx_file, working_folder):
    """Initialize and run a GROMACS molecular dynamics simulation step.

    This function works with resolved filesystem paths (no URI prefixes).
    All paths passed in should already be resolved by the execute function.
    All paths returned will be unformatted filesystem paths.

    Args:
        run_label: Label for the MD run (e.g., nvt_fixOri, mm1)
        top_file: Path to the GROMACS topology file (.top)
        gro_file: Path to the GROMACS structure file (.gro)
        mdp_file: Path to the GROMACS parameter file (.mdp)
        ndx_file: Path to the GROMACS index file (.ndx)
        working_folder: Working directory path where outputs will be saved

    Returns:
        Dictionary with processing results
    """
    print(f"--- running init_md in step {run_label}")
    CLEAN_HASH_GMX_FILES = 1
    if CLEAN_HASH_GMX_FILES:
        purge(working_folder, "^#...")
        purge(working_folder, "^step....pdb")

    out_tpr = os.path.join(working_folder, f"{run_label}.tpr")
    gromacs.grompp(
        f=mdp_file,
        c=gro_file,
        p=top_file,
        o=out_tpr,
        n=ndx_file,
        po=os.path.join(working_folder, f"mdout_{run_label}.mdp"),
        maxwarn=10,
    )
    currf = os.getcwd()
    os.chdir(working_folder)
    gromacs.mdrun(v=True, deffnm=run_label)
    os.chdir(currf)

    CLEAN_HASH_GMX_FILES = 1
    if CLEAN_HASH_GMX_FILES:
        purge(working_folder, "^#...")
        purge(working_folder, "^step....pdb")

    # Check if output files exist
    out_gro_file = os.path.join(working_folder, f"{run_label}.gro")
    out_success = os.path.exists(out_gro_file)

    return {
        "success": out_success,
        "message": (
            "Successfully completed GROMACS MD step"
            if out_success
            else "Failed to complete GROMACS MD step"
        ),
        "out_gro": f"{run_label}.gro",
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


class PdbAtom:
    """
    simple one
    """

    def __init__(self, label, index, name, residue, chain, resid, x, y, z, element):
        self.label = label
        self.index = index
        self.name = name
        # self.indicator = indicator
        self.residue = residue
        self.chain = chain
        self.resid = resid
        # self.insert    = insert
        self.x = x
        self.y = y
        self.z = z
        # self.occu      = occu
        # self.temp      = temp
        # self.seg       = seg
        self.element = element
        # self.charge    = charge


def read_pdb_simple(inputfilename):
    """Read and parse PDB file into atom list.

    This function works with resolved filesystem paths (no URI prefixes).

    Args:
        inputfilename: Path to input PDB file (resolved)

    Returns:
        List of PdbAtom objects
    """
    atom_list = []
    with open(inputfilename, "r") as f:
        data = f.readlines()
        for line in data:
            list = line.split()
            id = list[0]
            if id == "ATOM":  # 'ATOM'
                label = line[:6]
                index = int(line[6:11])
                name = line[12:16]
                residue = line[17:20]
                chain = line[21]
                resid = int(line[22:26])
                # insert    = line[26]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                # occu      = float(line[54:60])
                # temp      = float(line[60:66])
                # seg       = line[72:76]
                # element   = line[76:78]
                element = line[77]
                # charge    = line[78:80]
                atom_list.append(
                    PdbAtom(label, index, name, residue, chain, resid, x, y, z, element)
                )
    return atom_list
