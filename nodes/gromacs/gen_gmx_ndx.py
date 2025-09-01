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

# use gromacs wrapper
# conda install -c conda-forge gromacswrapper
# 0.9.1


class GenGmxOriNDX(Node):
    """GenGmxOriNDX
    This Node generates index groups (.ndx) files for GROMACS simulations.

    The node processes a PDB file to identify and group atoms based on their original
    structure versus those added through homology modeling. It creates two main groups:
    - OriHeavy: All heavy atoms from the original structure
    - OriBackBone: Backbone atoms (CA, O1P, P, O2P, O5', C5', C4', C3', O3') from
      the original structure

    [ Input ]
    - PDB structure file
    - System summary JSON file containing chain information
    - Chain selection (all chains or specific chain IDs)

    [ Output ]
    - index.ndx file with custom atom groups

    [ Dependencies ]
    Requires GROMACS and gromacswrapper (v0.9.1+) to be installed:
    conda install -c conda-forge gromacswrapper

    """

    # Basic configuration
    name = "GenGmxOriNDX"
    node_key = "GenGmxOriNDX"
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
        "input_pdb_file": FileParameterEdit(
            label="Input PDB File",
            docstring="Full PDB file to be processed",
            optional=False,
        ),
        "selected_chain_ids_string": StringParameter(
            label="Selected Chain IDs",
            default="all",
            docstring="chains need to be gapped by comma , e.g. A,B,C, all means all chains",
            optional=False,
        ),
        "system_summary_file": FileParameter(
            label="System Summary File",
            docstring="System summary file to be processed",
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
            case_name = flow_vars["case_name"].get_value() or input_data[
                "metadata"
            ].get("case_name")

            # Get parameters and resolve paths
            input_pdb_file = flow_vars["input_pdb_file"].get_value()
            resolved_input_pdb_file = (
                self.resolve_path(input_pdb_file)
                if hasattr(self, "resolve_path")
                else input_pdb_file
            )

            system_summary_file = flow_vars["system_summary_file"].get_value()
            resolved_system_summary_file = (
                self.resolve_path(system_summary_file)
                if hasattr(self, "resolve_path")
                else system_summary_file
            )

            selected_chain_ids_string = flow_vars[
                "selected_chain_ids_string"
            ].get_value()

            working_path = os.path.dirname(resolved_input_pdb_file)
            filename = Path(resolved_input_pdb_file).stem

            log_message(f"Starting {self.node_key} execution for case: {case_name}")
            log_message(f"Resolved input PDB file: {resolved_input_pdb_file}")
            log_message(f"Resolved system summary file: {resolved_system_summary_file}")
            log_message(f"Working directory: {working_path}")

            # Store input files in result with URI prefixes
            result.files["input"].update(
                {
                    "input_pdb_file": (
                        self.format_output_path(resolved_input_pdb_file)
                        if hasattr(self, "format_output_path")
                        else resolved_input_pdb_file
                    ),
                    "system_summary_file": (
                        self.format_output_path(resolved_system_summary_file)
                        if hasattr(self, "format_output_path")
                        else resolved_system_summary_file
                    ),
                }
            )

            # Get data from system summary file
            with open(resolved_system_summary_file, "r") as json_file:
                system_summary = json.load(json_file)
            pdb_chain_list = system_summary.get("pdb_chain_list")

            # Resolve paths in chain_data
            chain_data = system_summary.get("chain_data")
            resolved_chain_data = {}

            for chain_id, chain_info in chain_data.items():
                resolved_chain_info = {}
                for key, path in chain_info.items():
                    if key in [
                        "original_residues_file",
                        "missing_residues_file",
                    ] and isinstance(path, str):
                        resolved_path = (
                            self.resolve_path(path)
                            if hasattr(self, "resolve_path")
                            else path
                        )
                        resolved_chain_info[key] = resolved_path
                    else:
                        resolved_chain_info[key] = path
                resolved_chain_data[chain_id] = resolved_chain_info

            # Convert the ids_string to a list of chain ids
            if selected_chain_ids_string == "all":
                selected_pdb_chain_list = pdb_chain_list
            else:
                selected_pdb_chain_list = selected_chain_ids_string.split(",")

            # Call the processing function with resolved paths
            processing_result = gen_ori_ndx(
                case_name,
                resolved_input_pdb_file,
                selected_pdb_chain_list,
                resolved_chain_data,
                working_path,
            )

            # Add working path to result with URI prefix
            formatted_working_path = (
                self.format_output_path(working_path)
                if hasattr(self, "format_output_path")
                else working_path
            )

            # Define the output index.ndx file path
            index_ndx_path = os.path.join(working_path, "index.ndx")
            formatted_index_ndx_path = (
                self.format_output_path(index_ndx_path)
                if hasattr(self, "format_output_path")
                else index_ndx_path
            )

            # Store data with formatted paths
            result.data.update(
                {
                    "case_name": case_name,
                    "working_path": formatted_working_path,
                    "success": processing_result.get("success", False),
                    "selected_chains": selected_pdb_chain_list,
                }
            )

            # Store output files with URI prefixes
            result.files["output"].update({"index_ndx": formatted_index_ndx_path})

            # Update data with essential information only
            result.data.update(
                {
                    "case_name": case_name,
                    "working_path": formatted_working_path,
                    "ndx_file": formatted_index_ndx_path,
                    "structure_file": resolved_input_pdb_file,  # Pass along essential structure file
                }
            )

            # Preserve other essential data that this node doesn't modify
            if predecessor_data:
                for key in ["output_gro_file", "output_top_file", "force_field"]:
                    if key in input_data.get("data", {}) and key not in result.data:
                        result.data[key] = input_data["data"][key]

            result.success = processing_result.get("success", False)
            result.message = (
                "Successfully generated index groups (NDX) file"
                if result.success
                else "Failed to generate index groups"
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


def gen_ori_ndx(case_name, structure, select_pdb_chain_list, chain_data, working_path):
    """Generate original index groups for GROMACS.

    This function works with resolved filesystem paths (no URI prefixes).
    All paths passed in should already be resolved by the execute function.
    All paths returned will be unformatted filesystem paths.

    Args:
        case_name: Name of the case/protein
        structure: Path to PDB structure file (resolved)
        select_pdb_chain_list: List of selected chain IDs
        chain_data: Dictionary containing chain information with resolved paths
        working_path: Working directory path (resolved)

    Returns:
        Dictionary with processing results
    """
    md_folder = working_path
    in_pdb_file = structure

    ori_ndx_heavy_list = []
    ori_ndx_ca_list = []
    atom_list = read_pdb_simple(in_pdb_file)
    chain_list = []
    resid_list = []
    for atom in atom_list:
        resid_list.append(atom.resid)
        chain_list.append(atom.chain)

    df_demo = pd.DataFrame({"chain": chain_list, "resid": resid_list})
    resid_chain_dict = {}

    # tested in pdbmdauto, but not tested here
    for chain in sorted(select_pdb_chain_list):
        resid_min = df_demo[(df_demo["chain"] == chain)]["resid"].min()
        resid_chain_dict[chain] = resid_min

    for atom in atom_list:
        # _item_record = _df2.copy()[_df2["chain"].isin([atom.chain])]
        item_record = chain_data[atom.chain]
        ori_csv_file = item_record["original_residues_file"]
        df_ori = pd.read_csv(ori_csv_file)
        miss_csv_file = item_record["missing_residues_file"]
        df_miss = pd.read_csv(miss_csv_file)
        resid_list = sorted(
            df_ori["resid"].values.tolist() + df_miss["ssseq"].values.tolist()
        )

        resid_shift = resid_list[0] - resid_chain_dict[atom.chain]
        # print('resid_shift', resid_shift)

        if (
            atom.chain in df_miss["chain"].values.tolist()
            and (atom.resid + resid_shift) in df_miss["ssseq"].values.tolist()
        ):
            print(
                "Miss --",
                atom.chain,
                atom.index,
                atom.name,
                atom.resid,
                atom.element,
                case_name,
            )
        else:
            if atom.element != " ":
                ori_ndx_heavy_list.append(atom.index)
                if atom.name.strip() in [
                    "CA",
                    "O1P",
                    "P",
                    "O2P",
                    "O5'",
                    "C5'",
                    "C4'",
                    "C3'",
                    "O3'",
                ]:
                    ori_ndx_ca_list.append(atom.index)

    in_ndx = os.path.join(md_folder, "index.ndx")

    cmd = " echo q | " + " gmx make_ndx -f " + in_pdb_file + " -o " + in_ndx

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    time.sleep(0.5)

    ndx = gromacs.fileformats.ndx.NDX()
    ndx.read(in_ndx)
    ndx["OriHeavy"] = ori_ndx_heavy_list
    ndx["OriBackBone"] = ori_ndx_ca_list
    ndx.write(in_ndx)

    return {"success": True, "index_ndx_file": in_ndx}


# Utils ---------------------------------------------------------------------
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
