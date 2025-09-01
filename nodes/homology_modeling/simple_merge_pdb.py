import json  # noqa: E402
import os
import re
import sys
import textwrap
from datetime import datetime

import pymol

# import execution dependent modules
from pymol import cmd

from bocoflow_core.logger import log_message
from bocoflow_core.node import Node, NodeException, NodeResult
from bocoflow_core.parameters import *


class MergePDB(Node):
    """MergePDB

    This node merges multiple PDB chain files into a single PDB structure while
    maintaining proper chain IDs and handling both protein and non-protein (DNA) chains.

    [ Function ]
    - merges multiple PDB chain files into one structure
    - maintains chain IDs during merging
    - handles both protein and non-protein (DNA) chains
    - preserves atom connectivity
    - supports selective chain merging
    - generates chain type and name records
    - maintains atom order during merging
    - handles HETATM records appropriately

    [ Input ]
    - Case name (optional, can use predecessor data)
    - Merge folder name (optional, can use predecessor data)
    - Merge file name (optional, can use predecessor data)
    - Selected chain IDs (comma-separated or 'all')
    - Chain data including PDB files and types
    - Output directory information

    [ Output ]
    Files:
    - Merged PDB structure file
    - Chain type JSON record
    - Chain name JSON record

    Data:
    - Merged structure file path
    - Processing status
    - Chain selection information

    [ Note ]
    The node requires proper pymol installation with specific openssl version (3.1.3)
    for compatibility. It preserves atom connectivity and order during merging,
    which is crucial for subsequent molecular dynamics simulations.
    """

    # Basic configuration
    name = "Merge PDB Chains"
    node_key = "MergePDB"
    color = "green"
    num_in = 1
    num_out = 1

    # Define the parameters from GUI
    OPTIONS = {
        "case_name": StringParameter(
            label="Case Name",
            default="",
            docstring="Name of the case/protein (leave empty to use predecessor data)",
            optional=True,
        ),
        "merge_folder_name": StringParameter(
            label="Merge Folder Name",
            default="Merge",
            docstring="sub folder name for merged files (leave empty to use predecessor data)",
            optional=True,
        ),
        "merge_file_name": StringParameter(
            label="Merge File Name",
            default="merge.pdb",
            docstring="name for merged file (leave empty to use predecessor data)",
            optional=True,
        ),
        "selected_chain_ids_string": StringParameter(
            label="Selected Chain IDs",
            default="all",
            docstring="chains need to be gapped by comma , e.g. A,B,C, all means all chains",
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

            # Parse the input data
            input_data = predecessor_data[0]

            # Use predecessor data if option variables are not provided
            case_name = flow_vars["case_name"].get_value() or input_data[
                "metadata"
            ].get("case_name")
            merge_folder_name = flow_vars[
                "merge_folder_name"
            ].get_value() or input_data["data"].get("merge_folder_name", "Merge")
            merge_file_name = flow_vars["merge_file_name"].get_value() or input_data[
                "data"
            ].get("merge_file_name", "merge.pdb")
            selected_chain_ids_string = flow_vars[
                "selected_chain_ids_string"
            ].get_value()

            # Give error if compulsory variables are not provided
            if not case_name or not merge_folder_name:
                raise NodeException(
                    self.name, "case_name or merge_folder must be provided"
                )

            # Get data from input data
            pdb_chain_list = input_data["data"].get("pdb_chain_list")
            output_dir = input_data["data"].get("output_dir")
            chain_data = input_data["data"].get("chain_data")
            chain_type = input_data["data"].get("chainType")

            # Resolve the output directory path if it has a URI prefix
            resolved_output_dir = (
                self.resolve_path(output_dir)
                if hasattr(self, "resolve_path")
                else output_dir
            )
            log_message(f"Resolved output directory: {resolved_output_dir}")

            # Get the merge folder via combining the output folder and the merge folder name
            merge_folder = os.path.join(resolved_output_dir, merge_folder_name)
            os.makedirs(merge_folder, exist_ok=True)
            merge_file = os.path.join(merge_folder, merge_file_name)

            # Convert the ids_string to a list of chain ids
            if selected_chain_ids_string == "all":
                selected_pdb_chain_list = pdb_chain_list
            else:
                selected_pdb_chain_list = selected_chain_ids_string.split(",")

            # Check if the selected_pdb_chain_list is in the pdb_chain_list
            for chain_id in selected_pdb_chain_list:
                if chain_id not in pdb_chain_list:
                    raise NodeException(
                        self.name, f"chain_id {chain_id} is not in the pdb_chain_list"
                    )

            # Resolve paths in chain_data
            resolved_chain_data = {}
            for chain_id, data in chain_data.items():
                resolved_chain_data[chain_id] = {}
                for key, path in data.items():
                    resolved_chain_data[chain_id][key] = (
                        self.resolve_path(path)
                        if hasattr(self, "resolve_path")
                        else path
                    )

            log_message(f"Resolved paths for {len(resolved_chain_data)} chains")

            # Saving chain type and name files in the merge folder
            out_chain_type_file = os.path.join(
                merge_folder, f"{case_name}_chain_type.json"
            )
            with open(out_chain_type_file, "w") as outfile:
                json.dump(chain_type, outfile)

            out_chain_name_file = os.path.join(
                merge_folder, f"{case_name}_chain_name.json"
            )
            with open(out_chain_name_file, "w") as outfile:
                json.dump(selected_pdb_chain_list, outfile)

            # Call the processing function with resolved paths
            processing_result = self.remerge_pdb_chains(
                merge_file, resolved_chain_data, chain_type, selected_pdb_chain_list
            )

            # Format output paths with URI prefixes
            formatted_merge_file = (
                self.format_output_path(merge_file)
                if hasattr(self, "format_output_path")
                else merge_file
            )
            formatted_chain_type_file = (
                self.format_output_path(out_chain_type_file)
                if hasattr(self, "format_output_path")
                else out_chain_type_file
            )
            formatted_chain_name_file = (
                self.format_output_path(out_chain_name_file)
                if hasattr(self, "format_output_path")
                else out_chain_name_file
            )

            # Store processing results
            result.data.update({"output_pdb": formatted_merge_file})

            # Record output files with formatted paths
            result.files["output"].update(
                {
                    "merged_pdb_file": formatted_merge_file,
                    "chain_type_file": formatted_chain_type_file,
                    "chain_name_file": formatted_chain_name_file,
                }
            )

            # Update metadata
            result.metadata.update(
                {
                    "case_name": case_name,
                    "execution_time": datetime.now().isoformat(),
                    "output_dir": (
                        self.format_output_path(resolved_output_dir)
                        if hasattr(self, "format_output_path")
                        else output_dir
                    ),
                }
            )

            # Store essential data for downstream nodes in a flat structure
            result.data.update(
                {
                    "case_name": case_name,
                    "output_dir": (
                        self.format_output_path(resolved_output_dir)
                        if hasattr(self, "format_output_path")
                        else output_dir
                    ),
                    "pdb_chain_list": pdb_chain_list,
                    "chain_data": chain_data,
                    "selected_pdb_chain_list": selected_pdb_chain_list,
                    "merge_folder": (
                        self.format_output_path(merge_folder)
                        if hasattr(self, "format_output_path")
                        else merge_folder
                    ),
                    "merge_file": formatted_merge_file,
                }
            )

            result.success = True
            result.message = "Successfully merged PDB chains"

            return result.to_json()

        except Exception as e:
            log_message(f"Error in MergePDB: {str(e)}")
            raise NodeException(self.name, str(e))

    def remerge_pdb_chains(
        self, merge_file, chain_data, chain_type, selected_pdb_chain_list
    ):
        """Merge multiple PDB chain files into a single PDB structure.

        This function works with resolved filesystem paths (no URI prefixes).
        All paths passed in should already be resolved by the execute function.
        All paths returned will be unformatted filesystem paths.

        Args:
            merge_file: Path to output merged PDB file
            chain_data: Dictionary with chain data including PDB file paths
            chain_type: Dictionary mapping chains to their types
            selected_pdb_chain_list: List of chain IDs to include in merge

        Returns:
            Dictionary containing:
            - output_pdb: Path to the merged PDB file
        """
        try:
            import pymol
            from pymol import cmd
        except ImportError:
            raise NodeException(self.name, "PyMOL is required but not installed")

        log_message(f"Merging {len(selected_pdb_chain_list)} PDB chains")
        list_pdb_object = []

        for chain in sorted(selected_pdb_chain_list):
            log_message(f"Processing chain {chain}")

            cmd.set("retain_order", 1)
            cmd.set("connect_mode", 1)

            in_chain_type = chain_type.get(chain)
            in_chain_record = chain_data.get(chain)

            if not in_chain_record:
                raise NodeException(
                    self.name, f"Chain data not found for chain {chain}"
                )

            chain_pdb_file = in_chain_record.get("no_het_file")

            if not chain_pdb_file:
                raise NodeException(self.name, f"PDB file not found for chain {chain}")

            log_message(f"Loading PDB file: {chain_pdb_file}")
            cmd.load(chain_pdb_file)

            # Extract object name from file path
            id_ref = os.path.basename(chain_pdb_file)
            id_ref = re.sub("\.pdb", "", id_ref)

            # Set chain ID
            cmd.alter(id_ref, 'chain = "{}"'.format(chain))
            list_pdb_object.append(id_ref)

            # Handle DNA/RNA chains
            if in_chain_type == "DL":
                cmd.alter(id_ref, 'type = "HETATM"')

        # Save merged structure
        log_message(f"Saving merged PDB to: {merge_file}")
        cmd.save(merge_file, "(" + ",".join(list_pdb_object) + ")")
        cmd.delete("all")

        result = {}
        result["output_pdb"] = merge_file
        return result
