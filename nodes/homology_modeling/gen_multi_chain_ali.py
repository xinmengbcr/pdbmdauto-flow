import json  # noqa: E402
import os
import re
import sys
import textwrap
from datetime import datetime

import pandas as pd
from Bio import SeqIO  # noqa: E402
from Bio.Seq import Seq  # noqa: E402
from Bio.SeqRecord import SeqRecord  # noqa: E402

from bocoflow_core.logger import log_message
from bocoflow_core.node import Node, NodeException, NodeResult
from bocoflow_core.parameters import *


class GenMultiChainAli(Node):
    """GenMultiChainAli

    This node generates a merged alignment file for multi-chain homology modeling
    using MODELLER. It combines individual chain alignments into a single file
    while maintaining proper chain separation and sequence formatting.

    [ Lib dependence ]
    - biopython: 1.84 (tested)
    - pandas: for data processing

    [ Function ]
    - merges individual chain alignment files
    - handles both protein and DNA/RNA chains
    - maintains proper chain separation with '/' markers
    - generates MODELLER-compatible alignment format
    - supports selective chain processing
    - validates sequence consistency
    - handles mixed chain types (protein/DNA/RNA)

    [ Input ]
    - Case name (optional, can use predecessor data)
    - Merge folder name for output
    - Selected chain IDs (comma-separated or 'all')
    - Individual chain alignment files from GenAlignFileNode
    - Chain type information and sequence data

    [ Output ]
    Files:
    - Combined MODELLER alignment file (homology.ali)
    - Contains both template and target sequences
    - Properly formatted for multi-chain modeling

    Data:
    - Processing status
    - Chain type dictionary
    - Alignment file path
    - Sequence validation results

    [ Note ]
    The node requires properly formatted individual chain alignments as input.
    It preserves chain order and handles the transition between chains using
    appropriate markers. Special care is taken to maintain MODELLER's alignment
    format requirements for multi-chain systems.
    """

    # Basic configuration
    name = "Multi-Chain Ali"
    node_key = "GenMultiChainAli"
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
            docstring="subfolder name for merged file, which will be combined with output_dir to form the merge folder",
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
            merge_folder_name = flow_vars["merge_folder_name"].get_value()
            selected_chain_ids_string = flow_vars[
                "selected_chain_ids_string"
            ].get_value()

            # Give error if compulsory variables are not provided
            if not case_name or not merge_folder_name or not selected_chain_ids_string:
                raise NodeException(
                    self.name,
                    "case_name or merge_folder or ids_string must be provided",
                )

            # Get data from input data
            pdb_chain_list = input_data["data"].get("pdb_chain_list")
            chain_data = input_data["data"].get("chain_data")
            output_dir = input_data["data"].get("output_dir")

            # Resolve the output directory path if it has a URI prefix
            resolved_output_dir = (
                self.resolve_path(output_dir)
                if hasattr(self, "resolve_path")
                else output_dir
            )
            log_message(f"Resolved output directory: {resolved_output_dir}")

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

            # Get the merge folder via combining the output folder and the merge folder name
            merge_folder = os.path.join(resolved_output_dir, merge_folder_name)

            # Create the merge folder if it does not exist
            os.makedirs(merge_folder, exist_ok=True)

            # Get the chain alignment files and resolve their paths
            chain_ali_file_list = input_data["files"]["output"].get(
                "modeller_chain_ali_files"
            )
            resolved_chain_ali_file_list = []

            # Resolve all chain alignment file paths
            for ali_file in chain_ali_file_list:
                resolved_path = (
                    self.resolve_path(ali_file)
                    if hasattr(self, "resolve_path")
                    else ali_file
                )
                resolved_chain_ali_file_list.append(resolved_path)

            log_message(
                f"Resolved {len(resolved_chain_ali_file_list)} chain alignment files"
            )

            # Call the real execution function with resolved paths
            processing_result = self.process_ali_for_multi_chain_modeller(
                case_name,
                merge_folder,
                selected_pdb_chain_list,
                pdb_chain_list,
                resolved_chain_ali_file_list,
            )

            # Format the output alignment file path with URI prefix if needed
            ali_file_path = processing_result["aliFileMultiChain"]
            formatted_ali_file_path = (
                self.format_output_path(ali_file_path)
                if hasattr(self, "format_output_path")
                else ali_file_path
            )

            # Store processing results with formatted paths
            result.data.update(
                {
                    "processAliMultiChainModeller": processing_result[
                        "processAliMultiChainModeller"
                    ],
                    "chainType": processing_result["chainType"],
                }
            )

            # Record output files with formatted paths
            result.files["output"].update(
                {"modeller_multi_chain_ali_file": formatted_ali_file_path}
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

            # Preserve necessary upstream data for downstream nodes
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
                }
            )

            result.success = True
            result.message = "Successfully generated multi-chain alignment file"

            return result.to_json()

        except Exception as e:
            log_message(f"Error in GenMultiChainAli: {str(e)}")
            raise NodeException(self.name, str(e))

    def process_ali_for_multi_chain_modeller(
        self,
        case_name,
        merge_folder,
        selected_pdb_chain_list,
        pdb_chain_list,
        chain_ali_file_list,
    ):
        """Process chain alignment files to generate a multi-chain MODELLER alignment file.

        This function works with resolved filesystem paths (no URI prefixes).
        All paths passed in should already be resolved by the execute function.
        All paths returned will be unformatted filesystem paths.

        Args:
            case_name: Name of the case/protein
            merge_folder: Directory to save the merged alignment file
            selected_pdb_chain_list: List of selected chain IDs
            pdb_chain_list: Full list of all chain IDs
            chain_ali_file_list: List of resolved paths to individual chain alignment files

        Returns:
            Dictionary containing:
            - processAliMultiChainModeller: Success status ('Y' or 'N')
            - aliFileMultiChain: Path to the generated multi-chain alignment file
            - chainType: Dictionary mapping chain IDs to their types (P1 or DL)
        """
        # Load the input data
        seq_ori = []
        seq_full = []
        chain_protein_start = ""
        chain_protein_end = ""
        chain_list_only_protein = []
        type_dict = {}

        log_message(f"Processing multi-chain alignment for case: {case_name}")
        log_message(f"Selected chains: {', '.join(selected_pdb_chain_list)}")

        # Looping through the chains that need to be processed
        for chain in sorted(selected_pdb_chain_list):
            # Locate the ali_file
            chain_index = pdb_chain_list.index(chain)
            ali_file = chain_ali_file_list[chain_index]
            log_message(f"Processing chain {chain} alignment file: {ali_file}")

            # Parse the ali file
            for record in SeqIO.parse(ali_file, "fasta"):
                chain_type = record.description.split(";")[0]
                chain_seq = str(record._seq.split(":")[-1])
                catagory = re.split(";|_", record.description)[-1]

                # If handling the last chain?
                if chain == sorted(selected_pdb_chain_list)[-1]:
                    pass
                else:
                    demo_list = list(chain_seq)
                    demo_list[-1] = "/"
                    chain_seq = "".join(demo_list)

                if chain_type == "P1":
                    if catagory == "full":
                        chain_list_only_protein.append(chain)
                        type_dict[chain] = "P1"
                else:
                    demo_list = list(chain_seq)
                    demo_list = [
                        "." if (x != "*" and x != "/") else x for x in demo_list
                    ]
                    chain_seq = "".join(demo_list)
                    if catagory == "full":
                        type_dict[chain] = "DL"
                if catagory == "full":
                    seq_full += chain_seq
                else:
                    seq_ori += chain_seq

        # Set start and end protein chains
        if chain_list_only_protein:
            chain_protein_start = chain_list_only_protein[0]
            chain_protein_end = chain_list_only_protein[-1]

        # Create the output alignment file path
        _modeller_multi_chain_ali_file = os.path.join(merge_folder, "homology.ali")

        # Prepare alignment file content
        _ali_code = "P1"
        _first_line = ">" + _ali_code + ";" + case_name
        _f1 = "structure"
        _f2 = "merge"
        _f3 = ""
        _f4 = str(chain_protein_start)
        _f5 = ""
        _f6 = str(chain_protein_end)
        _f7 = ""
        _f8 = ""
        _f9 = ""
        _f10 = ""
        _second_line = ":".join([_f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8, _f9, _f10])
        _third_line = textwrap.fill("".join(list(seq_ori)), width=60)
        _first_line_2nd = ">" + _ali_code + ";" + case_name + "_full"
        _second_line_2nd = "sequence:::::::::"
        _third_line_2nd = textwrap.fill("".join(list(seq_full)), width=60)

        # Validation check
        _modeller_ali_file_list = []
        if len(_third_line) != len(_third_line_2nd):
            _modeller_ali_file_list.append(" ")
            log_message(
                "ERROR: Sequence length mismatch between structure and sequence"
            )
            raise Exception("ERROR, sequence does not match")
        else:
            # Write the alignment file
            with open(_modeller_multi_chain_ali_file, "w") as out_file:
                out_file.write(_first_line + "\n")
                out_file.write(_second_line + "\n")
                out_file.write(_third_line + "\n")
                out_file.write("\n")
                out_file.write(_first_line_2nd + "\n")
                out_file.write(_second_line_2nd + "\n")
                out_file.write(_third_line_2nd + "\n")
            log_message(
                f"Successfully wrote multi-chain alignment to: {_modeller_multi_chain_ali_file}"
            )

        # Prepare result
        result = {}
        if " " in _modeller_ali_file_list:
            result["processAliMultiChainModeller"] = "N"
        else:
            result["processAliMultiChainModeller"] = "Y"
        result["aliFileMultiChain"] = _modeller_multi_chain_ali_file
        result["chainType"] = type_dict

        return result
