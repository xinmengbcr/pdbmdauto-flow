import json  # noqa: E402
import os
import sys
import textwrap  # noqa: E402
from datetime import datetime

import pandas as pd
from Bio import SeqIO  # noqa: E402
from Bio.PDB import PDBIO, PDBParser, Select  # noqa: E402
from Bio.SeqRecord import SeqRecord  # noqa: E402

from bocoflow_core.logger import log_message
from bocoflow_core.node import IONode, Node, NodeException, NodeResult
from bocoflow_core.parameters import *


class GenAlignFileNode(Node):
    """GenAlignFileNode

    This is a node that generates alignment files for homology modeling
    using MODELLER. It processes PDB structure and FASTA sequence data to create
    properly formatted alignment files.

    [ Lib dependence ]
    - biopython: 1.84 (tested)
    - pandas: for data processing

    [ Function ]
    - generates raw FASTA sequence alignment files (.seq)
    - creates MODELLER-specific alignment files (.ali)
    - processes each chain separately
    - verifies sequence agreement between PDB and FASTA
    - handles missing residues in alignment format
    - generates chain-specific directories
    - produces sequence validation reports
    - supports DNA/RNA and protein sequences

    [ Input ]
    - Case name (optional, can use predecessor data)
    - Output directory (optional, can use predecessor data)
    - PDB chain list from predecessor node
    - Chain data including missing residues
    - FASTA sequence data

    [ Output ]
    Files:
    - Chain alignment files (.seq): raw sequence alignments
    - MODELLER alignment files (.ali): template and target sequences

    Data:
    - Sequence agreement status between PDB and FASTA
    - Processing status for single chain MODELLER
    - Sequence validation results
    - Chain-specific alignment paths

    [ Note ]
    The node maintains data continuity by preserving necessary upstream data
    for downstream nodes. It handles both protein and DNA/RNA sequences with
    appropriate alignment codes (P1 for protein, DL for DNA).
    """

    name = "create Alignment File"
    num_in = 1
    num_out = 1

    OPTIONS = {
        "case_name": StringParameter(
            "Case Name",
            default="",
            docstring="Name of the case/protein (leave empty to use predecessor data)",
            optional=True,
        ),
        "output_dir": FolderParameter(
            "Output Directory",
            default="",
            docstring="Output directory for generated files (leave empty to use predecessor data)",
            optional=True,
        ),
        "append_end_in_seq": BooleanParameter(
            "Append End in Sequence",
            default=True,
            docstring="Whether to append '*' at the end of the sequence",
        ),
        "force_to_run": BooleanParameter(
            "Force to Run",
            default=False,
            docstring="If true, the node will be executed regardless of the database record",
        ),
    }

    def execute(self, predecessor_data, flow_vars):
        try:
            # Initialize standard result
            result = NodeResult()

            # Parse input data
            input_data = predecessor_data[0]

            # Use predecessor data if case_name or output_dir is empty
            case_name = flow_vars["case_name"].get_value() or input_data[
                "metadata"
            ].get("case_name")
            output_dir = flow_vars["output_dir"].get_value() or input_data[
                "metadata"
            ].get("output_dir")

            if not case_name or not output_dir:
                raise NodeException(
                    "generate alignment file",
                    "Case name and output directory must be provided either in the node options or from predecessor data.",
                )

            # Resolve output directory path if it has a URI prefix
            resolved_output_dir = (
                self.resolve_path(output_dir)
                if hasattr(self, "resolve_path")
                else output_dir
            )
            log_message(f"Resolved output directory: {resolved_output_dir}")

            # Get required input data
            pdb_chain_list = input_data["data"].get("pdb_chain_list")
            chain_data = input_data["data"].get("chain_data")
            fasta_data = input_data["data"].get("fasta_data")

            if not all([pdb_chain_list, chain_data, fasta_data]):
                raise NodeException(
                    "generate alignment file",
                    "Missing required data from predecessor node.",
                )

            # Resolve all paths in chain_data and fasta_data before processing
            resolved_chain_data = {}
            for chain_id, data in chain_data.items():
                resolved_chain_data[chain_id] = {}
                for key, path in data.items():
                    resolved_chain_data[chain_id][key] = (
                        self.resolve_path(path)
                        if hasattr(self, "resolve_path")
                        else path
                    )

            resolved_fasta_data = {}
            for chain_id, data in fasta_data.items():
                resolved_fasta_data[chain_id] = dict(data)
                if "file" in resolved_fasta_data[chain_id]:
                    resolved_fasta_data[chain_id]["file"] = (
                        self.resolve_path(data["file"])
                        if hasattr(self, "resolve_path")
                        else data["file"]
                    )

            # Process alignments with resolved paths
            processing_result = self.process_ali_for_single_chain_modeller(
                case_name,
                resolved_output_dir,  # Use resolved path for processing
                flow_vars["append_end_in_seq"].get_value(),
                pdb_chain_list,
                resolved_chain_data,  # Use resolved paths
                resolved_fasta_data,  # Use resolved paths
            )

            # Format paths in processing result for storage
            formatted_chain_ali_files = []
            formatted_modeller_ali_files = []

            if hasattr(self, "format_output_path"):
                for path in processing_result["chainAliFile"]:
                    formatted_chain_ali_files.append(self.format_output_path(path))
                for path in processing_result["modellerChainAliFile"]:
                    formatted_modeller_ali_files.append(self.format_output_path(path))
            else:
                formatted_chain_ali_files = processing_result["chainAliFile"]
                formatted_modeller_ali_files = processing_result["modellerChainAliFile"]

            # Store main processing results
            result.data.update(
                {
                    "seqAgreePdbFasta": processing_result["seqAgreePdbFasta"],
                    "processSingleChainModeller": processing_result[
                        "processSingleChainModeller"
                    ],
                    "seqCheck": processing_result["seqCheck"],
                }
            )

            # Record output files with formatted paths
            result.files["output"].update(
                {
                    "chain_ali_files": formatted_chain_ali_files,
                    "modeller_chain_ali_files": formatted_modeller_ali_files,
                }
            )

            # Update metadata with formatted path
            formatted_output_dir = (
                self.format_output_path(resolved_output_dir)
                if hasattr(self, "format_output_path")
                else output_dir
            )
            result.metadata.update(
                {
                    "case_name": case_name,
                    "execution_time": datetime.now().isoformat(),
                    "output_dir": formatted_output_dir,
                }
            )

            # Store essential data for downstream nodes in a flat structure
            result.data.update(
                {
                    "case_name": case_name,
                    "output_dir": formatted_output_dir,
                    "pdb_chain_list": pdb_chain_list,
                    "chain_data": chain_data,  # Use original chain_data with URI prefixes
                    "chain_ali_files": formatted_chain_ali_files,
                    "modeller_chain_ali_files": formatted_modeller_ali_files,
                }
            )

            result.success = True
            result.message = "Successfully generated alignment files"

            return result.to_json()

        except Exception as e:
            log_message(f"Error in GenAlignFileNode: {str(e)}")
            raise NodeException("generate alignment file", str(e))

    def process_ali_for_single_chain_modeller(
        self,
        case_name,
        output_dir,
        append_end_in_seq,
        pdb_chain_list,
        chain_data,
        fasta_data,
    ):
        """Process chain data to generate alignment files for MODELLER

        This function works with resolved filesystem paths (no URI prefixes).
        All paths passed in should already be resolved by the execute function.
        All paths returned will be unformatted filesystem paths.
        """
        ali_record = []
        compare_record_list = []
        modeller_ali_file_list = []
        chain_ali_file_list = []

        log_message(f"---- Processing alignment files for case: {case_name}")

        for chain in pdb_chain_list:
            chain_folder = os.path.join(output_dir, chain)
            os.makedirs(chain_folder, exist_ok=True)
            log_message(f"Processing chain: {chain}")

            # Use the resolved path directly
            chain_split_fasta_file = fasta_data[chain]["file"]
            log_message(f"Chain split fasta file: {chain_split_fasta_file}")

            chain_split_ali_file = os.path.join(chain_folder, "raw_fasta_record.seq")

            for record in SeqIO.parse(chain_split_fasta_file, "fasta"):
                description = record.description
                description_list = description.split("|")
                check_res = description_list[2].lstrip().split(" ")[0]

                ali_code = "DL" if check_res == "DNA" else "P1"

                first_line = f">{ali_code};{case_name}{chain}"
                second_line = "sequence:::::::::"
                third_line = textwrap.fill(str(record.seq), width=60)

                if append_end_in_seq:
                    third_line += "*"

                with open(chain_split_ali_file, "w") as out_file:
                    out_file.write(f"{first_line}\n{second_line}\n{third_line}\n")
                chain_ali_file_list.append(chain_split_ali_file)

            length_full_fasta = len(record.seq)

            # Use resolved paths directly
            missing_residues_file = chain_data[chain]["missing_residues_file"]
            original_residues_file = chain_data[chain]["original_residues_file"]

            missing_info = pd.read_csv(missing_residues_file)
            ori_info = pd.read_csv(original_residues_file)

            merge_resid_list = (
                missing_info["ssseq"].tolist() + ori_info["resid"].tolist()
            )
            full_resid_list = sorted(merge_resid_list)
            full_seq_pdb_list = []
            fill_seq_pdb_list = []

            for resid in full_resid_list:
                if resid in ori_info["resid"].values:
                    one_letter_name = ori_info[ori_info["resid"] == resid][
                        "resnameOne"
                    ].values[0]
                    full_seq_pdb_list.append(one_letter_name)
                    fill_seq_pdb_list.append(one_letter_name)
                elif resid in missing_info["ssseq"].values:
                    one_letter_name = missing_info[missing_info["ssseq"] == resid][
                        "oneLetter"
                    ].values[0]
                    full_seq_pdb_list.append(one_letter_name)
                    fill_seq_pdb_list.append("-")
                else:
                    raise Exception(f"ERROR, record not found for resid {resid}")

            str1 = "".join(full_seq_pdb_list)
            str2 = str(record.seq)
            if str1 == str2:
                compare_record_list.append("Y")
            else:
                log_message(f"Sequence mismatch for chain {chain}")
                log_message(f"Full seq from pdb   : {str1}")
                log_message(f"Full seq from fasta : {str2}")
                compare_record_list.append("N")

            if append_end_in_seq:
                full_seq_pdb_list.append("*")
                fill_seq_pdb_list.append("*")

            modeller_ali_filename = os.path.join(chain_folder, "homology.ali")
            first_line = f">{ali_code};{case_name}{chain}"
            f1, f2, f3, f4 = (
                "structure",
                "Merge",
                (
                    str(ori_info["resid"].tolist()[0])
                    if ori_info["resid"].tolist()
                    else ""
                ),
                chain,
            )
            f5 = f'+{len(ori_info["resid"].tolist())}'
            f6, f7, f8, f9, f10 = chain, "", "", "", ""
            second_line = ":".join([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
            third_line = textwrap.fill("".join(fill_seq_pdb_list), width=60)
            first_line_2nd = f">{ali_code};{case_name}{chain}_full"
            second_line_2nd = "sequence:::::::::"
            third_line_2nd = textwrap.fill("".join(full_seq_pdb_list), width=60)

            if len(third_line) != len(third_line_2nd):
                raise Exception("ERROR, sequence does not match")
            else:
                with open(modeller_ali_filename, "w") as out_file:
                    out_file.write(f"{first_line}\n{second_line}\n{third_line}\n\n")
                    out_file.write(
                        f"{first_line_2nd}\n{second_line_2nd}\n{third_line_2nd}\n"
                    )
                modeller_ali_file_list.append(modeller_ali_filename)

        result = {
            "seqAgreePdbFasta": "Y" if "N" not in compare_record_list else "N",
            "processSingleChainModeller": (
                "Y" if " " not in modeller_ali_file_list else "N"
            ),
            "chainAliFile": chain_ali_file_list,
            "modellerChainAliFile": modeller_ali_file_list,
            "seqCheck": compare_record_list,
        }

        return result
