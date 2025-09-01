import json  # noqa: E402
import os
from datetime import datetime

import pandas as pd
from Bio import SeqIO  # noqa: E402
from Bio.Data.PDBData import protein_letters_3to1  # noqa: E402
from Bio.PDB import PDBIO, PDBParser, Select  # noqa: E402
from Bio.SeqRecord import SeqRecord  # noqa: E402

from bocoflow_core.logger import log_message
from bocoflow_core.node import IONode, NodeException, NodeResult
from bocoflow_core.parameters import (
    BooleanParameter,
    FileParameterEdit,
    FolderParameter,
    StringParameter,
)


class LoadPDBFastaNode(IONode):
    """LoadPDBFastaNode

    This is an entity initialization node that loads and processes PDB and FASTA files
    for homology modeling preparation. It extracts structural and sequence information,
    organizing them into chain-specific data.

    [ Lib dependence ]
    - biopython: 1.84 (tested)
    - pandas: for data processing

    [ Function ]
    - accesses and processes PDB structure data
    - extracts missing residues from PDB header
    - generates chain-specific PDB files (with/without HETATM)
    - processes residue information (3-letter and 1-letter codes)
    - parses and splits FASTA sequences by chains
    - organizes data into chain-specific directories
    - supports both protein and DNA/RNA residues

    [ Input ]
    - Case name: identifier for the protein system
    - PDB file: structure file in PDB format
    - FASTA file: sequence file in FASTA format
    - Output directory: location for generated files
    - Check PDB header: option to process missing residues

    [ Output ]
    Files:
    - Chain-specific PDB files (with/without HETATM)
    - Missing residues CSV files: records structural gaps
    - Original residues CSV files: lists present residues
    - Chain-specific FASTA files
    - System summary JSON: complete processing results

    Data:
    - Chain list and chain-specific data
    - Residue information and mappings
    - FASTA sequence data by chain

    [ Note ]
    The node handles both protein and DNA/RNA residues, with special consideration
    for HETATM records and residue name conversions. Care should be taken with
    biopython version compatibility for residue type and name conversions.
    """

    name = "Load PDB and FASTA"
    num_in = 0
    num_out = 1

    OPTIONS = {
        "case_name": StringParameter("Case Name", docstring="Name of the case/protein"),
        "pdb_file": FileParameterEdit("PDB File", docstring="Path to PDB file"),
        "fasta_file": FileParameterEdit("FASTA File", docstring="Path to FASTA file"),
        "output_dir": FolderParameter(
            "Output Directory", docstring="Output directory for generated files"
        ),
        "check_pdb_header": BooleanParameter(
            "Check PDB Header",
            default=True,
            docstring="Whether to check PDB header for missing residues",
        ),
        "force_to_run": BooleanParameter(
            "Force to Run",
            default=False,
            docstring="If true, the node will be executed regardless of the database record",
        ),
    }

    def execute(self, predecessor_data, flow_vars):
        log_message(
            f"Starting execution of LoadPDBFastaNode for case: {flow_vars['case_name'].get_value()}"
        )
        try:
            # Initialize standard result
            result = NodeResult()
            result.metadata.update(
                {
                    "case_name": flow_vars["case_name"].get_value(),
                    "execution_time": datetime.now().isoformat(),
                }
            )

            # Get the parameters
            case_name = flow_vars["case_name"].get_value()
            pdb_file = flow_vars["pdb_file"].get_value()
            fasta_file = flow_vars["fasta_file"].get_value()
            output_dir = flow_vars["output_dir"].get_value()
            check_pdb_header = flow_vars["check_pdb_header"].get_value()

            # Resolve paths using the Node class's resolve_path method
            pdb_file = (
                self.resolve_path(pdb_file)
                if hasattr(self, "resolve_path")
                else pdb_file
            )
            fasta_file = (
                self.resolve_path(fasta_file)
                if hasattr(self, "resolve_path")
                else fasta_file
            )
            output_dir = (
                self.resolve_path(output_dir)
                if hasattr(self, "resolve_path")
                else output_dir
            )

            log_message(f"Resolved PDB file path: {pdb_file}")
            log_message(f"Resolved FASTA file path: {fasta_file}")
            log_message(f"Resolved output directory: {output_dir}")

            result.metadata.update({"output_dir": self.format_output_path(output_dir)})

            # Validate file paths
            for file_path, file_type in [(pdb_file, "PDB"), (fasta_file, "FASTA")]:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"{file_type} file not found: {file_path}")

            os.makedirs(output_dir, exist_ok=True)

            # Process PDB file
            pdb_chain_list, chain_data = self.process_pdb(
                case_name, pdb_file, output_dir, check_pdb_header
            )

            # Process FASTA file
            fasta_data = self.process_fasta(
                fasta_file, output_dir, case_name, pdb_chain_list
            )

            # Record input files with URI prefixes
            result.files["input"].update(
                {
                    "pdb_file": self.format_output_path(pdb_file),
                    "fasta_file": self.format_output_path(fasta_file),
                }
            )

            # Format all paths in chain_data with URI prefixes
            formatted_chain_data = {}
            for chain_id, data in chain_data.items():
                formatted_data = {}
                for key, path in data.items():
                    formatted_data[key] = self.format_output_path(path)
                formatted_chain_data[chain_id] = formatted_data

            # Format all paths in fasta_data with URI prefixes
            formatted_fasta_data = {}
            for chain_id, data in fasta_data.items():
                formatted_data = dict(data)
                if "file" in formatted_data:
                    formatted_data["file"] = self.format_output_path(
                        formatted_data["file"]
                    )
                formatted_fasta_data[chain_id] = formatted_data

            # Store main processing results and write to a json file
            # Use formatted paths in the data structure but raw paths for file operations
            main_data = {
                "pdb_chain_list": pdb_chain_list,
                "chain_data": formatted_chain_data,
                "fasta_data": formatted_fasta_data,
            }

            result.data.update(main_data)

            # Create summary JSON with the raw paths for actual file writing
            summary_json_path = os.path.join(output_dir, "system_summary.json")
            with open(summary_json_path, "w", encoding="utf-8") as json_file:
                json_file.write(json.dumps(main_data))

            # Add the summary JSON file to the output files with URI prefix
            result.files["output"].update(
                {"system_summary": self.format_output_path(summary_json_path)}
            )

            # Update data with just essential information
            result.data.update(
                {
                    "case_name": case_name,
                    "output_dir": self.format_output_path(output_dir),
                    "pdb_chain_list": pdb_chain_list,
                    "chain_data": chain_data,
                    "working_path": self.format_output_path(output_dir),
                }
            )

            result.success = True
            result.message = "Successfully processed PDB and FASTA files"

            return result.to_json()

        except Exception as e:
            log_message(f"Error in LoadPDBFastaNode: {str(e)}")
            raise NodeException("load pdb and fasta", str(e))

    def process_pdb(self, case_name, pdb_file, output_dir, check_pdb_header):
        log_message("in processing pdb")

        class NonHetSelect(Select):
            def accept_residue(self, residue):
                return 1 if residue.id[0] == " " else 0

        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(case_name, pdb_file)
        io = PDBIO()

        if check_pdb_header and structure.header["has_missing_residues"]:
            missing_df = pd.DataFrame(structure.header["missing_residues"])
        else:
            missing_df = pd.DataFrame(
                columns=[
                    "model",
                    "res_name",
                    "chain",
                    "ssseq",
                    "insertion",
                    "oneLetter",
                ]
            )

        chain_list = []
        chain_data = {}
        for model in structure:
            for chain in model:
                chain_id = chain.id
                chain_list.append(chain_id)
                chain_dir = os.path.join(output_dir, chain_id)
                os.makedirs(chain_dir, exist_ok=True)

                # Process missing residues info
                chain_missing = missing_df[missing_df["chain"] == chain_id]
                # add one-letter code for missing residues
                miss_resname = chain_missing.res_name.values.tolist()
                one_letter_resname = []
                for residue in miss_resname:
                    one_letter_resname.append(protein_letters_3to1[residue])
                chain_missing["oneLetter"] = one_letter_resname
                # Save missing residues info to a csv file
                missing_file = os.path.join(chain_dir, "missing_residues.csv")
                chain_missing.to_csv(missing_file, index=False)

                # Process original residues info
                three_letter_ori_resname_list = []
                one_letter_ori_resname_list = []
                ori_resid_list = []
                for residue in chain.get_residues():
                    hetflag = residue.get_full_id()[3][0]
                    resname = residue.get_resname()
                    resid = residue.get_full_id()[3][1]
                    if not (hetflag == " " or hetflag == ""):  # skip HETATM
                        pass
                    elif len(resname.strip()) == 2:  # DNA/(RNA?)
                        three_letter_ori_resname_list.append(resname)
                        one_letter_ori_resname_list.append(resname.strip()[1])
                        ori_resid_list.append(resid)
                    else:  # Protein
                        three_letter_ori_resname_list.append(resname)
                        one_letter_ori_resname_list.append(
                            protein_letters_3to1[resname]
                        )
                        ori_resid_list.append(resid)

                ori_data = {
                    "resnameThree": three_letter_ori_resname_list,
                    "resnameOne": one_letter_ori_resname_list,
                    "resid": ori_resid_list,
                    "chain": [chain_id] * len(ori_resid_list),
                }

                # save original residues info to a csv file
                ori_data_df = pd.DataFrame(ori_data)
                ori_data_file = os.path.join(chain_dir, "original_residues.csv")
                ori_data_df.to_csv(ori_data_file, index=False)

                # Save PDB files for the chain
                io.set_structure(chain)
                with_het_file = os.path.join(chain_dir, f"withHET{chain_id}.pdb")
                no_het_file = os.path.join(chain_dir, f"noHET{chain_id}.pdb")
                io.save(with_het_file)
                io.save(no_het_file, NonHetSelect())

                chain_data[chain_id] = {
                    "missing_residues_file": missing_file,
                    "original_residues_file": ori_data_file,
                    "with_het_file": with_het_file,
                    "no_het_file": no_het_file,
                }
        return chain_list, chain_data

    def process_fasta(
        self, fasta_file, output_dir, case_name=None, pdb_chain_list=None
    ):
        fasta_refID_list = []
        fasta_chain_list = []
        split_fasta_file_list = []

        fasta_data = {}
        for record in SeqIO.parse(fasta_file, "fasta"):
            description = record.description
            desc_parts = description.split("|")
            ref_id = desc_parts[0].split("_")[1]

            chains = desc_parts[1].replace(",", " ").split(" ")
            del chains[0]
            length = len(record.seq)

            for chain in chains:
                chain_id = f"{case_name}_{ref_id}|chain {chain}|"
                chain_desc = desc_parts[2]
                chain_record = SeqRecord(
                    record._seq, id=chain_id, description=chain_desc
                )
                chain_dir = os.path.join(output_dir, chain)
                os.makedirs(chain_dir, exist_ok=True)
                out_file = os.path.join(chain_dir, "raw_fasta_record.faa")
                split_fasta_file_list.append(out_file)
                SeqIO.write(chain_record, out_file, "fasta")
                fasta_refID_list.append(ref_id)
                fasta_chain_list.append(chain)

        fasta_summary = dict(zip(fasta_chain_list, fasta_refID_list))
        fasta_summary2 = dict(zip(fasta_chain_list, split_fasta_file_list))
        refID_list = []
        fasta_list = []
        for chain in pdb_chain_list:
            refID = fasta_summary[chain]
            refID_list.append(refID)
            file = fasta_summary2[chain]
            fasta_list.append(file)

            fasta_data[chain] = {"refID": refID, "file": file}

        return fasta_data
