import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

from bocoflow_core.logger import log_message
from bocoflow_core.node import Node, NodeException, NodeResult
from bocoflow_core.parameters import *


class FixPkaGmxEM(Node):
    """FixPkaGmxEM

    This node performs pKa prediction and GROMACS energy minimization within a Docker container,
    handling protonation states and structure optimization.

    [ Lib dependence ]
    - Docker with gromacs_py_em image
    - pdb2pqr: 3.6.1
    - propka: 3.5.1
    - gromacs_py: 2.0.3
    - numpy: 1.24.3
    - scipy: 1.10.1
    - pandas: 2.2.3
    - seaborn: 0.13.2

    [ Function ]
    - predicts protonation states using pdb2pqr and propka
    - performs GROMACS energy minimization
    - handles force field selection and setup
    - supports multiple force fields (amber99sb-ildn, amber99sb)
    - generates energy minimization plots
    - manages Docker container execution
    - handles file transfers between host and container
    - monitors energy minimization progress
    - calculates maximum forces

    [ Input ]
    - Case name (optional, can use predecessor data)
    - Input PDB file
    - Working folder path
    - Force field selection
    - Energy minimization steps
    - Docker image name
    - Force run option

    [ Output ]
    Files:
    - Energy minimized structure (.gro)
    - Topology file (.top)
    - Energy minimization plot (.png)
    - PDB2PQR and minimization logs

    Data:
    - Maximum force after minimization
    - Processing status
    - Energy minimization results
    - Working directory path

    [ Note ]
    The node requires Docker to be installed and running. The Docker daemon must be
    started before running this node. The required Docker image (gromacs_py_em) will
    be built automatically on first use if it doesn't exist.

    All operations are containerized for reproducibility and dependency management.

    """

    # Basic configuration
    name = "FixPkaGmxEM"
    node_key = "FixPkaGmxEM"
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
            docstring="Input PDB file to be processed",
            optional=False,
        ),
        "relative_work_folder": StringParameter(
            label="Working folder relative to the input structure file",
            docstring="Working folder relative to the input structure file",
            optional=False,
        ),
        "force_field": SelectParameter(
            "Force Field",
            options=["amber99sb-ildn", "amber99sb"],
            default="amber99sb",
            docstring="Force field type",
        ),
        "em_steps": IntegerParameter(
            "EM Steps", default=1000, docstring="EM Steps", optional=True
        ),
        "docker_image": StringParameter(
            label="Docker Image",
            default="gromacs_py_em:latest",
            docstring="Docker image name (default will be built if not present)",
            optional=True,
        ),
        "force_to_run": BooleanParameter(
            "Force to Run",
            default=False,
            docstring="If true, the node will be executed regardless of the database record",
            optional=True,
        ),
    }

    def check_docker(self):
        """Check if Docker is available and running"""
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            raise NodeException("execute", "Docker is not running or accessible")
        except FileNotFoundError:
            raise NodeException("execute", "Docker is not installed")

    def ensure_docker_image(self, image_name):
        """Ensure the Docker image exists, build if necessary"""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", image_name],
                check=True,
                capture_output=True,
                text=True,
            )

            if not result.stdout.strip():
                log_message(f"Building Docker image {image_name}...")

                with tempfile.TemporaryDirectory() as tmp_dir:
                    dockerfile_content = """
                    FROM continuumio/miniconda3

                    # Install git and system dependencies
                    RUN apt-get update && apt-get install -y git && apt-get clean

                    # Create conda environment with specific Python version
                    RUN conda create -n gmx_env python=3.9 -y

                    # Configure shell to use conda environment
                    SHELL ["conda", "run", "-n", "gmx_env", "/bin/bash", "-c"]

                    # Install packages with specific versions
                    RUN conda install -c conda-forge numpy=1.24.3 -y && \
                        conda install -c conda-forge scipy=1.10.1 -y && \
                        conda install -c conda-forge -c bioconda gromacs_py=2.0.3 -y && \
                        conda install -c conda-forge propka=3.5.1 -y && \
                        conda install -c conda-forge pdb2pqr=3.6.1 -y && \
                        conda install -c conda-forge pandas=2.2.3 -y && \
                        conda install -c conda-forge seaborn=0.13.2 -y && \
                        conda clean -afy

                    # Create and set working directory
                    WORKDIR /workspace

                    # Copy entrypoint script
                    COPY entrypoint.py /workspace/

                    # Set entrypoint
                    ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "gmx_env", "python", "/workspace/entrypoint.py"]
                    """

                    entrypoint_content = """
                    import sys
                    import json
                    import traceback
                    import subprocess
                    import os
                    from gromacs_py import gmx

                    import pandas as pd
                    import matplotlib
                    import matplotlib.pyplot as plt
                    import numpy as np
                    import seaborn as sns


                    def search_string_in_file(file_name, string_to_search):
                        # Search for the given string in file and return lines containing that string,
                        # along with line numbers
                        line_number = 0
                        list_of_results = []
                        with open(file_name, 'r') as read_obj:
                            for line in read_obj:
                                line_number += 1
                                if string_to_search in line:
                                    list_of_results.append((line_number, line.rstrip()))
                        return list_of_results

                    def run_pdb2pqr(structure, case_name, force_field_name):
                        try:
                            # Determine force field for pdb2pqr
                            if force_field_name.startswith('amber'):
                                pdb2pqr_ff = 'AMBER'
                            else:
                                pdb2pqr_ff = 'CHARMM'

                            # Use /output directory for saving files (this is mounted to working_path)
                            output_dir = "/output"

                            # Define output filenames with full paths
                            output_pqr = os.path.join(output_dir, f"00_{case_name}.pqr")
                            output_log = os.path.join(output_dir, f"00_{case_name}_pdb2pqr.log")

                            # Change to output directory
                            os.chdir(output_dir)

                            # Construct pdb2pqr command with full logging
                            cmd = [
                                "pdb2pqr",
                                "--ff", pdb2pqr_ff,
                                "--ffout", pdb2pqr_ff,
                                "--keep-chain",
                                "--titration-state-method=propka",
                                "--with-ph=7.00",
                                "--log-level=INFO",
                                "--include-header",
                                structure,  # Input structure from /input
                                output_pqr  # Full path to output PQR
                            ]

                            print(f"Running pdb2pqr command: {' '.join(cmd)}", file=sys.stderr)

                            # Execute pdb2pqr and capture all output
                            with open(output_log, 'w') as log_file:
                                # Write command to log file
                                log_file.write(f"Command: {' '.join(cmd)}\\n\\n")

                                # Execute command and capture output
                                result = subprocess.run(
                                    cmd,
                                    check=True,
                                    capture_output=True,
                                    text=True
                                )

                                # Write stdout and stderr to log file
                                log_file.write("STDOUT:\\n")
                                log_file.write(result.stdout)
                                log_file.write("\\nSTDERR:\\n")
                                log_file.write(result.stderr)

                            # Print output to Docker logs as well
                            print(f"pdb2pqr stdout: {result.stdout}", file=sys.stderr)
                            if result.stderr:
                                print(f"pdb2pqr stderr: {result.stderr}", file=sys.stderr)

                            # Return relative paths for use in response
                            return {
                                "pqr_file": f"00_{case_name}.pqr",
                                "log_file": f"00_{case_name}_pdb2pqr.log"
                            }

                        except subprocess.CalledProcessError as e:
                            error_msg = f"pdb2pqr execution failed: {str(e)}\\nOutput: {e.stdout}\\nError: {e.stderr}"
                            print(error_msg, file=sys.stderr)
                            raise Exception(error_msg)

                    def fix_pka_gmx_em(case_name, structure, force_field_name, em_steps, working_path):
                        try:
                            print(f"Starting processing for {case_name}", file=sys.stderr)

                            # Run pdb2pqr
                            pdb2pqr_results = run_pdb2pqr(structure, case_name, force_field_name)
                            # print(f"PDB2PQR processing complete. Output files: {pdb2pqr_results}", file=sys.stderr)

                            # # Use the generated PQR file path for GROMACS
                            # # input_pqr_path = os.path.join("/output", pdb2pqr_results["pqr_file"])

                            # Initialize GROMACS system with PQR file
                            md_sys = gmx.GmxSys(name=case_name, coor_file=structure)

                            # Hope the pdb2pqr output can avoid being calucated again in the gromacs_py
                            # Use /output directory for saving files (this is mounted to working_path)
                            output_dir = "/output"
                            md_sys.prepare_top(out_folder=output_dir, ff=force_field_name)

                            # EM part
                            md_sys.create_box(dist=2.0, box_type="triclinic",
                                                      check_file_out=True)
                            md_sys.em_2_steps(out_folder=output_dir, no_constr_nsteps=em_steps, constr_nsteps=em_steps,
                                              posres="", create_box_flag=False)

                            # extract result
                            ener_pd_1 = md_sys.sys_history[-1].get_ener(selection_list=['Potential'])
                            ener_pd_2 = md_sys.get_ener(selection_list=['Potential'])
                            ener_pd_1['label'] = 'no bond constr'
                            ener_pd_2['label'] = 'bond constr'
                            ener_pd = pd.concat([ener_pd_1, ener_pd_2])
                            # ener_pd['time'] = np.arange(len(ener_pd))

                            ax = sns.lineplot(x='Time (ps)', y="Potential",
                                        hue="label",
                                        data=ener_pd)
                            ax.set_xlabel('step')
                            ax.set_ylabel('energy (KJ/mol)')
                            # save the plot
                            plt.savefig(os.path.join(output_dir, f"{case_name}_ener.png"))


                            check_log_file = os.path.join(output_dir, f"{case_name}.log")
                            search_string = 'Maximum force'
                            target_line = search_string_in_file(check_log_file, search_string)
                            target_line = target_line[0][1].split()
                            em_max_force = float(target_line[3])
                            em_gro = f"{case_name}.gro"
                            em_top = f"{case_name}_pdb2gmx.top"

                            result = {
                                "case_name": case_name,
                                "em_gro":  em_gro,
                                "em_top":  em_top,
                                "em_max_force": em_max_force,
                                "em_energy_plot": f"{case_name}_ener.png",
                            }

                            # Check if files exist in the output directory
                            em_gro_exists = os.path.exists(os.path.join(output_dir, em_gro))
                            if em_gro_exists:
                                result["sucess"]= True
                                result["message"] = "Successfully completed pdb2pqr and GROMACS setup"
                            else:
                                result["sucess"]= False
                                result["message"] = "Failed in completing pdb2pqr and GROMACS setup"

                            print(f"Operation completed successfully", file=sys.stderr)
                            print(json.dumps(result), flush=True)
                            return result

                        except Exception as e:
                            error_msg = f"Error in fix_pka_gmx_em: {str(e)}"
                            print(error_msg, file=sys.stderr)
                            print(traceback.format_exc(), file=sys.stderr)
                            error_result = {
                                "success": False,
                                "error": error_msg,
                                "traceback": traceback.format_exc()
                            }
                            print(json.dumps(error_result), flush=True)
                            return error_result

                    if __name__ == "__main__":
                        try:
                            print("Starting entrypoint execution", file=sys.stderr)

                            if len(sys.argv) < 2:
                                raise ValueError("No parameters provided")

                            params = json.loads(sys.argv[1])
                            print(f"Received parameters: {json.dumps(params, indent=2)}", file=sys.stderr)

                            fix_pka_gmx_em(**params)

                        except Exception as e:
                            error_msg = f"Failed to execute with parameters: {str(e)}"
                            print(error_msg, file=sys.stderr)
                            print(traceback.format_exc(), file=sys.stderr)
                            error_result = {
                                "success": False,
                                "error": error_msg,
                                "traceback": traceback.format_exc()
                            }
                            print(json.dumps(error_result), flush=True)
                    """

                    with open(os.path.join(tmp_dir, "Dockerfile"), "w") as f:
                        f.write(textwrap.dedent(dockerfile_content))

                    with open(os.path.join(tmp_dir, "entrypoint.py"), "w") as f:
                        f.write(textwrap.dedent(entrypoint_content))

                    subprocess.run(
                        ["docker", "build", "-t", image_name, tmp_dir],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

                    log_message(f"Docker image {image_name} built successfully")

            return True
        except subprocess.CalledProcessError as e:
            raise NodeException(
                "execute",
                f"Failed to build Docker image: {str(e)}\nOutput: {e.stdout}\nError: {e.stderr}",
            )

    def execute(self, predecessor_data, flow_vars):
        try:
            # Initialize standard result
            result = NodeResult()
            result.metadata.update({"execution_time": datetime.now().isoformat()})

            # Check Docker availability
            self.check_docker()

            # Collect input data
            input_data = predecessor_data[0] if predecessor_data else {}
            case_name = flow_vars["case_name"].get_value() or input_data.get(
                "case_name"
            )

            # Get parameters and resolve paths
            input_file = flow_vars["input_pdb_file"].get_value()
            resolved_input_file = (
                self.resolve_path(input_file)
                if hasattr(self, "resolve_path")
                else input_file
            )

            force_field = flow_vars["force_field"].get_value()
            relative_work_folder = flow_vars["relative_work_folder"].get_value()
            em_steps = flow_vars["em_steps"].get_value()
            docker_image = flow_vars["docker_image"].get_value()

            # Log the starting configuration
            log_message(f"Starting {self.node_key} execution for case: {case_name}")
            log_message(
                f"Input configuration: force_field={force_field}, em_steps={em_steps}"
            )
            log_message(f"Resolved input file: {resolved_input_file}")

            # Ensure paths are absolute
            input_path = os.path.dirname(os.path.abspath(resolved_input_file))
            working_path = os.path.abspath(
                os.path.join(input_path, relative_work_folder)
            )
            os.makedirs(working_path, exist_ok=True)
            log_message(f"Working directory created/verified: {working_path}")

            # Store input files in result with URI prefixes
            result.files["input"].update(
                {
                    "input_pdb_file": (
                        self.format_output_path(resolved_input_file)
                        if hasattr(self, "format_output_path")
                        else resolved_input_file
                    )
                }
            )

            # Ensure Docker image exists
            self.ensure_docker_image(docker_image)

            # Prepare parameters for Docker execution
            params = {
                "case_name": case_name,
                "structure": f"/input/{os.path.basename(resolved_input_file)}",
                "force_field_name": force_field,
                "em_steps": em_steps,
                "working_path": "/output",
            }

            # Run Docker container
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{os.path.dirname(resolved_input_file)}:/input",
                "-v",
                f"{working_path}:/output",
                docker_image,
                json.dumps(params),
            ]

            log_message("Executing Docker container...")

            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)

                # Log Docker output to project log file
                if result.stdout:
                    log_message("Docker stdout:", "info")
                    for line in result.stdout.split("\n"):
                        if line.strip():
                            log_message(f"  {line}", "info")

                if result.stderr:
                    log_message("Docker stderr:", "info")
                    for line in result.stderr.split("\n"):
                        if line.strip():
                            log_message(f"  {line}", "info")

                # Parse only the last line of stdout which should contain our JSON result
                output_lines = result.stdout.strip().split("\n")
                docker_output = json.loads(output_lines[-1])

                # Verify required fields are present
                required_fields = [
                    "case_name",
                    "em_gro",
                    "em_top",
                    "em_max_force",
                    "em_energy_plot",
                ]
                for field in required_fields:
                    if field not in docker_output:
                        raise ValueError(
                            f"Missing required field in Docker output: {field}"
                        )

                # Format file paths with URI prefixes
                output_files = {
                    "em_gro": os.path.join(working_path, docker_output["em_gro"]),
                    "em_top": os.path.join(working_path, docker_output["em_top"]),
                    "em_energy_plot": os.path.join(
                        working_path, docker_output["em_energy_plot"]
                    ),
                }

                # Format all output paths
                formatted_output_files = {}
                for key, path in output_files.items():
                    formatted_output_files[key] = (
                        self.format_output_path(path)
                        if hasattr(self, "format_output_path")
                        else path
                    )

                # Add working path to result with URI prefix
                formatted_working_path = (
                    self.format_output_path(working_path)
                    if hasattr(self, "format_output_path")
                    else working_path
                )

                # Create the final result object
                node_result = NodeResult()
                node_result.metadata.update(
                    {
                        "case_name": case_name,
                        "execution_time": datetime.now().isoformat(),
                        "working_path": formatted_working_path,
                    }
                )

                # Store data with formatted paths
                node_result.data.update(
                    {
                        "case_name": case_name,
                        "em_max_force": docker_output["em_max_force"],
                        "success": docker_output.get("sucess", False),
                        "message": docker_output.get("message", ""),
                        "working_path": formatted_working_path,
                    }
                )

                # Store output files with URI prefixes
                node_result.files["output"].update(formatted_output_files)

                # Update essential data with key information only
                node_result.data.update(
                    {
                        "case_name": case_name,
                        "working_path": formatted_working_path,
                        "output_gro_file": formatted_output_files["em_gro"],
                        "output_top_file": formatted_output_files["em_top"],
                        "force_field": force_field,
                    }
                )

                # Don't transfer everything from predecessor's data
                # Only keep truly essential data not already covered
                if predecessor_data:
                    for key in ["pdb_chain_list", "merge_folder"]:
                        if (
                            key in input_data.get("data", {})
                            and key not in node_result.data
                        ):
                            node_result.data[key] = input_data["data"][key]

                node_result.success = True
                node_result.message = f"Successfully completed {self.node_key} execution for case: {case_name}"

                log_message(
                    f"Successfully completed {self.node_key} execution for case: {case_name}"
                )
                return node_result.to_json()

            except subprocess.CalledProcessError as e:
                error_msg = f"Docker execution failed with code {e.returncode}"
                log_message(error_msg, "error")
                if e.stdout:
                    log_message("Docker stdout from error:", "error")
                    log_message(e.stdout, "error")
                if e.stderr:
                    log_message("Docker stderr from error:", "error")
                    log_message(e.stderr, "error")
                raise NodeException("execute", error_msg)

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse Docker output: {str(e)}"
                log_message(error_msg, "error")
                raise NodeException("execute", error_msg)

        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            log_message(error_msg, "error")
            if isinstance(e, NodeException):
                raise
            raise NodeException("execute", error_msg)
