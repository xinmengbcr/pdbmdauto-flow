import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from bocoflow_core.logger import log_message
from bocoflow_core.node import Node, NodeException
from bocoflow_core.parameters import *


class ClusterPBC(Node):
    """ClusterPBC
    This Node removes the PBC and clusters chains using AgglomerativeClustering within a Docker container.
    """

    name = "ClusterPBC"
    node_key = "ClusterPBC"
    color = "green"
    num_in = 1
    num_out = 1

    OPTIONS = {
        "case_name": StringParameter(
            label="Case Name",
            default="",
            docstring="Name of the case/protein (leave empty to use predecessor data)",
            optional=True,
        ),
        "input_gro_file": FileParameterEdit(
            label="Input Gro File",
            docstring="Input Gro file to be processed",
            optional=False,
        ),
        "output_gro_suffix": StringParameter(
            label="Output gro Filename",
            default="_pbc_removed",
            docstring="Output Gro file suffix, will be saved in the same folder as the input file",
            optional=True,
        ),
        "docker_image": StringParameter(
            label="Docker Image",
            default="cluster_pbc:latest",
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
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            raise NodeException("execute", "Docker is not running or accessible")
        except FileNotFoundError:
            raise NodeException("execute", "Docker is not installed")

    def ensure_docker_image(self, image_name):
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
                    # Copy the entrypoint script to the temp directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    entrypoint_src = os.path.join(
                        current_dir, "docker_entry_remove_pbc_cluster.py"
                    )
                    entrypoint_dst = os.path.join(tmp_dir, "entrypoint.py")
                    shutil.copy2(entrypoint_src, entrypoint_dst)

                    dockerfile_content = """
                    FROM python:3.9-slim

                    RUN pip install numpy pandas scikit-learn

                    WORKDIR /workspace
                    COPY entrypoint.py /workspace/

                    ENTRYPOINT ["python", "/workspace/entrypoint.py"]
                    """

                    with open(os.path.join(tmp_dir, "Dockerfile"), "w") as f:
                        f.write(textwrap.dedent(dockerfile_content))

                    subprocess.run(
                        ["docker", "build", "-t", image_name, tmp_dir],
                        check=True,
                        capture_output=True,
                        text=True,
                    )

            return True
        except subprocess.CalledProcessError as e:
            raise NodeException(
                "execute",
                f"Failed to build Docker image: {str(e)}\nOutput: {e.stdout}\nError: {e.stderr}",
            )

    def execute(self, predecessor_data, flow_vars):
        try:
            self.check_docker()

            input_data = predecessor_data[0]
            case_name = flow_vars["case_name"].get_value() or input_data.get(
                "case_name"
            )
            input_file = flow_vars["input_gro_file"].get_value()
            output_suffix = flow_vars["output_gro_suffix"].get_value()
            docker_image = flow_vars["docker_image"].get_value()

            log_message(f"Starting {self.node_key} execution for case: {case_name}")

            input_path = os.path.dirname(os.path.abspath(input_file))
            output_filename = Path(input_file).stem + output_suffix + ".gro"
            output_file = os.path.join(input_path, output_filename)

            self.ensure_docker_image(docker_image)

            params = {
                "input_file": f"/input/{os.path.basename(input_file)}",
                "output_file": f"/output/{output_filename}",
            }

            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{input_path}:/input",
                "-v",
                f"{input_path}:/output",
                docker_image,
                json.dumps(params),
            ]
            print(f"Command: {' '.join(cmd)}")
            # Add stderr output
            result = subprocess.run(
                cmd,
                check=False,  # Don't raise CalledProcessError
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"Docker stderr: {result.stderr}")
                raise NodeException(
                    "execute",
                    f"Docker container failed with exit code {result.returncode}: {result.stderr}",
                )

            print(f"Docker stdout: {result.stdout}")
            print(f"Docker stderr: {result.stderr}")

            try:
                stdout_stripped = result.stdout.strip()
                print(f"Stripped stdout: {stdout_stripped}")

                if not stdout_stripped:
                    raise NodeException(
                        "execute", "Docker container produced no output"
                    )

                docker_output = json.loads(stdout_stripped)
                if docker_output is None:
                    raise NodeException(
                        "execute", "Docker container returned null output"
                    )

                print(f"Parsed docker output: {docker_output}")

                if not docker_output.get("success", False):
                    error_msg = docker_output.get("error", "Unknown error")
                    print(f"Error message from docker: {error_msg}")
                    raise NodeException("execute", error_msg)

                return json.dumps(
                    {
                        "case_name": case_name,
                        "output_gro": output_filename,
                        "success": True,
                        "message": "Successfully removed PBC and clustered chains",
                    }
                )
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")
                raise NodeException(
                    "execute", f"Failed to parse Docker output: {str(e)}"
                )

        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            log_message(error_msg, "error")
            if isinstance(e, NodeException):
                raise
            raise NodeException("execute", error_msg)
