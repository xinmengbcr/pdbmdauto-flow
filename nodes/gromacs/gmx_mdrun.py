# gmx_mdrun.py
import getpass
import glob
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import paramiko
from fabric import Connection
from paramiko import Agent

from bocoflow_core.logger import log_message
from bocoflow_core.node import Node, NodeException
from bocoflow_core.parameters import *
from bocoflow_core.workflow import WorkflowContext

# conda activate mdflow
# conda install conda-forge::paramiko #paramiko-3.5.0
# pip install  fabric # fabric2-3.2.2-py3-none-any.whl (114 kB)


class GmxMdRun(Node):
    """GmxMdRun
    This Node executes GROMACS molecular dynamics simulations with support for both
    local and remote (cluster) execution. It handles the complete MD workflow:

    1. Preprocessing with 'gmx grompp' to generate the TPR file
    2. Running the simulation with 'gmx mdrun'
    3. Managing file transfers and job monitoring for remote execution

    [ Parameters ]
    - case_name: Name identifier for the simulation case (optional)
    - run_label: Label for the MD simulation (e.g., md, nvt, md_nvt)
    - use_remote: Toggle for remote cluster execution
    - force_run_remote: Force resubmission of remote job regardless of previous status
    - cluster_config: JSON configuration for remote cluster settings
    - force_to_run: Force execution regardless of database record

    [ Input Files ]
    - TOP file (.top): Topology file containing molecular structure and force field
    - GRO file (.gro): Structure file in GROMACS format
    - MDP file (.mdp): Parameters file for molecular dynamics
    - NDX file (.ndx): Index file defining atom groups
    - ITP files (*.itp): Include topology files (automatically included)

    [ Output Files ]
    - TPR file (.tpr): Portable binary run input file
    - Trajectory file (.xtc): Compressed trajectory
    - Energy file (.edr): Energy data
    - Structure file (.gro): Final coordinates
    - Log file (.log): Simulation log

    [ Remote Execution ]
    When running on a remote cluster (use_remote=True), requires:
    - SSH key authentication setup
    - Slurm job scheduler on remote system
    - Proper cluster_config with:
        - host: Cluster hostname
        - username: SSH username
        - ssh_key_file: Path to SSH private key
        - ssh_key_passphrase: Key passphrase (if applicable)
        - remote_workdir: Working directory on cluster
        - partition: Slurm partition
        - nodes: Number of nodes
        - ntasks-per-node: Tasks per node
        - walltime: Job time limit
        - account: Slurm account (if required)

    [ Dependencies ]
    Requires:
    - GROMACS installation (local or on cluster)
    - For remote execution:
        conda install conda-forge::paramiko
        pip install fabric
    """

    name = "GmxMdRun"
    node_key = "GmxMdRun"
    color = "purple"
    num_in = 1
    num_out = 1

    OPTIONS = {
        "case_name": StringParameter(
            label="Case Name",
            default="",
            docstring="Name of the case/protein (leave empty to use predecessor data)",
            optional=True,
        ),
        "run_label": StringParameter(
            label="MD Simulation Label",
            default="md",
            docstring="Label for the md run (e.g. md, nvt, md_nvt, md_npt)",
            optional=False,
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
        "use_remote": BooleanParameter(
            label="Run on Remote Cluster",
            default=False,
            docstring="Execute job on remote cluster",
        ),
        "force_run_remote": BooleanParameter(
            label="Force Remote Run",
            default=False,
            docstring="Force resubmission of remote job regardless of previous status",
            optional=True,
        ),
        "cluster_config": StringParameter(
            label="Cluster Configuration",
            default="",
            docstring="JSON string containing cluster configuration",
            optional=True,
        ),
        "force_to_run": BooleanParameter(
            label="Force to Run",
            default=False,
            docstring="Execute regardless of database record",
            optional=True,
        ),
    }

    def execute(self, predecessor_data, flow_vars):
        try:
            input_data = predecessor_data[0]
            case_name = flow_vars["case_name"].get_value() or input_data.get(
                "case_name"
            )
            run_label = flow_vars["run_label"].get_value()
            input_top_file = flow_vars["input_top_file"].get_value()
            input_gro_file = flow_vars["input_gro_file"].get_value()
            input_mdp_file = flow_vars["input_mdp_file"].get_value()
            input_ndx_file = flow_vars["input_ndx_file"].get_value()
            working_path = os.path.dirname(input_gro_file)
            use_remote = flow_vars["use_remote"].get_value()

            if not use_remote:
                return json.dumps(
                    self.run_local(
                        run_label,
                        input_top_file,
                        input_gro_file,
                        input_mdp_file,
                        input_ndx_file,
                        working_path,
                    )
                )
            else:
                cluster_config = flow_vars["cluster_config"].get_value()
                force_run_remote = flow_vars["force_run_remote"].get_value()

                # Generate output file path based on node ID
                # print(f"------local node Working path: {working_path}")
                # print(f" global working path === project path:  {WorkflowContext.get_instance().working_path}")
                project_path = WorkflowContext.get_instance().working_path
                output_file = os.path.join(
                    project_path,
                    f"{WorkflowContext.get_instance().name}-{self.node_id}",
                )
                # Read previous status if exists
                prev_result = None
                try:  # check if output file exists
                    with open(output_file, "r") as f:
                        prev_result = json.load(f)
                except FileNotFoundError:
                    pass

                log_message(f"------Output file: {output_file}", level="info")
                log_message(f"------Previous result: {prev_result}", level="info")

                result = self.run_remote(
                    run_label,
                    input_top_file,
                    input_gro_file,
                    input_mdp_file,
                    input_ndx_file,
                    working_path,
                    cluster_config,
                    force_run_remote,
                    prev_result,
                )
                return json.dumps(result)

        except Exception as e:
            log_message(f"Error in {self.node_key}: {str(e)}", level="error")
            return json.dumps({"success": False, "error": str(e)})

    def run_remote(
        self,
        run_label,
        top_file,
        gro_file,
        mdp_file,
        ndx_file,
        working_path,
        cluster_config,
        force_run_remote,
        prev_result,
    ):
        # if not force_run_remote and prev_result and prev_result.get('remote_job_status'):
        #     print(
        #         f"Previous job status: {prev_result['remote_job_status']}")
        #     # should do status checking and return status.
        #     return json.dumps({'status': 'success', 'message': 'Job already submitted'})
        # add more logic handling

        # https://claude.ai/chat/547097ea-6687-4819-bed3-660142c269b7
        # https://github.com/xinmengbcr/BoCoFlowBeta/issues/35#issue-2723185443

        result = prev_result
        # remove message
        result.pop("message", None)

        log_message(f"------Result: {result}", level="info")
        log_message(f"------prev_result: {prev_result}", level="info")
        remote_dir = cluster_config["remote_workdir"]
        log_message(f"------remote_dir: {remote_dir}", level="info")
        # First step: checking ssh connection first, does not matter what's the pre conditions
        try:
            ssh = self._get_ssh_connection(cluster_config)
            log_message(f"SSH connection established successfully", level="info")
        except Exception as e:
            log_message(
                f"Error in test ssh connection: {self.node_key}: {str(e)}",
                level="error",
            )
            result["success"] = False
            result["message"] = str(e)
            return result

        # step check sftp connection
        try:
            # Create SFTP client
            sftp = ssh.open_sftp()
            log_message(f"SFTP client created.... {sftp}")
        except Exception as e:
            log_message(
                f"Error in create sftp client: {self.node_key}: {str(e)}", level="error"
            )
            result["success"] = False
            result["message"] = str(e)
            return result

        # Second step: whether this is an old and not reinforced job, need to check the status of the job
        if prev_result.get("remote_job_status") and not force_run_remote:
            # check the status of the job
            job_id = prev_result.get("remote_job_id")
            log_message(f"------Job ID: {job_id}", level="info")
            prev_job_status = prev_result.get("remote_job_status")
            log_message(f"------Prev job status: {prev_job_status}", level="info")

            # Check current status
            try:
                stdin, stdout, stderr = ssh.exec_command(
                    f"sacct -j {job_id} -o State -n -P"
                )
                status = stdout.read().decode().strip().split("\n")[0]
                log_message(f"Previous job status: {prev_job_status}", level="info")
                log_message(f"Current job status: {status}", level="info")
            except Exception as e:
                log_message(
                    f"Error in checking job status: {self.node_key}: {str(e)}",
                    level="error",
                )
                return {
                    "success": False,
                    "remote_job_id": job_id,
                    "remote_job_status": prev_job_status,
                    "message": str(e),
                }

            if status == "RUNNING":
                return {
                    "remote_job_status": "running",
                    "remote_job_id": job_id,
                    "message": "Job is still running",
                    "success": True,
                }
            elif status == "COMPLETED":
                # Transfer results back
                output_files = [
                    f"{run_label}.{ext}" for ext in ["tpr", "gro", "edr", "log", "xtc"]
                ]
                warning = ""
                for file_name in output_files:
                    try:
                        remote_path = os.path.join(remote_dir, file_name)
                        local_path = os.path.join(working_path, file_name)
                        sftp.get(remote_path, local_path)
                    except Exception as e:
                        log_message(
                            f"Warning: Could not retrieve {file_name}: {str(e)}"
                        )
                        warning += (
                            f"Warning: Could not retrieve {file_name}: {str(e)}\n"
                        )
                return {
                    "remote_job_status": "finished",
                    "remote_job_id": job_id,
                    "success": True,
                    "message": warning,
                }
            elif status in ["FAILED", "CANCELLED"]:
                return {
                    "remote_job_status": "failed",
                    "remote_job_id": job_id,
                    "success": False,
                    "message": f"Remote job failed with status: {status}",
                }
        else:
            # Create remote working directory
            try:
                sftp.mkdir(remote_dir)
            except IOError:
                pass  # Directory might already exist

            # Transfer input files
            input_files = [top_file, gro_file, mdp_file, ndx_file]
            print(f"input_files: {input_files}")
            # also need to tranfer all the *.itp files
            input_files.extend(glob.glob(os.path.join(working_path, "*.itp")))

            for file_path in input_files:
                try:
                    remote_path = os.path.join(remote_dir, os.path.basename(file_path))
                    sftp.put(file_path, remote_path)
                    log_message(
                        f"file_path: {file_path}  transferred to remote_path: {remote_path}"
                    )
                except Exception as e:
                    log_message(
                        f"Error in transfer input files: {self.node_key}: {str(e)}",
                        level="error",
                    )
                    raise NodeException(
                        self.name,
                        f"Error in transfer input files: {self.node_key}: {str(e)}",
                    )

            # Create job script
            job_script = self._create_slurm_script(
                run_label,
                os.path.basename(top_file),
                os.path.basename(gro_file),
                os.path.basename(mdp_file),
                os.path.basename(ndx_file),
                cluster_config,
            )

            # print(f'job_script: {job_script}')

            # Write and transfer job script
            script_path = os.path.join(tempfile.gettempdir(), f"job_{run_label}.sh")
            with open(script_path, "w") as f:
                f.write(job_script)

            remote_script = os.path.join(remote_dir, f"job_{run_label}.sh")
            try:
                sftp.put(script_path, remote_script)
            except Exception as e:
                log_message(
                    f"Error in transfer job script: {self.node_key}: {str(e)}",
                    level="error",
                )
                return {"success": False, "error": str(e)}

            # Set execute permissions
            ssh.exec_command(f"chmod +x {remote_script}")

            # Submit job
            try:
                stdin, stdout, stderr = ssh.exec_command(
                    f"cd {remote_dir} && sbatch {remote_script}"
                )

                print(f"-----stderr: {stderr.read().decode()}")

                job_id = self._parse_job_id(stdout.read().decode())
                log_message(f"Job submitted with ID: {job_id}", level="info")
                if ssh:
                    ssh.close()
                return {
                    "remote_job_status": "submitted",
                    "remote_job_id": job_id,
                    "success": True,
                }
            except Exception as e:
                if ssh:
                    ssh.close()
                log_message(
                    f"Error in submit job: {self.node_key}: {str(e)}", level="error"
                )
                return {"success": False, "error": str(e)}

            # # Monitor job until completion
            # while True:
            #     stdin, stdout, stderr = ssh.exec_command(
            #         f'sacct -j {job_id} -o State -n -P')
            #     status = stdout.read().decode().strip().split('\n')[0]
            #     if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            #         break
            #     time.sleep(30)

    def _get_ssh_connection(self, cluster_config):
        """Create an SSH connection with passphrase support"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            private_key_path = os.path.expanduser(cluster_config["ssh_key_file"])

            # Try to load the private key with different key types
            if "ssh_key_passphrase" in cluster_config:
                try:
                    private_key = paramiko.RSAKey.from_private_key_file(
                        private_key_path, password=cluster_config["ssh_key_passphrase"]
                    )
                except paramiko.SSHException:
                    # Try Ed25519 key type
                    private_key = paramiko.Ed25519Key.from_private_key_file(
                        private_key_path, password=cluster_config["ssh_key_passphrase"]
                    )
            else:
                try:
                    private_key = paramiko.RSAKey.from_private_key_file(
                        private_key_path
                    )
                except paramiko.SSHException:
                    private_key = paramiko.Ed25519Key.from_private_key_file(
                        private_key_path
                    )

            # Connect with the key
            ssh.connect(
                hostname=cluster_config["host"],
                username=cluster_config["username"],
                pkey=private_key,
                timeout=30,
            )
            return ssh

        except Exception as e:
            raise NodeException(
                "ssh_connection",
                f"Failed to establish SSH connection: {str(e)} - Check if key file and passphrase are correct",
            )

    def prepare_job_script(
        self, run_label, top_file, gro_file, mdp_file, ndx_file, cluster_config
    ):
        """Prepare the GROMACS commands to run"""
        grompp_cmd = f"gmx grompp -f {mdp_file} -c {gro_file} -r {gro_file} -p {top_file} -n {ndx_file} -o {run_label}.tpr -maxwarn 10"
        mdrun_cmd = f"srun gmx_mpi mdrun -deffnm {run_label} -v"
        return f"{grompp_cmd}\n{mdrun_cmd}"

    def _create_slurm_script(
        self, run_label, top_file, gro_file, mdp_file, ndx_file, cluster_config
    ):
        """Create SLURM job submission script"""
        script = f"""#!/bin/bash
#SBATCH --job-name={run_label}
#SBATCH --output={run_label}.out
#SBATCH --nodes={cluster_config.get('nodes', 1)}
#SBATCH --ntasks-per-node={cluster_config.get('ntasks-per-node', 1)}
#SBATCH --partition={cluster_config.get('partition')}
#SBATCH --time={cluster_config.get('walltime', '01:00:00')}
#SBATCH --account={cluster_config.get('account', '')}
#SBATCH --mem-per-cpu=2G
module restore system
module load GROMACS/2023.3-foss-2022a

# Run grompp
gmx grompp -f {mdp_file} -c {gro_file} -r {gro_file} -p {top_file} -n {ndx_file} -o {run_label}.tpr -maxwarn 10

# Run MD simulation
srun gmx_mpi mdrun -deffnm {run_label} -v
"""
        return script

    def _parse_job_id(self, sbatch_output):
        """Extract job ID from sbatch output"""
        import re

        match = re.search(r"Submitted batch job (\d+)", sbatch_output)
        if match:
            return match.group(1)
        raise NodeException(
            "job_submission", "Could not parse job ID from sbatch output"
        )

    def run_local(
        self, run_label, top_file, gro_file, mdp_file, ndx_file, working_folder
    ):
        log_message(f"--- running init_md in step {run_label}", level="info")

        # excutable command
        cmd_grompp = f"gmx grompp -f {mdp_file} -c {gro_file} -r {gro_file} -p {top_file} -n {ndx_file} -o {run_label}.tpr -maxwarn 10"

        # cmd_run = f"srun gmx_mpi mdrun -deffnm {run_label} -v"
        cmd_run = f"gmx mdrun -deffnm {run_label} -v"

        subprocess.call(cmd_grompp, shell=True, cwd=working_folder)
        time.sleep(0.5)

        subprocess.call(cmd_run, shell=True, cwd=working_folder)
        time.sleep(0.5)

        return {"success": True, "message": "Success"}
