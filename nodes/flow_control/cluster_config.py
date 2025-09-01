# cluster_config.py
import json

from bocoflow_core.node import FlowNode
from bocoflow_core.parameters import *

# slurm example: https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/


class ClusterConfigNode(FlowNode):
    """Cluster(Slurm) Configuration Node"""

    name = "Cluster"
    node_key = "ClusterConfigNode"
    num_in = 0
    num_out = 0
    color = "purple"

    OPTIONS = {
        "host": StringParameter(
            label="Host Address", docstring="Remote cluster hostname"
        ),
        "account": StringParameter(label="Account Name"),
        "username": StringParameter(label="Username"),
        "ssh_key_file": FileParameterEdit(label="SSH Key File"),
        "ssh_key_passphrase": PasswordParameter(
            label="SSH Key Passphrase",
            docstring="Passphrase for SSH key (if required)",
            optional=True,
        ),
        "remote_workdir": StringParameter(label="Remote Working Directory"),
        "partition": StringParameter(label="Partition/Queue", default="normal"),
        "nodes": IntegerParameter(label="Number of Nodes", default=1),
        "ntasks_per_node": IntegerParameter(label="Tasks per Node", default=16),
        "mem-per-cpu": StringParameter(label="Memory per CPU", default="1024M"),
        "walltime": StringParameter(
            label="Wall Time", default="1-00:00:00", docstring="Format: DD-HH:MM:SS"
        ),
        "var_name": StringParameter(
            label="Variable Name",
            default="cluster_config",
            docstring="Name of the cluster configuration variable",
        ),
    }

    def get_replacement_value(self):
        """Override get_replacement_value to return options directly"""
        return {
            "host": self.options["host"].get_value(),
            "account": self.options["account"].get_value(),
            "username": self.options["username"].get_value(),
            "ssh_key_file": self.options["ssh_key_file"].get_value(),
            "ssh_key_passphrase": self.options["ssh_key_passphrase"].get_value(),
            "remote_workdir": self.options["remote_workdir"].get_value(),
            "partition": self.options["partition"].get_value(),
            "nodes": self.options["nodes"].get_value(),
            "ntasks-per-node": self.options["ntasks_per_node"].get_value(),
            "walltime": self.options["walltime"].get_value(),
        }

    def execute(self, predecessor_data, flow_vars):
        return json.dumps({"success": True})
