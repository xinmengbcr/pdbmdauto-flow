from bocoflow_core.node import Node, NodeResult
from bocoflow_core.parameters import (
    BooleanParameter,
    CondaEnvParameter,
    FileParameter,
    StringParameter,
)


class GmxWithCondaNode(Node):
    """
    Example node demonstrating the use of conda environments

    This node runs GROMACS commands within a specific conda environment.
    It uses the CondaEnvParameter to allow the user to select which
    conda environment to use for running GROMACS.
    """

    name = "GROMACS with Conda"
    node_key = "GmxWithCondaNode"
    color = "blue"
    num_in = 0
    num_out = 1

    OPTIONS = {
        "input_structure": FileParameter(
            label="Input Structure", docstring="Input structure file (PDB, GRO)"
        ),
        "output_file": StringParameter(
            label="Output File",
            default="rel:output.gro",
            docstring="Output structure file",
        ),
        "conda_env": CondaEnvParameter(
            label="Conda Environment",
            docstring="Conda environment containing GROMACS",
            optional=True,
        ),
        "force_to_run": BooleanParameter(
            label="Force to run",
            default=False,
            docstring="Force to run the node even if it has been executed successfully before",
        ),
    }

    def execute(self, predecessor_data, flow_vars):
        """
        Execute the GROMACS command using the specified conda environment

        The conda environment setup is handled by the Workflow.execute method,
        which prepares the environment before calling this method.
        """
        result = NodeResult()

        try:
            # Get parameters
            input_structure = flow_vars["input_structure"].get_value()
            output_file = flow_vars["output_file"].get_value()

            # At this point, the conda environment should already be set up
            # by the Workflow.execute method, so we can import GROMACS
            try:
                import gromacs as gmx

                result.message = f"Successfully imported GROMACS from conda environment"
            except ImportError as e:
                result.success = False
                result.message = f"Failed to import GROMACS: {str(e)}"
                return result.to_json()

            # Mock GROMACS execution for demonstration
            result.data["gromacs_version"] = gmx.__version__
            result.data["input_file"] = input_structure
            result.data["output_file"] = output_file

            # Record file paths
            result.files["input"]["structure"] = input_structure
            result.files["output"]["structure"] = output_file

            # Set success
            result.success = True
            result.message = "GROMACS command executed successfully"

        except Exception as e:
            result.success = False
            result.message = f"Error executing GROMACS command: {str(e)}"

        return result.to_json()
