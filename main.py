import sys
from script.core.pipeline_manager import PipelineManager


language = sys.argv[1]
dataset_name = sys.argv[2]
file_name = sys.argv[3]

Manager = PipelineManager(language, dataset_name, file_name)
Manager.run()
