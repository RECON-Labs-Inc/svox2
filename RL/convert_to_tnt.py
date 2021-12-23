import sys
sys.path.append("..")
sys.path.append("/workspace/aseeo-research")
import RLResearch.utils.gen_utils as gu



project_folder = "/workspace/datasets/plant"
output_folder = "/workspace/datasets/plant"

gu.make_tnt_dataset(project_folder, output_folder)


