import os
import platform
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
def Return_root_dir():
    if platform.system() == "Windows":
        root_dir = r"C:\Users\didri\Desktop\Programmering\ArtificalintelligenceModels\UNet_Gan_model_Video_Enchancer"
    elif platform.system() == "Linux":
        root_dir = "/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet_Gan_model_Video_Enchancer"
    else:
        raise OSError("Unsupported Platform")
    return root_dir

