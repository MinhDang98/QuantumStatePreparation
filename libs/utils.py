import numpy as np
import os
import shutil
from libs.quantum_state_preparation import QuantumStatePreparation
from qiskit.quantum_info import Statevector


def clean_log(log_folder_path: str = "./logs"):
	"""
	Cleans all files and subdirectories from a specified log folder.
	
	Args:
	    log_folder_path (str): The path to the log folder to be cleaned.
	"""
	if not os.path.exists(log_folder_path):
		print(f"Log folder '{log_folder_path}' does not exist.")
		return

	print(f"Cleaning contents of '{log_folder_path}'...")
	for item in os.listdir(log_folder_path):
		item_path = os.path.join(log_folder_path, item)
		try:
			if os.path.isfile(item_path) or os.path.islink(item_path):
				os.unlink(item_path)  # Remove file or link.
				print(f"  Deleted file: {item_path}")
			elif os.path.isdir(item_path):
				shutil.rmtree(item_path)  # Remove directory and its contents.
				print(f"  Deleted directory: {item_path}")
		except Exception as e:
			print(f"  Error deleting {item_path}: {e}")
	print(f"Finished cleaning '{log_folder_path}'.")