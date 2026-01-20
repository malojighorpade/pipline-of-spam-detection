def load_params(params_path: str)->dict:
#     """Load YAML parameters file."""
#     try:
#         with open('params.path','r')as file:
#             params=yaml.safe_load(file)
#             logger.debug('Parameters retrived from %s',params_path)
#             return params
#     except FileNotFoundError as e:
#         logger.error('Parameters file not found: %s',params_path)
#         raise 
#     except yaml.YAMLError as e:
#         logger.error('Error parsing YAML file: %s',params_path)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error loading parameters: %s',str(e))
#         raise