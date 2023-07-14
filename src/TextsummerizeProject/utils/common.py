import os
from box.exceptions import BoxValueError
import yaml
from src.TextsummerizeProject.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    '''
        Read the yaml file
        args: path_to_yaml
        raises: 
            valueError:if yaml file is empty,
            e: empty file
        returns: configbox contents
    '''
    try:
        with open(path_to_yaml, 'r') as YamlFile:
            content = yaml.safe_load(YamlFile)
            logger.info(f'Yaml file {YamlFile} read Successfully')
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError('Yaml file {YamlFile} is empty')
    except Exception as e:
        raise e
    
@ensure_annotations
def Create_directory(path_to_directory: list, verbose=True):
    '''
        Create a directory
        args: path_to_directory
        returns: None
    '''
    
    for path in path_to_directory:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'Directory {path} created Successfully')


@ensure_annotations
def get_size(path:Path) -> str:
    '''
        gets size in kb
        args: path_to_directory
        Returns: size in KB
    '''
    size = round(os.path.getsize(path)/1024)
    return f'~ {size} KB'
        