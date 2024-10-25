import os
import torch
from tqdm.notebook import tqdm

# transformers Model save path & gpu setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # set your gpus
os.environ["TRANSFORMERS_CACHE"] = '../hg_model_cache'
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
if torch.cuda.is_available():
    print("available num GPU", torch.cuda.device_count())
    
    for i in range(torch.cuda.device_count()):
        print(f'''GPU {i}: {torch.cuda.get_device_name(i)}''')
        print("  - Total G mem:", torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), "GB")
        print("  - #Cores:", torch.cuda.get_device_properties(i).multi_processor_count)
    device = 'cuda'
else:
    assert False, f'''No GPUs found in current environment'''
    print("No GPU.")
    device = 'cpu'

from typing import *
import random
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Heiti TC'] 
plt.rcParams['axes.unicode_minus'] = False 
from abc import ABC, abstractmethod
import time
import ast
import re
import copy
import pandas as pd
import datetime
import logging
import shutil
import concurrent.futures
import traceback

import world_basic_blocks as blocks
import space_manager as sm
import utils
import CHIBI
import plan_system
import fixed_interactive_pipeline_objects as fixed_blocks
import threading

from Judger import Judger

# Story profile
from all_puzzle_settings import *

# OPEN AI APIs
# openai api keys
import openai
from openai import AsyncOpenAI

Model_name = None
#'gpt-3.5-turbo-0125'#'gpt-4-1106-preview'#'gpt-3.5-turbo-0125'#'gpt-4-turbo-2024-04-09'#'gpt-3.5-turbo-0125'

def generate_puzzle_spaces(puzzle_setting_file:Dict[str, Any],
                           puzzle_level:int = 1,
                           puzzle_index:int = 1,
                           Model_name:str = 'gpt-3.5-turbo-0125')->List[blocks.Space_Base]:
    puzzle_level_key = 'Level'+str(puzzle_level)
    puzzle_index_key = 'puzzle'+str(puzzle_index)
    puzzle_setting = puzzle_setting_file[puzzle_level_key][puzzle_index_key]
    spaces =  sm.Space_helper.generate_all_room_with_database(puzzle_setting['Map'],
                                                              puzzle_setting['Space_items'],
                                                              puzzle_setting['Space_item_containers'],
                                                              puzzle_setting['Edges'],
                                                              Model_name = Model_name)
    Space_Manager_System_Global = sm.Space_Manager_System(spaces)
    CHIBI_profile = puzzle_setting['Agent']
    return Space_Manager_System_Global, CHIBI_profile, puzzle_setting
    
def init_puzzle(puzzle_setting_file_name:str,
                puzzle_level:int,
                puzzle_index:int,
                Do_abduction:bool = False,
                Model_name:str = 'gpt-3.5-turbo-0125',
                human_test_bool:bool = False,
                memory_buffer_size:int = 0,
                CHIBI_name:Optional[str] = None,
                Batch_generator:Optional[utils.Prompt_batch_generator] = None,
                forced_abduction:bool = False):
    
    puzzle_dict = {'Reactor_puzzles':Reactor_puzzles,
                   'Art_gallery_puzzles':Art_gallery_puzzles,
                   'Function_operator_puzzles':Function_operator_puzzles}
    
    puzzle_setting_file = puzzle_dict[puzzle_setting_file_name]
    Space_Manager_System_Global, CHIBI_setting, puzzle_setting = generate_puzzle_spaces(puzzle_setting_file, 
                                                                        puzzle_level, 
                                                                        puzzle_index, 
                                                                        Model_name = Model_name)
    CHIBI_profile = CHIBI.CHIBI_helper.create_profile_with_legacy_file(CHIBI_setting, CHIBI_name = CHIBI_name)
    if puzzle_setting_file_name == 'Reactor_puzzles':
        CHIBI_profile.Current_situation += f''' In this puzzle, you need to explore the patterns of reaction by conducting continuous experiments. Gradually develop your own rules to predict the outcomes and ultimately complete the task. Letters in a single material string is separated with space (but it is a single material).'''
        if puzzle_level == 1: # add rule for level 1 puzzle
            if 1<=puzzle_index<=5:
                CHIBI_profile.Current_situation += f'''The reaction is straightforward, simply combine two materials in the order you put them into the <Reactor>. E.g. C + A -> {utils.sms('CA')} and {utils.sms('CA')} + B -> {utils.sms('CAB')}. Please use this rule to guide your action.'''
            elif 6<=puzzle_index<=10:
                CHIBI_profile.Current_situation += f'''The reaction is straightforward, simply combine two materials in the reverse order you put them into the <Reactor>. C + A -> {utils.sms('CA')}, {utils.sms('CA')} + B -> {utils.sms('BCA')} and {utils.sms('AC')} + {utils.sms('BD')} = {utils.sms('BDAC')}.'''
            elif 11<=puzzle_index<=15:
                CHIBI_profile.Current_situation += f'''The reaction inserts the shorter material into the middle of the longer material. Eg, C+A -> {utils.sms('CA')} and {utils.sms('CA')}+B -> {utils.sms('CBA')}. Please use this rule to guide your action.'''
            elif 16<=puzzle_index<=20:
                CHIBI_profile.Current_situation += f'''In this reaction, when two materials of different lengths are combined, the longer material's initial character is retained while the remainder of it is substituted with the shorter material. The replaced segment of the longer material is then kept alongside the newly formed product. However, there are two exceptions to this rule: 1. If the two materials are of the same length, they are simply concatenated. 2. If each material consists of only one letter, they are also simply concatenated. For example, C + A -> {utils.sms('CA')}, {utils.sms('CA')} + B -> {utils.sms('CB')} + A, {utils.sms('AC')} + {utils.sms('DD')} -> {utils.sms('ACDD')}, A + {utils.sms('AA')} -> {utils.sms('AAA')}. Please use this rule to guide your action.'''
        elif puzzle_level == 2:
            if 1<=puzzle_index<=5:
                CHIBI_profile.Current_situation += f'''You currently know that: C + A -> {utils.sms('CA')} and {utils.sms('CA')} + B -> {utils.sms('CAB')}. You need to further discover how the reaction happens. And synthesis required target product.'''
            elif 6<=puzzle_index<=10:
                CHIBI_profile.Current_situation += f'''You currently know that: C + A -> {utils.sms('CA')} and {utils.sms('CA')} + B -> {utils.sms('BCA')}. You need to further discover how the reaction happens. And synthesis required target product. '''
            elif 11<=puzzle_index<=15:
                CHIBI_profile.Current_situation += f'''You currently know that: C + A -> {utils.sms('CA')} and {utils.sms('CA')} + B -> {utils.sms('CBA')}. You need to further discover how the reaction happens. And synthesis required target product.'''
            elif 16<=puzzle_index<=20:
                CHIBI_profile.Current_situation += f'''You currently know that: C + A -> {utils.sms('CA')} and {utils.sms('CA')} + B -> {utils.sms('CB')} + A. You need to further discover how the reaction happens. And synthesis required target product.'''
            
    elif puzzle_setting_file_name == 'Art_gallery_puzzles':
        CHIBI_profile.Current_situation = f'''In this puzzle, set in an art gallery, {CHIBI_name} must uncover the password for the <Code Secured Door> by discovering the relationships between the password and the paintings. And finally input the password into the <Code Secured door>.'''
        if puzzle_level == 1:
            CHIBI_profile.Current_situation += " Currently, you know that the 3-digit password for the <Code Secured Door> is determined as follows: the first digit corresponds to the number of oil paintings in a specific color, the second digit to the number of acrylic paintings in that color, and the third digit to the number of watercolor paintings in the same color."
        elif puzzle_level == 2:
            CHIBI_profile.Current_situation += ''' Currently, you see from a note on the ground that says: "Focus on blue it hides the truth."'''
        CHIBI_profile.Current_situation += f'''You can test your hypothesis by entering the password into the door. However, be aware that if you exceed the attempt limit, the password and hint will change.'''
    elif puzzle_setting_file_name == 'Function_operator_puzzles':
        CHIBI_profile.Current_situation += f'''You can test your hypothesis by entering values into the door. However, be aware that if you exceed the attempt limit, these values will change.'''
        if puzzle_level == 1:
            CHIBI_profile.Current_situation += " Currently, you know that the 3-digit password for the <Code Secured Door> is determined as follows: the first digit corresponds to the number of oil paintings in a specific color, the second digit to the number of acrylic paintings in that color, and the third digit to the number of watercolor paintings in the same color."
        elif puzzle_level == 2:
            CHIBI_profile.Current_situation += ''' Currently, you see a note on the ground that says: "Focus on blue it hides the truth."'''
        CHIBI_profile.Current_situation += f'''You can test your hypothesis by entering the password into the door. However, be aware that if you exceed the attempt limit, the password and hint will change.'''
    elif puzzle_setting_file_name == 'Function_operator_puzzles':
        CHIBI_profile.Current_situation += f'''You can test your hypothesis by entering values into the door. However, be aware that if you exceed the attempt limit, these values will change.'''
    if not human_test_bool:
        CHIBI_agent = CHIBI.CHIBI_main_character(CHIBI_profile,Space_Manager_System_Global,
                                                 Init_position = CHIBI_setting['Init_position'],
                                                 Model_name = Model_name, Do_abduction = Do_abduction, 
                                                 Special_label = puzzle_setting_file_name,
                                                 forced_abduction = forced_abduction)
    else:
        CHIBI_agent = CHIBI.CHIBI_Human(CHIBI_profile, Space_Manager_System_Global,
                                        Init_position = CHIBI_setting['Init_position'],
                                        Model_name = Model_name, Do_abduction = Do_abduction) # human agent do not need forced abduction
    CHIBI_agent.Memory_stream.Buffer_size = memory_buffer_size
    new_state = (f'{CHIBI_agent.Profile.Current_situation}',)
    for i in new_state:
        CHIBI_agent.Plan_system.add_state(i)
    return Space_Manager_System_Global,CHIBI_agent, puzzle_setting
    

def run_an_experiment(csv_file_name:str,
                      puzzle_name:str, 
                      puzzle_level:int,
                      puzzle_index:int, 
                      Model_name:int, 
                      Do_abduction:bool,
                      multiply_factor:int = 1,
                      forced_abduction:bool = False,
                      human_test_bool:bool = False,
                      title_information:Optional[str] = None,
                      memory_buffer_size:int = 0,
                      CHIBI_name:str = 'Kevin',
                      Batch_generator:Optional[utils.Prompt_batch_generator] = None,
                      log_file_root_path:str = 'log/',
                      round_index:int = None,
                      cur_parameter:Dict[str,Any] = None,
                      note = None):
    assert cur_parameter is not None, f'''Please also passin the current parameter for this task'''
    assert round_index is not None, f'''This task should be assigned a round index first.'''
    assert Batch_generator is not None, f'''need a batch generator to run this expriment'''
    # flag variables
    step_count = 0
    abduction_count = 0
    get_action_index_error_count = 0
    get_action_value_error_count = 0
    return_action_call_index_error_count = 0
    solution_found = False
    Batch_generator.register_task(cur_parameter)
    abduction_distribution = []

    if '/' in Model_name:
        Model_name = Model_name.replace('/','_') 
        # Model_name only used for logging file, so if there is '/' in model name it is a huggingface name, the inferences happens in the Batch_generator
    Space_Manager_System_Global, CHIBI_agent, puzzle_setting = init_puzzle(puzzle_name, 
                                                                           puzzle_level = puzzle_level, puzzle_index = puzzle_index,
                                                                           Do_abduction = Do_abduction, 
                                                                           Model_name = Model_name,
                                                                           human_test_bool = human_test_bool, 
                                                                           CHIBI_name = CHIBI_name,
                                                                           memory_buffer_size = memory_buffer_size,
                                                                           Batch_generator = Batch_generator,
                                                                           forced_abduction = forced_abduction)
    all_state_machine_objects = []
    for single_space in Space_Manager_System_Global.Vertices_dict.values():
        objects_in_this_space = single_space.retrieve_item_in_this_space(object_type = 'All')
        for single_object in objects_in_this_space:
            if isinstance(single_object, fixed_blocks.State_machine_objects_Base):
                all_state_machine_objects.append(single_object)

    
    log_file_root_path = log_file_root_path
    log_file_folder = f'{puzzle_name}/level{puzzle_level}/index{puzzle_index}/round{round_index}'
    log_file_path = os.path.join(log_file_root_path, log_file_folder)

    # for backup files
    log_file_root_path_backup = log_file_root_path.replace('/','')+'_backup/'
    log_file_backup_folder = f'{puzzle_name}/level{puzzle_level}/index{puzzle_index}/round{round_index}'
    log_file_backup_path = os.path.join(log_file_root_path_backup, log_file_folder)
    with Batch_generator.lock: 
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)

    log_file_name = f'{puzzle_name}_{puzzle_level}_{puzzle_index}_{Model_name}_{Do_abduction}_{memory_buffer_size}_{round_index}_forceabd:{forced_abduction}'
    log_file = os.path.join(log_file_path,log_file_name+'.log')
    
    previous_log_information = []
    if os.path.exists(log_file): # exist previous log file, load previous log file and make a backup
        with Batch_generator.lock: 
            with open(log_file, 'r') as file:
                for line in file:
                    previous_log_information.append(line.strip()) # read every generated sentence
            if not os.path.exists(log_file_backup_path):
                os.makedirs(log_file_backup_path)
            log_file_backup_name = log_file_name
            log_file_backup = os.path.join(log_file_backup_path,log_file_backup_name+'.log')
            if os.path.exists(log_file_backup): # if there is alread one backup file, delete previous one
                os.remove(log_file_backup)
            shutil.move(log_file, log_file_backup) # backup previouslog information into a new folder

    Cur_puzzle_logger = utils.generate_logger(log_file_name, log_file)
    CHIBI_agent.Logger = Cur_puzzle_logger
    CHIBI_agent.Batch_generator = Batch_generator
    CHIBI_agent.previous_log_information = previous_log_information
    Drop_this_puzzle = False
    while step_count < int(puzzle_setting['Optimal_step_count']*multiply_factor) and abduction_count <= int(puzzle_setting['Optimal_step_count']*multiply_factor)+1:
        if human_test_bool:
            print('******************************************************')
            print('******************Instructions************************')
            print('******************************************************')
            print(title_information)
            print('\n\n\n')
        CHIBI_agent.Plan_system.generate_actions()
        for single_object in all_state_machine_objects:
            single_object.update()
        try:
            return_action = CHIBI_agent.Plan_system.get_action()
            if not isinstance(return_action, plan_system.Attemptation_Perceptual_Action):
                print('******************************************************')
                print(f'''************Current puzzle progress: {step_count}/{int(puzzle_setting['Optimal_step_count']*multiply_factor)}*************''')
                print(f'''Cur_puzzle_name: {puzzle_name}, Cur_puzzle_level: {puzzle_level}, Cur_puzzle_index:{puzzle_index},Cur_buffer_size:{memory_buffer_size}, Cur_Model_name: {Model_name}, Cur_round:{round_index}, Do_abduction:{Do_abduction}, Forced_abduction:{forced_abduction}''')
                print('******************************************************')
        except IndexError: 
            # when matching do not find the correct action index, do not follow the correct output format
            # eg: no parentheses bracket finded in the output
            get_action_index_error_count += 1
            step_count += 1 
            continue
        except ValueError: 
            # when matching do not find the correct action index, do not follow the correct output format
            # eg: Therefore, the best choice is to **[4. Input code to the Code Secured Door and try opening it]**.
            get_action_value_error_count += 1
            step_count += 1
            traceback.print_exc()
            continue
        except TypeError as e:
            ("An error occurred: ", e)
            get_action_value_error_count += 1
            step_count += 1
            traceback.print_exc()
            continue
        except utils.GenerateErrorException:
            traceback.print_exc()
            Drop_this_puzzle = True
            break
        try:
            return_action()
            if isinstance(return_action, plan_system.Attemptation_Perceptual_Action):
                print('A perceptual action called.')
            elif isinstance(return_action, plan_system.Attemptation_Abduction_Action):
                print('A abduction action called.')
                abduction_count += 1
                abduction_distribution.append(str(step_count))
                if abduction_count >= int(puzzle_setting['Optimal_step_count']*multiply_factor)+1: 
                    print(f'''Doing too much abduction, puzzle failed''')
            else:
                step_count += 1
            CHIBI_agent.Plan_system.Previous_called_action = return_action

        except IndexError:
            return_action_call_index_error_count += 1
            step_count += 1
            traceback.print_exc()
            continue
        except TypeError as e:
            ("An error occurred: ", e)
            return_action_call_index_error_count += 1
            step_count += 1
            traceback.print_exc()
            continue
        except SyntaxError as e:
            return_action_call_index_error_count += 1
            step_count += 1
            CHIBI_agent.Memory_stream.memory_add(e.msg)
            continue
        except utils.TaskCompletedException:
            # mission completed
            solution_found = True
            break
        except utils.TaskFailedException:
            break
        except utils.GenerateErrorException:
            traceback.print_exc()
            Drop_this_puzzle = True
            break
        
    if not Drop_this_puzzle:
        All_memories_str = '<New Row>'.join([memory_piece.get_information() for memory_piece in CHIBI_agent.Memory_stream.Memories])
        All_memories_str += '<New Row>'.join([memory_piece.get_information() for memory_piece in CHIBI_agent.Memory_stream.Buffer_memories])
        All_assumptions_str = '<New Row>'.join(CHIBI_agent.Memory_stream.All_assumptions)
        All_plans_str = '<New Row>'.join(CHIBI_agent.Memory_stream.All_plans)
        result_dict = {'puzzle_name':puzzle_name,
                       'puzzle_level':puzzle_level,
                       'puzzle_index':puzzle_index,
                       'Do_abduction':Do_abduction,
                       'forced_abduction':forced_abduction,
                       'round_index':round_index,
                       'memory_buffer_size':memory_buffer_size,
                       'Model_name':Model_name,
                       'finish_step_count':step_count,
                       'optimal_step_count':puzzle_setting['Optimal_step_count'],
                       'solution_found':solution_found,
                       'get_action_index_error_count':get_action_index_error_count,
                       'get_action_value_error_count':get_action_value_error_count,                           
                       'return_action_call_index_error_count':return_action_call_index_error_count,
                       'experiment_run_time':str(datetime.datetime.now()),
                       'CHIBI_name':CHIBI_name,
                       'All_memories_str':All_memories_str,
                       'All_assumptions_str':All_assumptions_str,
                       'All_plans_str':All_plans_str,
                       'Abduction_distrubution':'<step>'.join(abduction_distribution),
                       'note':note,
                      }
        with Batch_generator.lock:
            if os.path.exists(csv_file_name):
                result_df = pd.read_csv(csv_file_name)
            else:
                result_df = pd.DataFrame()
            row_df = pd.DataFrame([result_dict])
            result_df = pd.concat([result_df, row_df], ignore_index = True)
            
            result_df.to_csv(csv_file_name, index=False)
        print(result_dict)
        Batch_generator.unregister_task(cur_parameter,Drop_this_puzzle)
        cur_thread =  threading.current_thread()
        print(f'Thread: {cur_thread.name} finished')
    else:
        Batch_generator.unregister_task(cur_parameter,Drop_this_puzzle)

if __name__ == '__main__':
    #"meta-llama/Meta-Llama-3-70B-Instruct"
    #"meta-llama/Meta-Llama-3-8B-Instruct"
    #"google/gemma-7b-it"
    #'gpt-4o-2024-05-13'
    #'gpt-3.5-turbo-0125'
    #'gpt-4o-2024-08-06'
    max_length = 512
    openai_model_names = ['gpt-4o-2024-05-13','gpt-3.5-turbo-0125','gpt-4o-2024-08-06','gpt-4o-global-batch','gpt-4-global-batch']
    model_name = 'gpt-4o-global-batch'#'gpt-4o-2024-08-06' #"meta-llama/Meta-Llama-3-70B-Instruct"

    # setting keys and api tokens
    use_azure_api = True
    azure_endpoint = "your endpoint"
    azure_api_key = "your key"
    if model_name in openai_model_names: # use openai api
        token = 'your openai token'
        batch_size = 60
    else: 
        token = 'your hugging face token' # use huggingface models
        batch_size = 15
    
    root_file_name = model_name.split('/')[-1].replace('-','_')
    root_file_name = root_file_name + 'azure_test'
    csv_file_name = f'{root_file_name}.csv'
    log_file_root_path = f'log_{root_file_name}/'
    multiply_factor = 0.3 # 50*0.3 = 15 steps, this control max_step
    temperature = 0.5
    note = None
    
    global_batch_generator = utils.Prompt_batch_generator(model_name = model_name, 
                                                          max_length = max_length,
                                                          token = token,
                                                          batch_size = batch_size,
                                                          openai_model_list = openai_model_names,
                                                          use_azure_api = use_azure_api,
                                                          endpoint = azure_endpoint,
                                                          azure_api_key = azure_api_key,
                                                          temperature = temperature
                                                          )
    
    all_experiment_parameters = []
    for round_index in [1,2,3]:#[1,2,3,4,5]:#[1,2,3,4,5]:
        for level in [2,1]: # Level 1 is the oracle setting and agent is provided with ground truth rule
            for puzzle_index in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
                for Do_abduction in [True, False]:
                    for forced_abduction in [False]: # Do not use False abduction (currently disabled)
                        for puzzle_name in ['Reactor_puzzles']:#['Reactor_puzzles','Art_gallery_puzzles','Function_operator_puzzles']:
                            if level == 1 and Do_abduction:
                                pass
                            elif not Do_abduction and forced_abduction:
                                pass
                            else:
                                puzzle_parameters = {'puzzle_name':puzzle_name,
                                                     'level':level,
                                                     'puzzle_index':puzzle_index,
                                                     'Do_abduction':Do_abduction,
                                                     'round_index':round_index,
                                                     'forced_abduction':forced_abduction}
                                all_experiment_parameters.append(puzzle_parameters)
    if os.path.exists(csv_file_name):
        finished_parameters = utils.get_all_finished_experiments(csv_file_name)
        for parameter in finished_parameters:
            if parameter in all_experiment_parameters:
                all_experiment_parameters.remove(parameter)
    global_batch_generator.all_experiment_parameters = all_experiment_parameters.copy()
    
    if len(all_experiment_parameters) < batch_size:
        global_batch_generator.batch_size = len(all_experiment_parameters)
    
    total_num_tasks = len(all_experiment_parameters)
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for cur_experiment_parameter in all_experiment_parameters:
            tuple_args = (csv_file_name, 
                          cur_experiment_parameter['puzzle_name'], 
                          cur_experiment_parameter['level'], 
                          cur_experiment_parameter['puzzle_index'], 
                          model_name, 
                          cur_experiment_parameter['Do_abduction'])
            keyword_args = {'multiply_factor':multiply_factor,
                            'Batch_generator': global_batch_generator,
                            'log_file_root_path':log_file_root_path,
                            'round_index':cur_experiment_parameter['round_index'],
                            'cur_parameter':cur_experiment_parameter,
                            'forced_abduction':cur_experiment_parameter['forced_abduction'],
                            'note':note}
            futures.append(executor.submit(run_an_experiment, *tuple_args, **keyword_args))
    
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(f"Task Result: {result}")
            except Exception as e:
                print(f"An error occurred: {e}")
            
    print('All tasks completed!')
    print(f'Total_batch_generated:{global_batch_generator.total_batch_processed}, Batch_size:{global_batch_generator.batch_size}, Average_batch_time:{global_batch_generator.average_batch_time}')