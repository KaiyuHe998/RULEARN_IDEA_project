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
import os
import sys

# CHIBI framework components
import world_basic_blocks as blocks
import space_manager as sm
import utils
import CHIBI
import plan_system
import fixed_interactive_pipeline_objects as fixed_blocks

from Judger import Judger

# Story profile
from all_puzzle_settings import *

# OPEN AI APIs
# openai api keys
import openai
from openai import AsyncOpenAI

Model_name = None

do_clear_screen = True
def clear_screen(do_clear_screen = do_clear_screen):
    if do_clear_screen:
        os.system('cls' if os.name == 'nt' else 'clear')
    
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
                CHIBI_name:Optional[str] = None):
    
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
        CHIBI_profile.Current_situation += f''' In this puzzle, you need to explore the patterns of reaction by conducting continuous experiments(The law is simple and can be described in one sentense). Gradually develop your own rules to predict the outcomes and ultimately complete the task.'''
        if puzzle_level == 1: # Only for LLM setting
            if 1<=puzzle_index<=5:
                CHIBI_profile.Current_situation += f'''The reaction is straightforward, simply combine two materials in the order you put them into the <Reactor>. Eg if you put A,B into the reactor it produces AB as output. Please use this rule to guide your action.'''
            elif 6<=puzzle_index<=10:
                CHIBI_profile.Current_situation += f'''The reaction is straightforward, simply combine two materials in the reverse order you put them into the <Reactor>. Eg if you put A,B into the reactor it produces BA as output. If you put AC, BD into the reactor it produces BDAC. Please use this rule to guide your action.'''
            elif 11<=puzzle_index<=15:
                CHIBI_profile.Current_situation += f'''The reaction inserts the shorter material into the middle of the longer material. Eg, If you put AB and C, it produces ACB. Please use this rule to guide your action.'''
            elif 16<=puzzle_index<=20:
                CHIBI_profile.Current_situation += f'''In this reaction, when two materials of different lengths are combined, the longer material's initial character is retained while the remainder of it is substituted with the shorter material. The replaced segment of the longer material is then kept alongside the newly formed product. However, there are two exceptions to this rule: 1. If the two materials are of the same length, they are simply concatenated. 2. If each material consists of only one letter, they are also simply concatenated. For example, AB+C = AC+B, AC+DD = ACDD. Please use this rule to guide your action.'''
        elif puzzle_level == 2:
            if 1<=puzzle_index<=5:
                CHIBI_profile.Current_situation += f'''You know from an incomplete list of reaction equations that: XY+Z = XYZ. X, Y, and Z are placeholders representing different reactants or products, not specific substances.'''
            elif 6<=puzzle_index<=10:
                CHIBI_profile.Current_situation += f'''You know from an incomplete list of reaction equations that: XY+Z = ZXY. X, Y, and Z are placeholders representing different reactants or products, not specific substances.'''
            elif 11<=puzzle_index<=15:
                CHIBI_profile.Current_situation += f'''You know from an incomplete list of reaction equations that: XY+Z = XZY. X, Y, and Z are placeholders representing different reactants or products, not specific substances.'''
            elif 16<=puzzle_index<=20:
                CHIBI_profile.Current_situation += f'''You know from an incomplete list of reaction equations that: XY+Z = XZ + Y. X, Y, and Z are placeholders representing different reactants or products, not specific substances.'''
            
    elif puzzle_setting_file_name == 'Art_gallery_puzzles':
        CHIBI_profile.Current_situation = f'''In this puzzle, set in an art gallery, {CHIBI_name} must uncover the password for the <Code Secured Door> by discovering the relationships between the password and the paintings. And finally input the password into the <Code Secured door>.'''
        if puzzle_level == 1: # only happen when using LLMs
            CHIBI_profile.Current_situation += " Currently, you know that the 3-digit password for the <Code Secured Door> is determined as follows: the first digit corresponds to the number of oil paintings in a specific color, the second digit to the number of acrylic paintings in that color, and the third digit to the number of watercolor paintings in the same color."
        elif puzzle_level == 2:
            CHIBI_profile.Current_situation += f''' Currently, you see from a note on the ground that says: "{utils.decorate_text_with_color('Focus on blue it hides the truth.','red',deep = True,bold = True)}"'''
        CHIBI_profile.Current_situation += f'''You can test your assumption by entering the password into the door. However, be aware that if you exceed the attempt limit, the password and hint will change.'''
    elif puzzle_setting_file_name == 'Function_operator_puzzles':
        CHIBI_profile.Current_situation += f'''You can test your assumption by entering values into the door. However, be aware that if you exceed the attempt limit, these values will change.'''
    if not human_test_bool:
        CHIBI_agent = CHIBI.CHIBI_main_character(CHIBI_profile,Space_Manager_System_Global,
                                                 Init_position = CHIBI_setting['Init_position'],
                                                 Model_name = Model_name, Do_abduction = Do_abduction,Special_label = puzzle_setting_file_name)
    else:
        CHIBI_agent = CHIBI.CHIBI_Human(CHIBI_profile, Space_Manager_System_Global,
                                        Init_position = CHIBI_setting['Init_position'],
                                        Model_name = Model_name, Do_abduction = Do_abduction,
                                        Special_label = puzzle_setting_file_name)
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
                      multiply_factor:int = 2,
                      human_test_bool:bool = False,
                      title_information:Optional[str] = None,
                      log_file_root_path:str = 'log_human/',
                      CHIBI_name:str = 'Sam',
                      round_index:int = None):
    assert round_index is not None,f'''You should input a round index'''
    Batch_generator = None

    # flag variables
    step_count = 0
    get_action_index_error_count = 0
    get_action_value_error_count = 0
    return_action_call_index_error_count = 0
    solution_found = False
    abduction_distribution = []
    
    Space_Manager_System_Global, CHIBI_agent, puzzle_setting = init_puzzle(puzzle_name, 
                                                                           puzzle_level = puzzle_level, puzzle_index = puzzle_index,
                                                                           Do_abduction = Do_abduction, 
                                                                           Model_name = Model_name,
                                                                           human_test_bool = human_test_bool, 
                                                                           CHIBI_name = CHIBI_name)
    all_state_machine_objects = []
    for single_space in Space_Manager_System_Global.Vertices_dict.values():
        objects_in_this_space = single_space.retrieve_item_in_this_space(object_type = 'All')
        for single_object in objects_in_this_space:
            if isinstance(single_object, fixed_blocks.State_machine_objects_Base):
                all_state_machine_objects.append(single_object)

    log_file_root_path = log_file_root_path
    log_file_folder = f'{puzzle_name}/level{puzzle_level}/index{puzzle_index}/round{round_index}'
    log_file_path = os.path.join(log_file_root_path, log_file_folder)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    memory_buffer_size = 0
    log_file_name = f'{puzzle_name}_{puzzle_level}_{puzzle_index}_Human_{Do_abduction}_{memory_buffer_size}_{round_index}_forceabd:False'
    log_file = os.path.join(log_file_path,log_file_name+'.log')
    if Batch_generator is None:
        utils.setup_logging(log_file)
    else:
        Cur_puzzle_logger = generate_logger(log_file_name, log_file)
        CHIBI_agent.Logger = Cur_puzzle_logger
    
    while step_count <= int(puzzle_setting['Optimal_step_count']*multiply_factor):
        print('******************************************************')
        print('******************Instructions************************')
        print('******************************************************')
        print(title_information)
        print('\n\n\n')
        print('******************************************************')
        print(f'''************Current puzzle progress: {step_count}/{int(puzzle_setting['Optimal_step_count']*multiply_factor)}*************''')
        print('******************************************************')

        CHIBI_agent.Plan_system.generate_actions()
        for single_object in all_state_machine_objects:
            single_object.update()
        try:
            return_action = CHIBI_agent.Plan_system.get_action()
            print(f'Cur step: {step_count}')
        except IndexError: 
            # when matching do not find the correct action index, do not follow the correct output format
            # eg: no parentheses bracket finded in the output
            get_action_index_error_count += 1
            step_count += 1 
            # traceback.print_exc()
            clear_screen()
            continue
        except ValueError: 
            # when matching do not find the correct action index, do not follow the correct output format
            # eg: Therefore, the best choice is to **[4. Input code to the Code Secured Door and try opening it]**.
            get_action_value_error_count += 1
            step_count += 1
            # traceback.print_exc()
            clear_screen()
            continue
        try:
            clear_screen()
            if human_test_bool:
                print('******************************************************')
                print('******************Instructions************************')
                print('******************************************************')
                print(title_information)
                print('\n\n\n')
                print('******************************************************')
                print(f'''************Current puzzle progress: {step_count}/{int(puzzle_setting['Optimal_step_count']*multiply_factor)}*************''')
                print('******************************************************')
            return_action()  
            if isinstance(return_action, plan_system.Attemptation_Perceptual_Action):
                print('A perceptual action called.')
            elif isinstance(return_action, plan_system.Attemptation_Abduction_Action):
                print('A abduction action called.')
                abduction_distribution.append(str(step_count))
            else:
                step_count += 1
            clear_screen()

        except IndexError:
            return_action_call_index_error_count += 1
            step_count += 1
            # traceback.print_exc()
            clear_screen()
            continue
        except TypeError as e:
            ("An error occurred: ", e)
            return_action_call_index_error_count += 1
            step_count += 1
            # traceback.print_exc()
            clear_screen()
            continue
        except SyntaxError as e:
            return_action_call_index_error_count += 1
            step_count += 1
            CHIBI_agent.Memory_stream.memory_add(e.msg)
            # traceback.print_exc()
            continue
        except utils.TaskCompletedException:
            # mission completed
            solution_found = True
            break
        except utils.TaskFailedException:
            break

        
    All_memories_str = '<New Row>'.join([memory_piece.get_information() for memory_piece in CHIBI_agent.Memory_stream.Memories])
    All_memories_str += '<New Row>'.join([memory_piece.get_information() for memory_piece in CHIBI_agent.Memory_stream.Buffer_memories])
    All_assumptions_str = '<New Row>'.join(CHIBI_agent.Memory_stream.All_assumptions)
    All_plans_str = '<New Row>'.join(CHIBI_agent.Memory_stream.All_plans)
    result_dict = {'puzzle_name':puzzle_name,
                   'puzzle_level':puzzle_level,
                   'puzzle_index':puzzle_index,
                   'Do_abduction':Do_abduction,
                   'forced_abduction':False,
                   'round_index':round_index,
                   'memory_buffer_size':0,
                   'Model_name':'Human',
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
                   'note':None,
                  }
    
    if os.path.exists(csv_file_name):
        result_df = pd.read_csv(csv_file_name)
    else:
        result_df = pd.DataFrame()
    row_df = pd.DataFrame([result_dict])
    result_df = pd.concat([result_df, row_df], ignore_index = True)
    
    result_df.to_csv(csv_file_name, index=False)

if __name__ == '__main__':
    Puzzle_subsample_dict = {
    'Function_operator_puzzles':[3, 4, 5, 7, 8, 11, 13, 15, 16, 17],
    'Art_gallery_puzzles':[1, 2, 4, 7, 10, 12, 13, 15, 17, 20],
    'Reactor_puzzles':[1, 3, 5, 7, 9, 11, 12, 14, 15, 19]
    }
    l1 = Puzzle_subsample_dict['Function_operator_puzzles']
    l2 = Puzzle_subsample_dict['Art_gallery_puzzles']
    l3 = Puzzle_subsample_dict['Reactor_puzzles']
    initial_index = [0,3,6] 
    final_assign = []
    for i in range(10):
        cur_assign = [l1[initial_index[0]], l2[initial_index[1]], l3[initial_index[2]]]
        final_assign.append(cur_assign)
        for n in range(len(initial_index)):
            initial_index[n] += 1
            if initial_index[n] == 10:
                initial_index[n] = 0
    root_file_name = 'human_test'
    csv_file_name = f'{root_file_name}.csv'
    log_file_root_path = f'log_{root_file_name}/'

    if getattr(sys, 'frozen', False):
        # 如果是通过PyInstaller打包的可执行文件，使用这个路径
        application_path = os.path.dirname(sys.executable)
    else:
        # 否则就使用脚本的路径
        application_path = os.path.dirname(os.path.abspath(__file__))
    csv_file_name = os.path.join(application_path, csv_file_name)
    log_file_root_path = os.path.join(application_path, log_file_root_path)
    

    puzzle_names = ['Function_operator_puzzles','Art_gallery_puzzles','Reactor_puzzles']
    puzzle_level = 2
    human_test_bool = True
    Model_name = 'Human'
    multiply_factor = 0.3 # 50*0.3 = 15 steps
    if human_test_bool:
        CHIBI_name = input(f'''Please think of a nickname for your character (Do not use your real name): ''')
    else:
        CHIBI_name = 'Sam'

    title_information = f'''Dear {CHIBI_name}: Thank you for participating in this research project. In this research project, you are going to solve {utils.decorate_text_with_color('3', 'blue', bold=True)} puzzles. For each puzzle, you will need to select an action at each step and, based on the feedback from that action, guess the puzzle's mechanism before ultimately solving it. If your observations do not match your assumptions about the puzzle's mechanism, or if you have a new modified assumption, you will need to write a short description of your current assumption and your next plan: whether to conduct more experiments to verify your assumption or to use it to solve the puzzle.  {utils.decorate_text_with_color('Please treat these puzzles as games, taking notes and writing down your thoughts as necessary. There is no need to force yourself to solve them. Just enjoy discovering the mechanisms of these puzzles, as this process is what we aim to evaluate. Essentially, you do not need to worry about the step count; it simply indicates when the puzzle will end, so just let it go.', 'red', bold=True)}\n\n{utils.decorate_text_with_color('Your storage information will show in green color.(Only the reactor puzzle will need storage information.)','green')}\n{utils.decorate_text_with_color('Your most recent action will show in red.','red')}\n{utils.decorate_text_with_color('Your previous information will show in blue color(Every time you make a new assumption your most recent actions will move to previous actions)','blue')}\n{utils.decorate_text_with_color('Your plan and assumption will show in magenta','magenta')}\n{utils.decorate_text_with_color('The question you need to answer will show in cyan','cyan',bold = True,deep = True)}'''

    # gpt-4o-2024-05-13
    # gpt-3.5-turbo-0125
    #print(f'You are currently dealting with {puzzle_name}, level:{puzzle_level}, index:{puzzle_index}')
    task_assign_index = int(input(f'''Please input your assigned task index (You should get the index from Instructor): '''))
    assert task_assign_index in [0,1,2,3,4,5,6,7,8,9], f'''Your input task assign index is not defined'''
    round_index = int(input(f'''Please input your assigned round index (You should get the index from Instructor): '''))
    assert round_index in [1,2,3,4,5], f'''Your input round index: {round_index} is invalid'''
    puzzle_indecies = final_assign[task_assign_index]
    run_an_experiment(csv_file_name, puzzle_names[0], puzzle_level, puzzle_indecies[0], Model_name, True, multiply_factor=multiply_factor, human_test_bool = True, CHIBI_name = CHIBI_name, title_information = title_information,log_file_root_path = log_file_root_path, round_index = round_index)

    sleep_count = 10
    for i in range(10):
        print(f'''You finished the first puzzle, next puzzle will starts in {sleep_count-i} seconds.''', end = '\r')
        time.sleep(1)
    clear_screen()
    print(f'''Now start the second puzzle.''')
    
    run_an_experiment(csv_file_name, puzzle_names[1], puzzle_level, puzzle_indecies[1], Model_name, True, multiply_factor=multiply_factor, human_test_bool = True, CHIBI_name = CHIBI_name, title_information = title_information,log_file_root_path = log_file_root_path, round_index = round_index)

    for i in range(10):
        print(f'''You finished the first puzzle, next puzzle will starts in {sleep_count-i} seconds.''', end = '\r')
        time.sleep(1)
    clear_screen()
    print(f'''Now start the third puzzle.''')
    run_an_experiment(csv_file_name, puzzle_names[2], puzzle_level, puzzle_indecies[2], Model_name, True, multiply_factor=multiply_factor, human_test_bool = True, CHIBI_name = CHIBI_name, title_information = title_information,log_file_root_path = log_file_root_path, round_index = round_index)
    
    clear_screen()
    print('''All puzzles are finished, thank you for participating!''')