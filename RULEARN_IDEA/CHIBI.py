# basics
from typing import *
from abc import ABC, abstractmethod
import datetime
import re
import logging
import os

# CHIBI framework components
import world_basic_blocks as blocks
import space_manager
import memory_stream
import utils
import plan_system
# -----------------------------------------------------------------------------------------------------------
# --------------------------------------------Perception Classes------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
class Perception_Base(ABC):
    def __init__(self):
        pass
        
    @abstractmethod
    def perceive(self):
        '''Use this method to let CHIBI generate customized observation based on their profile and task'''
        pass

class Perception_main_character(Perception_Base):
    def __init__(self,
                 Host_CHIBI:'CHIBI_main_character')->str:
        self.Host_CHIBI = Host_CHIBI
        
    def perceive(self, 
                 Object:blocks.CHIBI_Object,
                 mode:str = 'Identical'):
        object_information = Object.get_information(self.Host_CHIBI)

        if mode == 'Identical':
            return object_information
        elif mode == 'Prompt':
            print('TODO:use Prompt to generate customized observation')
            return ''
        else:
            raise ValueError(f'current mode:{mode} is not supported')

# -----------------------------------------------------------------------------------------------------------
# --------------------------------------------Profile Classes------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
class Profile(ABC):
    def __init__(self, 
                 Name:str,
                 Current_situation:str):
        self.Name = Name
        self.Current_situation = Current_situation
    
    @abstractmethod
    def edit(self):
        pass
    
    @abstractmethod
    def get_relative_profile_information(self):
        pass

    @abstractmethod
    def show(self):
        '''Show all contents'''
        pass
        
class Profile_main_character(Profile):
    ''' CHIBI's profile is hard code need some paper or theory to generate personalized profile like curiosty behavior style...'''
    def __init__(self,
                 Name:str, 
                 Current_situation:str = None,
                 Items:blocks.Thing_container = None,
                 Solid_memory: blocks.Information_piece_container = None,
                 Action_style: blocks.Information_piece_container = None,
                 ):
        self.Name = Name
        self.Current_situation = Current_situation
        # TODO: will need a better way to do following initialization
        Keyword_for_empty_container = {'Items':f'Your belongings',
                                       'Solid_memory':f'Memorization of specific information such as your identity, occupation, habits and lifestyle.',
                                       'Action_style':f'Your style and habit of taking action',}
                                       #'Relationship_memory':f'memories for other people,
                                       #'Experiences':f'experience for different type of events'}

        if Items is None:
            self.Items = blocks.Thing_container(Keyword_for_empty_container['Items'],{})
        else:
            self.Items = Items
        
        if Solid_memory is None:
            self.Solid_memory = blocks.Information_piece_container(
                Keyword_for_empty_container['Solid_memory'], {})
        else:
            self.Solid_memory = Solid_memory
            
        if Action_style is None:
            self.Action_style = blocks.Information_piece_container(
                Keyword_for_empty_container['Action_style'], {})
        else:
            self.Action_style = Action_style
        
        #if Relationship_memory is None:
        #    self.Relationship_memory = blocks.Experience_container(
        #        Keyword_for_empty_container['Relationship_memory'],{})
        #else:
        #    self.Relationship_memory = Relationship_memory
        #    
        #if Experiences is None:
        #    self.Experiences = blocks.Experience_container(
        #        Keyword_for_empty_container['Experiences'],{})
        #else:
        #    self.Experiences = Experiences
                                #f"{self.Relationship_memory.Keyword}":self.Relationship_memory,
    def edit(self):
        pass

    def get_all_items(self)->List[blocks.Thing]:
        return self.Items.object_retrieve()
    
    def update_cur_situation(self): # don't need this anymore,  this already handled in the States
        '''When something happens need to update the character's current situation, the current situation attribute needs to summarize the character's current recent memories as well as plans and stories and so on, and it will be used every time you need global prompts (eg to make a plan, a decision, a way to talk) so that you need to use the global information.''' 
        pass
        
    def show(self):
        print_str = ''
        print_str += self.Current_situation + '\n'
        print_str += self.Items.object_retrieve(mode = 'Return_string') + '\n'
        print_str += self.Action_style.object_retrieve(mode = 'Return_string') + '\n'
        print_str += self.Solid_memory.object_retrieve(mode = 'Return_string') + '\n'
        print(print_str)
    
    def get_relative_profile_information(self,
                                         task:Optional[str] = None) ->str:
        # TODO: 之后这里再好好修缮一下
        return_str = ''
        if task is None:
            '''if task is None by default return all profile information in this profile'''
            return_str += self.Current_situation + '\n'
            return_str += self.Items.object_retrieve(mode = 'Return_string') + '\n'
            return_str += self.Action_style.object_retrieve(mode = 'Return_string') + '\n'
            return_str += self.Solid_memory.object_retrieve(mode = 'Return_string') + '\n'
            return return_str

        elif task == 'generate_successor':
            return_str += self.Current_situation + '\n'
            return_str += self.Items.object_retrieve(mode = 'Return_string') + '\n'
            return_str += self.Action_style.object_retrieve(mode = 'Return_string') + '\n'
            return return_str
            
        elif task == 'backpack':
            return_str += self.Items.object_retrieve(mode = 'Return_string') + '\n'
            return return_str
        else:
            assert False,f'{retrieve_mode} not supported'

# -----------------------------------------------------------------------------------------------------------
# --------------------------------------------CHIBI Classes--------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
class CHIBI_Base(blocks.CHIBI_Object,ABC):
    def __init__(self,
                 Name:str,
                 Profile:Profile,
                 Spaces:List[blocks.Space_System_global],
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Plan_system:Optional[plan_system.Plan_System_Base] = None,
                 Do_abduction:bool = False,
                 Special_label:str = None
                ):
        self.Name = Name
        self.Keyword = Name
        self.Profile = Profile
        self.Space_manager = space_manager.Space_Manager_CHIBI(Spaces)
        self.Plan_system = Plan_system
        self.Do_abduction = Do_abduction
        self.Special_label = Special_label
        
    @abstractmethod
    def get_action(self)->plan_system.Attemptation_Action_Base: # Only in free text mode not in use currently
        '''All CHIBI agent should at least let other's know it's status or action
         Take an action'''
        pass


    @abstractmethod
    def take_action(self,
                    action:plan_system.Attemptation_Action_Base):
        pass

    @abstractmethod
    def CHIBI_input(self, Prompt:str, Input:str)->Any:
        pass
        
class CHIBI_main_character(CHIBI_Base):
    def __init__(self, 
                 Profile:Profile,
                 Space_Manager_System_Global:space_manager.Space_Manager_System,
                 Init_position:str = 'Haven', #TODO change this default value
                 Perception:Perception_main_character = None,
                 Spaces_memories:Optional[Dict[str,blocks.Space_CHIBI_impression]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:'openai.openai_object.OpenAIObject' = None,
                 Memory_stream: Optional[memory_stream.Memory_stream_CHIBI_main_character] = None,
                 Plan_system:Optional[plan_system.Plan_System_CHIBI_main_character] = None,
                 Init_time:Optional[datetime.datetime] = None,
                 Do_abduction:bool = False,
                 chibi_name:str = 'Sam',
                 Batch_generator:Optional[utils.Prompt_batch_generator] = None,
                 Logger:Optional[logging.Logger] = None,
                 Special_label:str = None,
                 Huggingface_model:'huggingface model' = None,  # only used when you want to try a single puzzle with hugging face model
                 forced_abduction:bool = False,
                 ):
        # TODO: add a function to parse Initial_position
        self.Space_Manager_System_Global = Space_Manager_System_Global
        self.Profile = Profile
        self.Usage = Usage
        self.Model_name = Model_name
        self.Recent_memory = []
        self.Cur_action = 'This chibi is in a daze right now. The CHIBI isn\'t taking any action'
        self.Name = self.Profile.Name
        self.Keyword = self.Name
        self.Plan_system = Plan_system
        self.Do_abduction = Do_abduction
        self.CHIBI_type = 'GPT_agent'
        self.Special_label = Special_label
        self.Huggingface_model = None             # will be assigned in the outer environment for example notebook
        self.Huggingface_tokenizer = None         # will be assigned in the outer environment for example notebook
        self.Huggingface_generate_max_len = None  # will be assigned in the outer environment for example notebook
        if Spaces_memories is None:
            self.Space_manager = space_manager.Space_Manager_CHIBI(self, 
                                                               Init_position = Init_position, 
                                                               Vertices_dict = {})
        else:
            self.Space_manager = space_manager.Space_Manager_CHIBI(self, 
                                                               Init_position = Init_position, 
                                                               Vertices_dict = Spaces_memories)

        if Memory_stream is None:
            self.Memory_stream =  memory_stream.Memory_stream_CHIBI_main_character([],Belongs_to = self)
        else:
            self.Memory_stream = Memory_stream 
        if Plan_system is None:
            self.Plan_system = plan_system.Plan_System_CHIBI_main_character(self, Model_name = self.Model_name, forced_abduction = forced_abduction)
        self.Space_Manager_System_Global.Vertices_dict[Init_position].object_add(self)
        if Perception is None:
            self.Perception = Perception_main_character(self)
        if Init_time is None:
            self.Cur_time = datetime.datetime.now()
        else:
            self.Cur_time = Init_time
        self.Batch_generator = Batch_generator
        self.Logger = Logger
        
    # ------------------------CHIBI get informations--------------------
    def get_profile_information(self, 
                                task:str = None):
        return self.Profile.get_relative_profile_information(task = task)

    def look_around(self,
                    mode:str = 'Default') -> Tuple[str,Dict[str,Any]]:
        # Please make sure that for all the real object keyword and impression keyword are the same
        '''update all object impression in current space'''
        cur_space_str = self.Space_manager.Cur_position
        cur_real_objects = []
        for value in self.Space_Manager_System_Global.Vertices_dict[cur_space_str].All_objects.values():
            cur_real_objects.extend(value)
            
        object_have_impression = [i.Impression_of for i in self.Space_manager.Vertices_dict[cur_space_str].All_objects]
        # for edge double side object, it will only have one impression
        
        for cur_real_object in cur_real_objects:
            if cur_real_object not in object_have_impression and not isinstance(cur_real_object, CHIBI_Base):
                # Currently impression of other agent is not handled yet
                new_impression_information = f'Looks like a {cur_real_object.Keyword} haven\'t investigated it closely yet.'
                new_impression = blocks.Object_Impression(cur_real_object.Keyword, new_impression_information, cur_real_object,cur_real_object.Belongs_to,self.Space_manager.Vertices_dict[cur_space_str])
                self.Space_manager.Vertices_dict[cur_space_str].All_objects.append(new_impression)
        # currently object will not disappear unless CHIBI interact with it, so no need to check if an object has impression but real object has disappeared
        self.Space_manager.update_space_description(generate_method = 'system')

    def recall_impressions(self,
                           top_n:Optional[int] = None,
                           path_length_decay_factor:float = 0.75):
        # recall impression based on current state information
        cur_state = self.Plan_system.Cur_state
        all_impressions = []
        for space in self.Space_manager.Vertices_dict.values():
            all_impressions.extend(space.All_objects)
        if top_n is None:# return all impressions
            top_n = len(all_impressions)
        if len(all_impressions) <= top_n:
            return all_impressions
            
        scores = []
        for impression in all_impressions:
            # TODO if this impression related to an edge object this could cause a bug
            impression_space_str = impression.Impression_space.Space_name
            path_length = len(self.Space_manager.find_path(self.Space_manager.Cur_position, impression_space_str))
            path_score = path_length_decay_factor**path_length

            # relevance score
            relevance_score = utils.calculate_cosine_similiarity(cur_state.Embedding, impression.Embedding)
            scores.append(path_score + relevance_score)

        paired = sorted(zip(all_impressions, scores), key=lambda x: x[1], reverse=True)
        return [obj for obj, score in paired[:top_n]]

    # ------------------------CHIBI Interfaces-----------------
    # preception related functions (get information from environment or from profile or anywhere else)
    
    # CHIBI action related functions
    def get_action(self)->List[plan_system.Attemptation_Action_Base]:# takes 2300 tokens per plan_node_simpe
        return self.Plan_system.generate_action() # Only in free text mode not in use currently

    # CHIBI movement related functions
    def move(self,
             move_to:Union['blocks.Space_System_global','blocks.Space_CHIBI_impression',str],
             mode = 'Explorative_move'):
        # TODO use edge object instead of Space name
        if isinstance(move_to, blocks.Space_Base):
            move_to = move_to.Space_name
        if mode == 'Explorative_move':
            '''only move to a space previously haven't been before'''
            self.Space_Manager_System_Global.Vertices_dict[self.Space_manager.Cur_position].object_delete(self)
            self.Space_manager.Cur_position = move_to
            self.Space_Manager_System_Global.Vertices_dict[move_to].object_add(self)
            if move_to not in self.Space_manager.Vertices_dict:
                new_space_impression = self.Space_manager.create_new_space_CHIBI_impression(move_to)
            # TODO add information here if the CHIBI travel through the map
            # only explorative move will be added to memory stream
            # self.Memory_stream.memory_add()
        elif mode == 'GO':
            '''go to a path previously arrived and have impression of it'''
            # currently we have assumption the edge will not be locked onec it is been opened and CHIBI can freely go to any space he have impression of. (Maybe not like this if in the future CHIBI's action will change the space structure this may not be a good choice)
            assert move_to in self.Space_manager.Vertices_dict,f'{self.Name} have never been to {move_to}, he doesn\'t know how to go there'
            self.Space_Manager_System_Global.Vertices_dict[self.Space_manager.Cur_position].object_delete(self)
            self.Space_manager.Cur_position = move_to
            self.Space_Manager_System_Global.Vertices_dict[move_to].object_add(self)

            # check every edge to see if the edge is passable (is this needed?)
        else:
            raise ValueError(f'{mode} is not supported')
        
    def take_action(self,
                    attempt_action_object:plan_system.Attemptation_Action_Base):
        # TODO, when an action is been called, 
        if isinstance(attempt_action_object, plan_system.Attemptation_Movement_Action) or \
           isinstance(attempt_action_object, plan_system.Attemptation_Perceptual_Action) or \
           isinstance(attempt_action_object, plan_system.Plan_attemptation_interactive_action):
            attempt_action_object()
        else:
            raise ValeError(f'{type(attempt_action_object)} unknown supported type for action')
    # ------------------------Object related interface-----------------
    def destory(self):
        pass
    
    def edit(self):
        pass
    
    def get_information(self)->str:
        return self.Name
    
    def get_keyword(self)->str:
        return self.Name
    
    def show(self):
        pass
    # ------------------------Memory related functions-----------------
    def memory_add(self,
                   memory_to_be_added:Union[blocks.Memory_piece,str]):
        self.Memory_stream.memory_add(memory_to_be_added)
        
    def update_unit_step(self,
                         unit_time: Optional[datetime.timedelta] = None): # Currently not in use
        '''This function should handle all system level hard coded numerical status' update like time, hungry, energy......'''
        # memory stream update
        if unit_time is None:
            print('Unit time should provided')
        else:
            self.Memory_stream.update(unit_time)

    def CHIBI_input(self, Prompt:str, Input:str, parse_function_str = 'str_with_tuple', logging_label:Optional[str] = None, max_input_len = 6000) -> str:
        if self.Batch_generator is None:
            # only for showing one puzzle in note book for example
            if self.Huggingface_model is None:
                # using openai model for generation
                @utils.Prompt_constructor_for_system(self.Model_name,
                                                     Usage = self.Usage,
                                                     parse_function_str = parse_function_str,
                                                     logging_label = logging_label)
                def _prompt_and_input():
                    return Prompt, Input
                
        
                generated_whole_test = _prompt_and_input()
                generated_result = generated_whole_test['parsed_result']
                whole_message = generated_whole_test['messages']
                whole_message = whole_message[-1]
            else:
                input_data = [Prompt + '\n' + Input + '\Please finish your answer within 100 words make it consice and do not repeat above instructions.']
                input_ids = self.Huggingface_tokenizer(input_data, return_tensors="pt", padding=True).to(self.Huggingface_model.device)
                outputs = self.Huggingface_model.generate(**input_ids,
                                              max_new_tokens=self.Huggingface_generate_max_len,     
                                              num_beams=1,      
                                              do_sample = True,
                                              temperature = 0.01,
                                              top_p=None,                 
                                              top_k=None)
                generated_str = [self.Huggingface_tokenizer.decode(outputs[i],skip_special_tokens=True) for i in range(len(outputs))]
                assert len(generated_str) == 1, f'''In a single puzzle test, can only generate one sentence at once.'''
                generated_result = generated_str[0]
                whole_message = generated_str[0]
                if parse_function_str is None:
                    generated_result = generated_result.replace(input_data[0],'') # do nothing
                else:
                    if parse_function_str == 'ast':
                        parse_function = ast.literal_eval
                    elif parse_function_str == 'str_with_tuple':
                        parse_function = utils._parse_str_with_tuple
                    elif parse_function_str == 'str_with_angle_bracket':
                        parse_function = utils._parse_str_with_angle_bracket
                    elif parse_function_str == 'str_with_square_bracket':
                        parse_function = utils._parse_str_with_square_bracket
                    else:
                        assert False, f'{parse_function_str} not known parse function'
                    generated_result = parse_function(generated_result) 
            
            print(f'''------------------------------------------------Prompt------------------------------------------------''')
            print(Prompt, Input)
            print(f'''------------------------------------------------Generated Result------------------------------------------------''')
            print(whole_message)
            print(f'''------------------------------------------------Parsed Result------------------------------------------------''')
            print(generated_result)
            input(f'''Input anything to continue!''')
            generated_result = utils.truncate_with_max_len(generated_result)
            return generated_result
        else:
            #assert self.Logger is not None, f'''If you are using batch generator and threading, you need to specify a logger for each puzzle running'''
            input_prompt = Prompt + '\n\n' + Input + "\Please finish your answer within 100 words make it consice and do not repeat above instructions."
            if len(self.previous_log_information) > 0:
                generated_result = self.previous_log_information.pop(0).split('**Generated_answer**:')[1].replace('<New Row>','\n')
            else:
                self.Batch_generator.add_data(input_prompt)
                generated_result = self.Batch_generator.get_result(input_prompt)
                if generated_result == "Generating Failed!!!!!":
                    raise utils.GenerateErrorException(f'''Puzzle hault when generating this batch of puzzle, return result is None''')
            if logging_label is not None:
                if self.Logger is not None:
                    Prompt_str = f'''**{logging_label}**: {Prompt}\n{Input}\n\n''' 
                    generated_str = f'''**Generated_answer**:{generated_result}'''
                    logging_information = Prompt_str+generated_str
                    clean_logging_information = logging_information.replace('\n','<New Row>')
                    self.Logger.info(clean_logging_information)
            if parse_function_str is None:
                parsed_result = generated_result
            else:
                if parse_function_str == 'ast':
                    parse_function = ast.literal_eval
                elif parse_function_str == 'str_with_tuple':
                    parse_function = utils._parse_str_with_tuple
                elif parse_function_str == 'str_with_angle_bracket':
                    parse_function = utils._parse_str_with_angle_bracket
                elif parse_function_str == 'str_with_square_bracket':
                    parse_function = utils._parse_str_with_square_bracket
                else:
                    assert False, f'{parse_function_str} not known parse function'
                parsed_result = parse_function(generated_result) 
                if isinstance(parsed_result, str): # only string output could exceed length limit
                    parsed_result = utils.truncate_with_max_len(parsed_result)
            return parsed_result
 
    def get_impression_object(self, 
                              real_object:blocks.CHIBI_Object,
                              space_name:Optional[str] = None):
        '''Use real object to get impression object for current CHIBI'''
        # TODO: check if there are any pointers are not able to delete
        if space_name is not None:
            impression_space = self.Space_manager.Vertices_dict[space_name]
            return impression_space.Descovered_objects[real_object.Keyword]
        else:
            for space_impression in self.Space_manager.Vertices_dict.values():
                if real_object.Keyword in space.Descovered_objects:
                    return space.Descovered_objects[real_object.Keyword]
            
        print(f'impression of {real_object.get_keyword()} didn\'t found!!!!!!!!!!!!')
        
    # ------------------------Some custom functions-----------------
    def grab_item(self,
                  item:blocks.Thing,):
        '''grab an item from current space and collect it into personal storage'''
        self.Profile.Items.object_add(item)
        # add grab item experience and information edit
        carry_experience_string = f'''{self.Name} puts {item.get_keyword()} into his storage.'''
        self.Memory_stream.memory_add(carry_experience_string)
        self.Plan_system.Cur_state.edit(carry_experience_string)

        # delete impression object
        impression_object = self.Space_manager.find_impression_object(item)
        self.Space_manager.get_cur_space(space_type = 'impression').object_delete(impression_object)

    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''

    def retrieve_prompt_information(self, memory_use:Optional[int] = None):
        most_recent_memories = self.Memory_stream.Buffer_memories.copy()
        memory_str = '\n'.join([i.get_information() for i in most_recent_memories])
        if len(most_recent_memories) == 0:
            most_recent_memories_str = ''
        else: 
            if self.Memory_stream.Cur_assumption_and_plan is None:
                most_recent_memories_str = f'''Here are {len(most_recent_memories)} most recent activities that {self.Name} has done has done (listed in chronological order from oldest to most recent):\n{memory_str}\n\n'''
            else:
                most_recent_memories_str = f'''Here {len(most_recent_memories)} most recent activities that {self.Name} has done under your current assumption (listed in chronological order from oldest to most recent):\n{memory_str}\n\n'''

        previous_memories = self.Memory_stream.memory_retrieve(top_n = memory_use, memory_retrieve_type = 'Most_relevant')
        if len(previous_memories) == 0:
            previous_memories_str = ''
        else:
            previous_memories_str = '\n'.join([i.get_information() for i in previous_memories])
            previous_memories_str = f'''Here are actions that {self.Name} did previously:\n{previous_memories_str}\n\n'''

        if self.Memory_stream.Cur_assumption_and_plan is None:
            cur_assumption_str = ''
        else:
            cur_assumption_str = f'''''' # included in the memory stream

        if len(self.Profile.Items.All_objects) == 0:
            storage_information = f'Currently your storage is empty!\n\n'
        else:
            storage_items = list(self.Profile.Items.All_objects.values())
            
            storage_information = f'''You currently have the following items in your storage: {', '.join(['<'+utils.sms(i.get_keyword()[1:-1])+'>' for i in storage_items])}\n\n'''

        cur_plan_str = ''

        if self.Special_label != 'Reactor_puzzles':
            storage_information = ''
        return previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str
        

class CHIBI_Human(CHIBI_Base):
    def __init__(self, 
                 Profile:Profile,
                 Space_Manager_System_Global:space_manager.Space_Manager_System,
                 Init_position:str = 'Haven', #TODO change this default value
                 Perception:Perception_main_character = None,
                 Spaces_memories:Optional[Dict[str,blocks.Space_CHIBI_impression]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:'openai.openai_object.OpenAIObject' = None,
                 Memory_stream: Optional[memory_stream.Memory_stream_Human] = None,
                 Plan_system:Optional[plan_system.Plan_System_Human] = None,
                 Init_time:Optional[datetime.datetime] = None,
                 Do_abduction:bool = False,
                 Special_label:Optional[str] = None,
                 ):
        # TODO: add a function to parse Initial_position
        self.Space_Manager_System_Global = Space_Manager_System_Global
        self.Special_label = Special_label # Currently the puzzle name
        self.Profile = Profile
        self.Usage = Usage
        self.Model_name = Model_name
        self.Recent_memory = []
        self.Cur_action = 'This chibi is in a daze right now. The CHIBI isn\'t taking any action'
        self.Name = self.Profile.Name
        self.Keyword = self.Name
        self.Plan_system = Plan_system
        self.Do_abduction = Do_abduction
        self.CHIBI_type = 'Human'
        if Spaces_memories is None:
            self.Space_manager = space_manager.Space_Manager_CHIBI(self, 
                                                               Init_position = Init_position, 
                                                               Vertices_dict = {})
        else:
            self.Space_manager = space_manager.Space_Manager_CHIBI(self, 
                                                               Init_position = Init_position, 
                                                               Vertices_dict = Spaces_memories)
        if Memory_stream is None:
            self.Memory_stream =  memory_stream.Memory_stream_Human([],Belongs_to = self)
        else:
            self.Memory_stream = Memory_stream 
        if Plan_system is None:
            self.Plan_system = plan_system.Plan_System_Human(self)
        self.Space_Manager_System_Global.Vertices_dict[Init_position].object_add(self)
        if Perception is None:
            self.Perception = Perception_main_character(self)
        if Init_time is None:
            self.Cur_time = datetime.datetime.now()
        else:
            self.Cur_time = Init_time


    # ------------------------CHIBI get informations--------------------
    def get_profile_information(self, 
                                task:str = None):
        return self.Profile.get_relative_profile_information(task = task)

    def look_around(self,
                    mode:str = 'Default') -> Tuple[str,Dict[str,Any]]:
        # Please make sure that for all object the real object keyword and impression keyword are the same
        '''update all object impression in current space'''
        cur_space_str = self.Space_manager.Cur_position
        cur_real_objects = []
        for value in self.Space_Manager_System_Global.Vertices_dict[cur_space_str].All_objects.values():
            cur_real_objects.extend(value)
            
        object_have_impression = [i.Impression_of for i in self.Space_manager.Vertices_dict[cur_space_str].All_objects]
        # for edge double side object, it will only have one impression
        
        for cur_real_object in cur_real_objects:
            if cur_real_object not in object_have_impression and not isinstance(cur_real_object, CHIBI_Base):
                # Currently impression of CHIBI is not handled yet
                new_impression_information = f'Looks like a {cur_real_object.Keyword} haven\'t investigated it closely yet.'
                new_impression = blocks.Object_Impression(cur_real_object.Keyword, new_impression_information, cur_real_object,cur_real_object.Belongs_to,self.Space_manager.Vertices_dict[cur_space_str], Need_embedding = False)
                self.Space_manager.Vertices_dict[cur_space_str].All_objects.append(new_impression)
        # currently object will not disappear unless CHIBI interact with it, so no need to check if an object have impression but real object is disappeared
        self.Space_manager.update_space_description(generate_method = 'system')

    def recall_impressions(self,
                           top_n:Optional[int] = None,
                           path_length_decay_factor:float = 0.75):
        # recall impression based on current state information
        cur_state = self.Plan_system.Cur_state
        all_impressions = []
        for space in self.Space_manager.Vertices_dict.values():
            all_impressions.extend(space.All_objects)
        if top_n is None:# return all impressions
            top_n = len(all_impressions)
        if len(all_impressions) <= top_n:
            return all_impressions
            
        scores = []
        for impression in all_impressions:
            # TODO if this impression related to an edge object this could cause a bug
            impression_space_str = impression.Impression_space.Space_name
            path_length = len(self.Space_manager.find_path(self.Space_manager.Cur_position, impression_space_str))
            path_score = path_length_decay_factor**path_length

            # relevance score
            relevance_score = utils.calculate_cosine_similiarity(cur_state.Embedding, impression.Embedding)
            scores.append(path_score + relevance_score)

        paired = sorted(zip(all_impressions, scores), key=lambda x: x[1], reverse=True)
        return [obj for obj, score in paired[:top_n]]

    # ------------------------CHIBI Interfaces-----------------
    # preception related functions (get information from environment or from profile or anywhere else)
    
    # CHIBI action related functions
    def get_action(self)->List[plan_system.Attemptation_Action_Base]:# takes 2300 tokens per plan_node_simpe
        return self.Plan_system.generate_action() # Only in free text mode not in use currently

    def CHIBI_input(self, Prompt:str, Input:str, parse_function_str:Optional[str] = None, logging_label:Optional[str] = None):
        if logging_label == 'Interact_input':
            print(f'''{utils.decorate_text_with_color('-----------------Now decide your input based on the followin instruction:-----------------','cyan',bold = True, deep = True)}''')
        if logging_label == 'Induction' or logging_label == 'Abduction':
            human_input = input(Prompt + '\n\n' + Input +'\n\nPlease finish your answer within 512 words make it as concise as possible.\n') # TODO read the max length from the parameters
        else:
            human_input = input(Prompt + '\n\n' + Input)
        if parse_function_str is None:
            pass
        else:
            human_input = re.sub(r'[^\w,-.]', '', human_input)
        if parse_function_str == 'str_with_tuple':
            if '(' not in human_input and ')' not in human_input:
                human_input = f'''({human_input})'''
            return utils._parse_str_with_tuple(human_input)

        if logging_label is not None:
            Prompt_str = f'''**{logging_label}**: {Prompt}\n{Input}\n\n''' 
            generated_str = f'''**Generated_answer**: {human_input}'''
            logging_information = Prompt_str+generated_str
            clean_logging_information = logging_information.replace('\n','<New Row>')
            logging.info(clean_logging_information)
            
        return human_input

    def decorate_list_str(self,list_of_str:List[str], color:str):
        if len(list_of_str) == 1:
            return utils.decorate_text_with_color(list_of_str[-1], color)
        elif len(list_of_str) >= 2:
            return_str = '\n'.join(list_of_str[:-1])
            return_str = utils.decorate_text_with_color(return_str, color) + '\n'
            end_return_str = '\n'.join(list_of_str[-1:])
            return_str += utils.decorate_text_with_color(end_return_str, color, bold = True)
            return return_str

    def retrieve_prompt_information(self, memory_use:Optional[int] = None):
        most_recent_memories = self.Memory_stream.Buffer_memories.copy()
        most_recent_memories_str = [i.get_information() for i in most_recent_memories]

        memory_str = self.decorate_list_str(most_recent_memories_str, 'red')
        if len(most_recent_memories) == 0:
            most_recent_memories_str = ''
        else: 
            if self.Memory_stream.Cur_assumption_and_plan is None:
                most_recent_memories_str = f'''Here are {len(most_recent_memories)} most recent activities that you have done (listed in chronological order from oldest to most recent):\n{memory_str}\n\n'''
            else:
                most_recent_memories_str = f'''Here are {len(most_recent_memories)} most recent activities that you have done under your current assumption (listed in chronological order from oldest to most recent):\n{memory_str}\n\n'''

        
        previous_memories = self.Memory_stream.memory_retrieve(top_n = memory_use, memory_retrieve_type = 'Most_relevant')
        previous_memories_str_list = [i.get_information() for i in previous_memories]
        previous_memories_str = self.decorate_list_str(previous_memories_str_list, 'blue')
        
        if len(previous_memories) == 0:
            previous_memories_str = ''
        else:
            previous_memories_str = f'''Here are actions that you took previously:\n{previous_memories_str}\n\n'''

        if self.Memory_stream.Cur_assumption_and_plan is None:
            cur_assumption_str = ''
        else:
            assumption_str = utils.decorate_text_with_color(self.Memory_stream.get_assumption(), 'magenta', bold = True)
            cur_assumption_str = f'''''' #moved to memories

        # if self.Memory_stream.Cur_plan is None:
        #     cur_plan_str = ''
        # else:
        #     cur_plan_str = utils.decorate_text_with_color(self.Memory_stream.Cur_plan.get_information(), 'magenta', bold = True) 
        #     cur_plan_str = f'''Currently you have the following plan: {cur_plan_str}\n\n'''
        cur_plan_str = '' # Cur plan is not in use

        if len(self.Profile.Items.All_objects) == 0:
            storage_information = f'Your storage is empty!\n\n'
        else:
            storage_items = list(self.Profile.Items.All_objects.values())
            storage_str = utils.decorate_text_with_color(', '.join([i.get_keyword() for i in storage_items]), 'green')
            storage_information = f'''You currently have the following items in your storage: {storage_str}\n\n'''

        if self.Special_label != 'Reactor_puzzles':
            storage_information == ''
        return previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str
        
    # CHIBI movement-related functions
    def move(self,
             move_to:Union['blocks.Space_System_global','blocks.Space_CHIBI_impression',str],
             mode = 'Explorative_move'):
        # TODO use edge object instead of Space name
        if isinstance(move_to, blocks.Space_Base):
            move_to = move_to.Space_name
        if mode == 'Explorative_move':
            ''' Only move to a space previously haven't been before'''
            self.Space_Manager_System_Global.Vertices_dict[self.Space_manager.Cur_position].object_delete(self)
            self.Space_manager.Cur_position = move_to
            self.Space_Manager_System_Global.Vertices_dict[move_to].object_add(self)
            if move_to not in self.Space_manager.Vertices_dict:
                new_space_impression = self.Space_manager.create_new_space_CHIBI_impression(move_to)
            # TODO add information here if the CHIBI travel through the map
            # only explorative move will be added to memory stream
            # self.Memory_stream.memory_add()
        elif mode == 'GO':
            '''go to a path previously arrived and have impression of it'''
            # currently we have assumption the edge will not be locked onec it is been opened and CHIBI can freely go to any space he have impression of. (Maybe not like this if in the future CHIBI's action will change the space structure this may not be a good choice)
            assert move_to in self.Space_manager.Vertices_dict,f'{self.Name} have never been to {move_to}, he doesn\'t know how to go there'
            self.Space_Manager_System_Global.Vertices_dict[self.Space_manager.Cur_position].object_delete(self)
            self.Space_manager.Cur_position = move_to
            self.Space_Manager_System_Global.Vertices_dict[move_to].object_add(self)

            # check every edge to see if the edge is passable (is this needed?)
        else:
            raise ValueError(f'{mode} is not supported')
        
    def take_action(self,
                    attempt_action_object:plan_system.Attemptation_Action_Base):
        # TODO, when an action is been called, 
        if isinstance(attempt_action_object, plan_system.Attemptation_Movement_Action) or \
           isinstance(attempt_action_object, plan_system.Attemptation_Perceptual_Action) or \
           isinstance(attempt_action_object, plan_system.Plan_attemptation_interactive_action):
            attempt_action_object()
        else:
            raise ValeError(f'{type(attempt_action_object)} unknown supported type for action')
    # ------------------------Object related interface-----------------
    def destory(self):
        pass
    
    def edit(self):
        pass
    
    def get_information(self)->str:
        return self.Name
    
    def get_keyword(self)->str:
        return self.Name
    
    def show(self):
        pass
    # ------------------------Memory related functions-----------------
    def memory_add(self,
                   memory_to_be_added:Union[blocks.Memory_piece,str]):
        self.Memory_stream.memory_add(memory_to_be_added)
        
    def update_unit_step(self,
                         unit_time: Optional[datetime.timedelta] = None):
        ''' This function should handle all system level hard coded numerical status' update like time, hungry, energy......'''
        # Currently not in use
        # memory stream update
        if unit_time is None:
            print('Unit time should provided')
        else:
            self.Memory_stream.update(unit_time)
            
    def get_impression_object(self, 
                              real_object:blocks.CHIBI_Object,
                              space_name:Optional[str] = None):
        '''Use real object to get impression object for current CHIBI'''
        # TODO: check if there are any pointers are not able to delete
        if space_name is not None:
            impression_space = self.Space_manager.Vertices_dict[space_name]
            return impression_space.Descovered_objects[real_object.Keyword]
        else:
            for space_impression in self.Space_manager.Vertices_dict.values():
                if real_object.Keyword in space.Descovered_objects:
                    return space.Descovered_objects[real_object.Keyword]
            
        print(f'impression of {real_object.get_keyword()} didn\'t found!!!!!!!!!!!!')
    # ------------------------Some custom functions-----------------
    def grab_item(self,
                  item:blocks.Thing,):
        '''grab an item from current space and collect it into personal storage'''
        self.Profile.Items.object_add(item)
        # add grab item experience and information edit
        carry_experience_string = f'''{self.Name} puts {item.get_keyword()} into his storage.'''
        self.Memory_stream.memory_add(carry_experience_string)
        self.Plan_system.Cur_state.edit(carry_experience_string)

        # delete impression object
        impression_object = self.Space_manager.find_impression_object(item)
        self.Space_manager.get_cur_space(space_type = 'impression').object_delete(impression_object)

    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''
        
# --------------------------------------------Helper Class------------------------------------------------
class CHIBI_helper:
    @staticmethod
    def create_profile_with_legacy_file(NPC_dict,
                                        Model_name:str = 'gpt-3.5-turbo-0125',
                                        CHIBI_name:Optional[str]= None):
        if CHIBI_name is None:
            CHIBI_name = NPC_dict['Name']

        Solid_memory = blocks.Information_piece_container(f'{CHIBI_name}\'s identity, occupation, habits and lifestyle.',
                {keyword:blocks.Information_piece(keyword, NPC_dict['Solid_Memory'][keyword], Model_name = Model_name) \
                                                         for keyword in NPC_dict['Solid_Memory'].keys()})
        Items = blocks.Thing_container(f'{CHIBI_name}\'s belongings', f'{CHIBI_name}\'s belongings',
                {item_name:blocks.Thing(item_name, NPC_dict['Items'][item_name], Model_name = Model_name) \
                                                         for item_name in NPC_dict['Items'].keys()})
        legacy_action_style = {keyword:blocks.Information_piece(keyword, NPC_dict['Action_style'][keyword], Model_name = Model_name)\
                                                         for keyword in NPC_dict['Action_style'].keys()}

        Action_style = blocks.Information_piece_container(f'{CHIBI_name}\'s style of acting and habit of taking action', legacy_action_style)
        return Profile_main_character(CHIBI_name,
                                      Current_situation     = NPC_dict['Current_situation'].format(Name = CHIBI_name),
                                      Items                 = Items,
                                      Solid_memory          = Solid_memory,
                                      Action_style          = Action_style,)
                                      #Relationship_memory   = Relationship_memory,
                                      #Experiences           = Experiences) 

    @staticmethod
    def create_memorystream_for_main_character(legacy_file):
        fluid_memory = legacy_file['Fluid_Memory']
        relation_memory = legacy_file['Relation_Memory']
        memory_list = []
        for type_memories in fluid_memory.values():
            for memory_piece_str in type_memories:
                memory_list.append(blocks.Memory_piece(memory_piece_str))
        for type_memories in relation_memory.values():
            for memory_piece_str in type_memories:
                memory_list.append(blocks.Memory_piece(memory_piece_str))
        tem_memory_stream = memory_stream.Memory_stream_CHIBI_main_character(memory_list)
        return tem_memory_stream
        