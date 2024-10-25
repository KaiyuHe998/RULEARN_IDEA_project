''' Memory stream system, 
# Each observation is named as an memory_piece 
# Memory stream is a container of such pieces and can retrieve these memory piece when needed effificiently
# Task:
  1, store memory_piece and: (1)create embeddings (2)assign importance scores using different metircs 
  2, organize information (3)perhaps do some pre-classification
  3, retrieve memory (4)retrival efficiently (5) reflect and form high level concept according to topics
'''
# basics
from typing import *
from abc import ABC, abstractmethod
import datetime
import numpy as np

# CHIBI framework components
import world_basic_blocks as blocks
import utils


class Memory_stream_Base(ABC):
    def __init__(Memories:List[blocks.Memory_piece],
                 Model_name:str = "text-embedding-ada-002",
                 Belongs_to:'CHIBI_Base' = None,):
        pass
        
    @abstractmethod
    def memory_retrieve(task:str):
        '''get most relevant memories based on task description'''
        pass

    @abstractmethod
    def memory_add(memory:blocks.Memory_piece):
        '''Add a new memory to list'''
        pass

    @abstractmethod
    def update():
        '''Do something when each frame passed'''
        pass
        
    @abstractmethod
    def memory_delete():
        '''Do something to forget some memories'''
        pass
        
class Memory_stream_CHIBI_main_character(Memory_stream_Base):
    def __init__(self,
                 Memories:List[blocks.Memory_piece],                 
                 Model_name:str = "text-embedding-ada-002",
                 Belongs_to:'CHIBI_main_character' = None,
                 Usage:'openai.openai_object.OpenAIObject' = None,
                 recency_decay_factor:float = 0.99,
                 emperical_weight:Optional[List[float]] = None,
                 Buffer_size:int = 7,):#[relevance_score_weight, rececy_score_weight, importance_score_weight]
        if emperical_weight is None:
            self.emperical_weight = [3,0.5,2]
        else:
            self.emperical_weight = emperical_weight
        self.Model_name = Model_name
        self.Memories = Memories
        self.Host_CHIBI = Belongs_to
        self.Usage = Usage 
        self.recency_decay_factor = recency_decay_factor
        # init all embeddings for all memories
        self.embed_all_memories()
        # init importance score for all memories
        for memory in self.Memories:
            self.assign_memory_importance_score(memory, mode = 'all_one')
        if Usage is None:
            self.Usage = {'completion_tokens': 0,   # returned by parse function
                          'prompt_tokens':0,            # openai usage
                          'total_tokens': 0} 
        else:
            self.Usage = Usage
        self.Buffer_memories = []
        self.Cur_assumption_and_plan = None
        self.Cur_plan = None # currently not in use

        self.Buffer_size = Buffer_size
        self.All_assumptions = [] # List[str]
        self.All_plans = [] # List[str]

    def memory_retrieve(self,
                        task:Optional[str] = None,
                        top_n:Optional[int] = None,
                        memory_retrieve_type = 'Most_relevant') -> List[blocks.Memory_piece]: # if top_n is none then return all memories
        # task_relational_score:  #range(0,1)
        if memory_retrieve_type == 'Most_relevant':
            task_relevance_scores = []
            if task is None:
                task_embedding = self.Host_CHIBI.Plan_system.Cur_state.Embedding
            else:
                task_embedding, new_usage, text = utils.get_embedding(task)
    
            for memory_piece in self.Memories:
                assert memory_piece.Embedding is not None, f'Piece memory:{memory_piece.Information} do not have embeddings'
                tem_task_relevance_score = utils.calculate_cosine_similiarity(memory_piece.Embedding,task_embedding)
                task_relevance_scores.append(tem_task_relevance_score)
            
            # time_importance_score: #range(0,1)
            time_recency_scores = []
            for memory_piece in self.Memories:
                time_passed_in_hour = memory_piece.get_time_passed_in_hour()
                time_recency_scores.append(np.power(self.recency_decay_factor,time_passed_in_hour))
                
            # memory importance_scores #range(0,1) 
            memory_importance_scores = [memory.Importance_score for memory in self.Memories]
    
            # calculate final score of all memories
            final_score_dict = {}
            for task_relevance_score, time_recency_score, memory_importance_score, memory_piece in \
                    zip(task_relevance_scores,time_recency_scores,memory_importance_scores,self.Memories):
                relevance_weight = self.emperical_weight[0]
                recency_weight = self.emperical_weight[1]
                importance_weight = self.emperical_weight[2]
                final_score = task_relevance_score*relevance_weight +\
                                time_recency_score*recency_weight + memory_importance_score * importance_weight
                final_score_dict.update({memory_piece:final_score})
    
            if top_n is None:
                top_n = len(final_score_dict)
            sorted_top_n = sorted(final_score_dict.items(), key=lambda item: item[1],reverse = True)[:top_n]
            sorted_memories = [i[0] for i in sorted_top_n]
            
            return_list = [] # return the memories in the time order
            for i in self.Memories:
                if i in sorted_memories:
                    return_list.append(i)
            return return_list
            
        elif memory_retrieve_type == 'Most_recent':
            if top_n is None or top_n > len(self.Memories):
                top_n = len(self.Memories)
            return self.Memories[-top_n:]
        else:
            assert False, f'{memory_retrieve_type} is not a supported method for retrieving memories!'
                    
    def memory_add(self,
                   memory_to_be_added:Union[blocks.Memory_piece,str, blocks.Assumption],
                   Memory_type:Optional[str] = None,
                   Importance_score:Optional[int] = None,):
        if isinstance(memory_to_be_added, str):
            # input is string need to create a new experiece_piece with given string
            memory_to_be_added = blocks.Memory_piece(memory_to_be_added, Memory_type = Memory_type, Importance_score = Importance_score)
            memory_to_be_added.Time_passed = datetime.timedelta(minutes = 0)
            if Importance_score is None:
                self.assign_memory_importance_score(memory_to_be_added, mode = 'constant_importance_socre')
            
        elif isinstance(memory_to_be_added, blocks.Memory_piece) or isinstance(memory_to_be_added, blocks.Assumption):
            if memory_to_be_added.Time_passed != datetime.timedelta(minutes = 0):
                memory_to_be_added.Time_passed == datetime.timedelta(minutes = 0)
            if memory_to_be_added.Importance_score is None:
                self.assign_memory_importance_score(memory_to_be_added, mode = 'constant_importance_socre')
        else:
            assert False, f'type of memory_type_to_be_added is not supported, input type is: {type(memory_to_be_added)}'
        if memory_to_be_added.Embedding is None:
            self._embed_single_memory_piece(memory_to_be_added)
    
        self.Buffer_memories.append(memory_to_be_added)
        # every time when a memory_piece is added to memory stream will need to check if the threshold reaches the threshold, if reaches the threshold we will need to reflect
        # if len(self.Buffer_memories) >= self.Buffer_size: # need to do abduction or induction and and flushing the memories
        #     if self.Host_CHIBI.Do_abduction:
        #         self.abduction_loop()
        #     else:
        #         self.flushing_buffer()
            

    def flushing_buffer(self):
        # Add all recent memory into the state information
        all_buffer_observations = [i for i in self.Buffer_memories if i.Memory_type == 'Observation']
        all_buffer_observations_strings = [i.get_information() for i in all_buffer_observations]
        next_row_str = '\n\n'
        buffer_memories_str = f'''{next_row_str.join(all_buffer_observations_strings)}'''
        self.Host_CHIBI.Plan_system.Cur_state.edit(buffer_memories_str)
        self.Memories.extend(all_buffer_observations)
        self.Buffer_memories = []
        
    def memory_delete(self,
                      memory_to_be_delete:Union[blocks.Memory_piece, blocks.Assumption]):
        assert memory_to_be_delete in self.Memories, f'the memory to be delted {memory_to_be_delete.get_information()} is not in memory list.'
        self.Memories.remove(memory_to_be_delete)

    def abduction_loop(self, 
                  top_n:Optional[int] = None,): # Use all memeories
        # based on observations(recency, importance, relevance), agent need to generate the hypothesis that could solve or explain the observations

        previous_relevant_memory_str, buffer_memory_str, storage_information, _, _ = self.Host_CHIBI.retrieve_prompt_information(memory_use = top_n)
        assert buffer_memory_str!='' or previous_relevant_memory_str != '', f'Not enough information to be abduct on, this function should not be called without new observations' + '\n\n'
        # prompt function to get hypothesis
        if self.Cur_assumption_and_plan is None:
            logging_label = 'Abduction'
        else:
            logging_label = 'Induction'
        action_space = self.Host_CHIBI.Plan_system.get_action(return_list = True)
        action_space_str_list = [f'{index+1}th action: '+action[0].get_information() for index,action in enumerate(action_space)]
        action_space_str = '\n'.join(action_space_str_list) + '\n'

        if self.Cur_assumption_and_plan is None: # create a new abduction rule in the very beginning with limited observations (Abduction & Deduction)
            if self.Host_CHIBI.Special_label is None:
                Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()} Your task is to develop a general, clearly falsifiable, explanatory rules that explains the observed phenomena, a process known as abduction. Please consider the given observations and propose an initial hypothesis that explains them, make sure your hypothesis is robust and align with all your observations. Your response should include your current hypothesis and your planned actions.'''
            elif self.Host_CHIBI.Special_label == 'Reactor_puzzles':
                Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()} Your task is to formulate a hypothesis based on the reactions you observe. Please use the given observations to propose an initial rule that explains all reactions observed. Ensure your hypothesis is robust and consistent with these reactions. Next, describe your plan for further verification: which two materials from the following list will you use to test your hypothesis? Available materials: {storage_information}. Your response should include your current hypothesis and your planned actions. You can choose from the following actions:\n{action_space_str}'''
            elif self.Host_CHIBI.Special_label == 'Art_gallery_puzzles':
                Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()} Your task is to formulate a hypothesis explaining how the password for the <Code Secured Door> relates to all the paintings in the gallery. Consider the observations provided and propose an initial hypothesis that accounts for your findings. Ensure your hypothesis is robust and consistent with all observations. Next, describe your plan for further verification: What password do you want to input to the <Code secured door>, if there is any gallery you haven't checked will you go and investigate those galleries? Your response should include your current hypothesis and your planned actions. You can choose from the following actions:\n{action_space_str}'''
                
            elif self.Host_CHIBI.Special_label == 'Function_operator_puzzles':
                Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()} Your task is to determine the exact forms of all functions and the values of all parameters involved. First, focus on your observations to identify how many terms are in each function, the parameters within each, and any possible sub-functions involved in this puzzle. Then, hypothesize the actual forms of each function, including the values of constants and coefficients. Next, describe your plan for further verification, what value would you want to assign to which function, or do you want to input the password to the <Code secured door> to test your current result. Your response should include your current hypothesis and your planned actions. You can choose from the following actions:\n{action_space_str}'''
            Input = f'''{previous_relevant_memory_str}\n\n{buffer_memory_str}'''
                
        else: # already have a previous hypothesis and need to modify or change previous hypothesis based on new observations (Induction & Deduction)
            Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()} Your task is to validate and modify your previous hypothesis, detailed here: {self.get_assumption()}, using your new observations. Review your most recent observation: {buffer_memory_str}, to determine if your current hypothesis is still valid. If it is, describe the next steps you plan to take towards your goal. If it is not, revise your hypothesis to accurately reflect all observations, both recent and prior. Finally, provide a plan for your next steps. Your response should include both your current hypothesis and your planned actions. You can choose from the following actions:\n{action_space_str}'''
            Input = f'''{previous_relevant_memory_str}'''

        parse_function_str = None
        new_assumption_str = self.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = logging_label)
        self.Cur_assumption_and_plan = blocks.Assumption(new_assumption_str, Belongs_to = self)
        self.All_assumptions.append(new_assumption_str)
        self.flushing_buffer()
        self.memory_add(f'You just updated your plan and your hypothesis following is your new hypothesis and plan: {new_assumption_str}')

    # def deduction(self): (Cobined into abduction loop saving tokens and api calls)
    #     # generate a plan to guide the action selection in the following steps, this step is not the generated by the agent 
    #     assert self.Cur_assumption_and_plan is not None, f'''To generate a plan we need to first have an assumption, but {self.Host_CHIBI.Name} currently do not have an assumption but want to form a plan'''
    #     previous_relevant_memory_str, buffer_memory_str, _, _, _ = self.Host_CHIBI.retrieve_prompt_information()
            
    #     Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()} You have just formulated a new assumption: {self.get_assumption()} Now, your task is to develop a plan to either validate this new assumption or use it to achieve your goal. When creating your new plan, please review all your previous observations carefully to ensure that you utilize as much of the previous information as possible and avoid planning actions that you have already undertaken.'''
        
    #     Input = f'''{previous_relevant_memory_str}\n\n{buffer_memory_str}'''
    #     logging_label = 'Deduction'
    #     parse_function_str = None
    #     new_plan_str = self.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = logging_label)
    #     new_plan = blocks.Plan_piece(new_plan_str, Belongs_to = self)
    #     self.Cur_plan = new_plan
    #     self.All_plans.append(new_plan_str)
        
    def get_assumption(self)->str:
        if self.Cur_assumption_and_plan is None:
            return f'''{self.Host_CHIBI.Name} do not have any assumptions now, need to explore the environment to get more information first.'''
        else:
            return self.Cur_assumption_and_plan.get_information(self.Host_CHIBI)
            
    def update(self, 
               time_frame:datetime.timedelta): # in minutes
        for memory in self.Memories:
            memory.Time_passed += time_frame

    def assign_memory_importance_score(self,
                                       memory_piece:Union[blocks.Memory_piece, blocks.Assumption],
                                       mode:str = 'Initialization',
                                       important_constant_score:float = 4): #(Currently not in use)
        '''Need to use profile and goal of the CHIBI to construct pormpt to get importance score, 
            also need to make sure the generated score is between 0~10'''
        if mode == 'constant_importance_socre':
            # if setting to constance the improtance of assumption is double the importance of observation (test)
            if isinstance(memory_piece, blocks.Assumption):
                memory_piece.Importance_score = 2*important_constant_score
            else:
                memory_piece.Importance_score = important_constant_score
            
        elif mode == 'Initialization':
            for memory in self.Memories:
                memory.Importance_score = important_constant_score
        else:
            assert False, 'mem importance score mode not supported'

    # embedding related 
    def embed_all_memories(self):
        for memory_piece in self.Memories:
            if memory_piece.Embedding is None:
                self._embed_single_memory_piece(memory_piece)
                    
    def _embed_single_memory_piece(self, 
                                   memory_piece:blocks.Memory_piece):
        assert memory_piece.Embedding is None, 'memory piece already have assigned embedding'
        embedding, new_usage, text = utils.get_embedding(memory_piece.Information)
        memory_piece.Embedding = embedding

    # Prompt methods---------------------------------------------
    def _assign_memory_importance_score(self,
                                     memory:blocks.Memory_piece):
        pass
        
    def get_recent_activities(self,
                              time_duration_hour:Union[int,datetime.timedelta], # in hour
                              viewer:Optional[Any] = None,
                             )->List[str]:
        if isinstance(time_duration_hour, int):
            time_duration_hour = datetime.timedelta(time_duration_hour)
        memory_pieces_within_duration = [memory_piece.get_information(viewer) for memory_piece in self.Memories if memory_piece.Time_passed <= time_duration_hour]
        return memory_pieces_within_duration

class Memory_stream_Human(Memory_stream_Base):
    def __init__(self,
                 Memories:List[blocks.Memory_piece],                 
                 Model_name:str = "text-embedding-ada-002",
                 Belongs_to:'CHIBI_human' = None,
                 Usage:'openai.openai_object.OpenAIObject' = None,
                 recency_decay_factor:float = 0.99,
                 emperical_weight:Optional[List[float]] = None,
                 Buffer_size:int = 0):#[relevance_score_weight, rececy_score_weight, importance_score_weight]
        if emperical_weight is None:
            self.emperical_weight = [3,0.5,2]
        else:
            self.emperical_weight = emperical_weight
        self.Model_name = Model_name
        self.Memories = Memories
        self.Host_CHIBI = Belongs_to
        self.Usage = Usage 
        self.recency_decay_factor = recency_decay_factor
        # init importance score for all memories

        if Usage is None:
            self.Usage = {'completion_tokens': 0,   # returned by parse function
                          'prompt_tokens':0,            # openai usage
                          'total_tokens': 0} 
        else:
            self.Usage = Usage
        self.Buffer_memories = []
        self.Cur_assumption_and_plan = None
        self.Cur_plan = None
        self.Buffer_size = Buffer_size
        self.All_assumptions = [] # List[str]
        self.All_plans = [] # List[str]

    def memory_retrieve(self,
                        task:Optional[str] = None,
                        top_n:Optional[int] = None,
                        memory_retrieve_type = 'Most_relevant') -> List[blocks.Memory_piece]: # if top_n is none then return all memories

        if top_n is None or top_n > len(self.Memories):
            top_n = len(self.Memories)
        return self.Memories[-top_n:]

                    
    def memory_add(self,
                   memory_to_be_added:Union[blocks.Memory_piece,str, blocks.Assumption],
                   Memory_type:Optional[str] = None,
                   Importance_score:Optional[int] = None,):
        if isinstance(memory_to_be_added, str):
            # input is string need to create a new experiece_piece with given string
            memory_to_be_added = blocks.Memory_piece(memory_to_be_added, Memory_type = Memory_type, Importance_score = Importance_score)
            memory_to_be_added.Time_passed = datetime.timedelta(minutes = 0)
            if Importance_score is None:
                self.assign_memory_importance_score(memory_to_be_added, mode = 'constant_importance_socre')
            
        elif isinstance(memory_to_be_added, blocks.Memory_piece) or isinstance(memory_to_be_added, blocks.Assumption):
            if memory_to_be_added.Time_passed != datetime.timedelta(minutes = 0):
                memory_to_be_added.Time_passed == datetime.timedelta(minutes = 0)
            if memory_to_be_added.Importance_score is None:
                self.assign_memory_importance_score(memory_to_be_added, mode = 'constant_importance_socre')
        else:
            assert False, f'type of memory_type_to_be_added is not supported, input type is: {type(memory_to_be_added)}'

    
        self.Buffer_memories.append(memory_to_be_added)
        # Human interface do not need reflection        
        
    def memory_delete(self,
                      memory_to_be_delete:Union[blocks.Memory_piece, blocks.Assumption]):
        assert memory_to_be_delete in self.Memories, f'the memory to be delted {memory_to_be_delete.get_information()} is not in memory list.'
        self.Memories.remove(memory_to_be_delete)

        
    def get_assumption(self)->str:
        if self.Cur_assumption_and_plan is None:
            return f'''{self.Host_CHIBI.Name} do not have any hypothesis now, need to explore the environment to get more information first.'''
        else:
            return_str = self.Cur_assumption_and_plan.get_information(self.Host_CHIBI)
            return utils.decorate_text_with_color(return_str, 'magenta', bold = True)
            
    def update(self, 
               time_frame:datetime.timedelta): # in minutes
        for memory in self.Memories:
            memory.Time_passed += time_frame

    def flushing_buffer(self):
        # Add all recent memory into the state information
        all_buffer_observations = [i for i in self.Buffer_memories if i.Memory_type == 'Observation']
        all_buffer_observations_strings = [i.get_information() for i in all_buffer_observations]
        next_row_str = '\n\n'
        buffer_memories_str = f'''{next_row_str.join(all_buffer_observations_strings)}'''
        #self.Host_CHIBI.Plan_system.Cur_state.edit(buffer_memories_str)
        self.Memories.extend(all_buffer_observations)
        self.Buffer_memories = []
        
    def abduction_loop(self, top_n:Optional[int] = None):
        previous_relevant_memory_str, buffer_memory_str, storage_information, _, _ = self.Host_CHIBI.retrieve_prompt_information(memory_use = top_n)
        assert buffer_memory_str!='' or previous_relevant_memory_str != '', f'Not enough information to be abducted on, this function should not be called without new observations' + '\n\n'
        # prompt function to get hypothesis
        if self.Cur_assumption_and_plan is None:
            logging_label = 'Abduction'
        else:
            logging_label = 'Induction'
        action_space = self.Host_CHIBI.Plan_system.get_action(return_list = True)
        action_space_str_list = [f'{index+1}th action: '+action[0].get_information() for index,action in enumerate(action_space)]
        action_space_str = '\n'.join(action_space_str_list) + '\n'

        if self.Cur_assumption_and_plan is None: # create a new abduction rule in the very beginning with limited observations (Abduction & Deduction)
            if self.Host_CHIBI.Special_label is None:
                Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()}\nTo begin with your task is to develop a general, clearly falsifiable, explanatory rules that explains the observed phenomena, a process known as abduction. Please consider the given observations and propose an initial hypothesis that explains them, make sure your hypothesis is robust and align with all your observations. Your response should include your current hypothesis and your planned actions. You can choose from the following actions:\n{action_space_str}'''
            elif self.Host_CHIBI.Special_label == 'Reactor_puzzles':
                Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()}\n{utils.decorate_text_with_color("To begin with your task is to formulate a hypothesis based on the reactions you observe. Please use the given observations to propose an initial rule that explains all reactions observed. Ensure your hypothesis is robust and consistent with these reactions. Next, describe your plan for further verification: which two materials from the following list will you use to test your hypothesis? Your response should include your current hypothesis and your planned actions.",'cyan',bold = True,deep = True)} Available materials: {storage_information} You can choose from the following actions:\n{action_space_str}'''
            elif self.Host_CHIBI.Special_label == 'Art_gallery_puzzles':
                Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()}\n{utils.decorate_text_with_color("To begin with your task is to formulate a hypothesis explaining how the password for the <Code Secured Door> relates to all the paintings in the gallery. Consider the observations provided and propose an initial hypothesis that accounts for your findings. Ensure your hypothesis is robust and consistent with all observations. Next, describe your plan for further verification: What password do you want to input to the <Code secured door>, if there is any gallery you haven't checked will you go and investigate those galleries? Your response should include your current hypothesis and your planned actions.", 'cyan',bold = True, deep = True)} You can choose from the following actions:\n{action_space_str}'''
                
            elif self.Host_CHIBI.Special_label == 'Function_operator_puzzles':
                Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()}\n{utils.decorate_text_with_color("To begin with your task is to determine the exact forms of all functions and the values of all parameters involved. First, focus on your observations to identify how many terms are in each function, the parameters within each, and any possible sub-functions involved in this puzzle. Then, hypothesize the actual forms of each function, including the values of constants and coefficients. Next, describe your plan for further verification, what value would you want to assign to which function, or do you want to input the password to the <Code secured door> to test your current result? Your response should include your current hypothesis and your planned actions.",'cyan', bold = True,deep = True)} You can choose from the following actions:\n{action_space_str}'''
            Input = f'''{buffer_memory_str}'''
                
        else: # already have a previous hypothesis and need to modify or change previous hypothesis based on new observations (Induction & Abduction)
            prompt_str = f"Your task is to validate and modify your previous hypothesis, detailed here: {self.get_assumption()}, using your new observations. Review your most recent observation: {buffer_memory_str}, to determine if your current hypothesis is still valid. If it is, describe the next steps you plan to take toward your goal. If it is not, revise your hypothesis to accurately reflect all observations, both recent and prior. Finally, provide a plan for your next steps. Your response should include both your current hypothesis and your planned actions. You can choose from the following actions:\n{action_space_str}"
            Prompt = f'''{self.Host_CHIBI.Plan_system.Cur_state.get_information()}\n{utils.decorate_text_with_color(prompt_str,'cyan',bold = True, deep = True)}'''
            Input = f'''{previous_relevant_memory_str}'''
            
        new_assumption_str = self.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = None, logging_label = logging_label)

        self.Cur_assumption_and_plan = blocks.Assumption(new_assumption_str, Belongs_to = self)
        self.All_assumptions.append(new_assumption_str)
        self.flushing_buffer()
        

    # def deduction(self): # Combined with abduction loop
    #     def _prompt_template():
    #         Prompt = f'''Following is your current assumption:{self.get_assumption()}'''
    #         Input = f'''Please state roughly in 20 words of whether you plan to conduct more experiments to verify your assumption or use it to achieve your goal.'''
    #         return Prompt, Input, None, 'Deduction'
            
    #     Prompt, Input, parse_function_str, logging_label = _prompt_template()
    #     new_plan_str = self.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = logging_label)

    #     new_plan = blocks.Plan_piece(new_plan_str, Belongs_to = self)
    #     self.Cur_plan = new_plan
    #     self.All_plans.append(new_plan_str)
                
        

    def assign_memory_importance_score(self, 
                                       memory_piece:Union[blocks.Memory_piece, blocks.Assumption],
                                       mode:str = 'Initialization',
                                       important_constant_score:float = 4):#(Currently not in use)
        '''Need to use profile and goal of the CHIBI to construct pormpt to get importance score, 
            also need to make sure the generated score is between 0~10'''
        if mode == 'constant_importance_socre':
            # if setting to constance the improtance of hypothesis is double the importance of observation (test)
            if isinstance(memory_piece, blocks.Assumption):
                memory_piece.Importance_score = 2*important_constant_score
            else:
                memory_piece.Importance_score = important_constant_score
            
        elif mode == 'Initialization':
            for memory in self.Memories:
                memory.Importance_score = important_constant_score
        else:
            assert False, 'mem importance score mode not supported'

    # embedding related 
    def embed_all_memories(self):
        pass
                    
    def _embed_single_memory_piece(self, 
                                   memory_piece:blocks.Memory_piece):
        pass
    # Prompt methods---------------------------------------------
    def _assign_memory_importance_score(self,
                                     memory:blocks.Memory_piece):
        pass
        
    def get_recent_activities(self,
                              time_duration_hour:Union[int,datetime.timedelta], # in hour
                              viewer:Optional[Any] = None,
                             )->List[str]:
        if isinstance(time_duration_hour, int):
            time_duration_hour = datetime.timedelta(time_duration_hour)
        memory_pieces_within_duration = [memory_piece.get_information(viewer) for memory_piece in self.Memories if memory_piece.Time_passed <= time_duration_hour]
        return memory_pieces_within_duration

