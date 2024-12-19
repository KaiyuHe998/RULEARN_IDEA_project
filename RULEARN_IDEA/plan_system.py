# basics
from abc import ABC, abstractmethod
from typing import *
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import random
import datetime
import numpy as np

# CHIBI framework components
import world_basic_blocks as blocks
import utils
import narrators
from Judger import Judger
import fixed_interactive_pipeline_objects as fixed_blocks


# --------------------------- ABC component for plan system --------------------------------
# --------------------------- ABC component for plan system --------------------------------

class Attemptation_Action_Base(blocks.CHIBI_Object, ABC):
    '''In the future may add more action types including cooperation ......'''
    def __init__(self,
                 Information:str,
                 Belongs_to:'State_Simple',
                 Cost:Optional[int] = None,
                 FOD_score:Optional[int] = None,
                 Embedding:Optional[List[float]] = None,
                 Init_time:Optional[datetime.datetime] = None,
                 recency_decay_factor:float = 0.99,
                 path_penalty_cost_weight:float = 0.1,
                 Model_name:str = 'gpt-3.5-turbo-1106',
                 Usage:Optional[Dict[str,int]] = None,
                 Impression_object:Optional[blocks.Object_Impression] = None,
                ):
        if Usage is None:
            self.Usage = {'completion_tokens':0,
                          'prompt_tokens':0,
                          'total_tokens':0}
        else:
            self.Usage = Usage
        self.Model_name = Model_name
        self.Belongs_to = Belongs_to
        self.Cost = Cost
        self.FOD_score = FOD_score # feeling of deprivation
        self.Information = Information
        self.Success_fail_state = None
        self.Tried_times = 1
        self.Space_move_aware_decoration = None
        
        if Init_time is None:
            self.Init_time = datetime.datetime.now()
        else:
            self.Init_time = Init_time
        self.recency_decay_factor = recency_decay_factor
        self.path_penalty_cost_weight = path_penalty_cost_weight
        self.Host_CHIBI = self.Belongs_to.Host_CHIBI

        if self.Host_CHIBI.CHIBI_type == 'GPT_agent':
            if Embedding is None:
                self.update_embedding()
            else:
                self.Embedding = Embedding
        

    def update_embedding(self):
        self.Embedding = utils.get_embedding(self.get_information())[0]
        
    @abstractmethod
    def get_relative_objects(self)->List[blocks.CHIBI_Object]:
        '''only return REAL object that can be edited or delete'''
        pass
        
    @abstractmethod
    def __call__(self):
        '''when an action is been taken(called)'''
        '''should be called only in CHIBI_Base.take_action()'''
        '''Should follow exact the following steps
        # step1, decide if the action need interation, if need interaction call the interact() function the intend 
        # step2, based on judge result, if action is success then parse the enviromnment information based on the object, if fail use prompt to decide if generating a new attempt based on fail experience
        # step3, generate a sentence of the action, and edit the information to the state, also handle the memory strem and other stuffs'''
        pass
        
    @abstractmethod
    def cost_function(self)->int: # need to return a number in range (0,1)
        '''generate cost of taking this action, this cost can be changed by character's status and environment information'''
        if self.Cost is not None:
            return self.Cost
        else:
            return 10

    @abstractmethod
    def FOD_function(self)->int: # need to return a number in range (0,1)
        '''generate the expectation of finish the plan by taking this action
           This FOD is only related to a single plan node, not the root goal plan'''
        if self.FOD_score is not None:
            return self.FOD_score
        else:
            return 10

    @abstractmethod
    def callable(self)->bool:
        '''decide if current action is callable or not '''

    def get_keyword(self):
        print(f'This class {type(self)} don\'t have a keyword, please use get_information instead')
        return self.Information
        
    def decoreate_space_move_information(self):
        is_callable = self.callable()
        if not is_callable[0] and is_callable[1] == 'Space_error':
            cur_space = self.Host_CHIBI.Space_manager.Cur_position
            self.Space_move_aware_decoration = f' You are currently at {cur_space}, to do this you will need to go to {self.Impression_object.Impression_space.Space_name} first.'
            self.Embedding = utils.get_embedding(self.get_information())[0]
            
    def get_information(self, viewer:blocks.CHIBI_Object = None):
        return_information = self.Information
        # if self.Tried_times != 1:
        #     tried_str = None
        #     if self.Tried_times == 1:
        #         tried_str = 'first'
        #     elif self.Tried_times == 2:
        #         tried_str = 'second'
        #     elif self.Tried_times == 3:
        #         tried_str = 'third'
        #     else:
        #         tried_str = f'{self.Tried_times}th'
        #     return_information = f'''Try {return_information} again, this is going to be your {tried_str} times.'''
        if self.Space_move_aware_decoration is not None:
            return_information += self.Space_move_aware_decoration
        return return_information 

    # prompt functions----------------------------------------------------------
    def _get_cost_prompt_input_function(self): # Currently not in use
        @utils.Prompt_constructor_for_system(self.Host_CHIBI.Model_name,
                                             Usage = self.Host_CHIBI.Usage,
                                             parse_function_str = 'str_with_tuple')
        def prompt_and_input():
            Prompt = f'''lease rate the following action plan in terms of the specific task information, taking into account the cost of realization, the ease of realization, the time required, and the possible risks, please refer to the detail of the object to give a reasonalble score. Scores range from 0 to 10. When generating the final score, please give your answer in parentheses and Arabic numerals, e.g., a score of 2 is a (2), a score of 10 is a (10), and finally, please output your answer after the <Answer> sign. After the reasoning, the specific template is <Answer>:(2)'''
            Input = f'Character\'s information: {self.Belongs_to.get_information()}, current planned action: {self.get_information()}, and the detailed information of the object is {self.Impression_object.get_information()}. Please rate the above actions according to the above instructions, with scores ranging from 0 (very simple: no cost or effort) to 5 (normal difficulty, similar to homework, etc., which takes hours of time and physical or mental effort) to 10 (very difficult which takes a lot of time, physical, resource, and mental effort). Please combine the above rating scale to give a score of 0 to 10 points, and finally use parentheses wrapped in arabic numerals in the form of the answer, for example, if your answer is 3, then print <Answer>: (3) Please only have one parentheses included in your final answer for the final comprehensive cose, only the final score have the parentheses, if you fail to do so I will not tip you.'
            return Prompt, Input
        return prompt_and_input
        
    def _get_FOD_prompt_input_function(self): # Currently not in use
        @utils.Prompt_constructor_for_system(self.Host_CHIBI.Model_name,
                                             Usage = self.Host_CHIBI.Usage,
                                             parse_function_str = 'str_with_tuple')
        def prompt_and_input():
            Prompt = f'''Please evaluate the intensity of the character's desire for the action. Assess this in terms of three dimensions:

1. **Knowledge Competence**: To what extent does the action potentially increase the character's understanding or skill? Consider the character's perceived gap in knowledge or competence and how the action might fill this gap.

2. **Emotional Impact**: Evaluate the level of anxiety, dissatisfaction, or restlessness the character experiences when unable to pursue the action. How significant is the emotional impact of not obtaining what the action might provide?

3. **Goal Alignment**: Consider how well the action aligns with the character's immediate goals in their current situation. Does the action significantly advance these goals, or is it only marginally relevant?

Based on these dimensions, rate how much the character desires this action on a scale from 0 to 10. Provide your score in parentheses and Arabic numerals. For example, a moderate level of desire might be expressed as (5), while an intense, overwhelming desire could be (10). The higher the score, the stronger the character's desire.

After presenting your reasoning, format the response as follows:
<Explanation of your assessment>
Desire Score: (Your Score)'''


            Input = f'''Character's Specific Information: {self.Belongs_to.get_information()}. How much do you think {self.Host_CHIBI.Name} desires the action: {self.get_information()}? Please rate the action based on the following criteria, with scores ranging from 0 to 10:

- Score 0: The action provides absolutely no useful information, the character does not feel any distress or dissatisfaction due to insufficient information, and the action is completely irrelevant to their current situation and goals.

- Score 5: The action might provide some useful information, but it's not certain. The character feels a certain degree of dissatisfaction or unrest when information is unavailable or insufficient, but this feeling is controllable. There is a sense of tension or urgency when needing to solve a problem, but it doesn't necessarily drive them to act.

- Score 10: The information provided by the action is extremely useful. The character shows a strong level of dissatisfaction and impatience when information cannot be obtained or is insufficient, and this feeling may drive them to take additional actions to acquire the needed information. Facing a problem, they feel intense tension and urgency, and this significantly motivates their problem-solving behavior.

Please combine the above criteria to give a score between 0 and 10. Provide your answer in the format of a number enclosed in parentheses, for example, a score of 3 would be expressed as <Answer>:(3). Make sure to check that your response ends with <Answer>:(Arabic numeral) format.'''
            return Prompt, Input
        return prompt_and_input

class State_Base(blocks.CHIBI_Object,ABC):
    '''Base calss of high level node'''
    def __init__(self,
                 Information:str,
                 Editable:bool = True,
                 Belongs_to:Optional['Plan_system_Base'] = None,
                 Attemptations:Optional[List[Attemptation_Action_Base]] = None,
                 Cost:Optional[int] = None,
                 FOD_score:Optional[int] = None,
                 Embedding:Optional[List[float]] = None,
                ):
        self.Belongs_to = Belongs_to
        self.Attemptations = Attemptations
        self.Editable = Editable
        self.Cost = Cost
        self.FOD_score = FOD_score
        self.Host_CHIBI = self.Belongs_to.Host_CHIBI
        if Embedding is None:
            self.Embedding = utils.get_embedding(self.get_information(self.Host_CHIBI))[0] 
        else:
            self.Embedding = Embedding
        
    @abstractmethod
    def generate_perceptual_action_based_on_impressions(self,
                            cur_information:str)->List[Attemptation_Action_Base]:
        '''generate attempts for current plan with context fo current inforation'''
        return []

    @abstractmethod
    def cost_function(self)->int: 
        '''generate the cost of this action'''
        return 0

    @abstractmethod          
    def FOD_function(self)->int:
        '''Generating a FOD estimation for the distance to final goal after achieving this sub-goal'''
        return 0

    def get_keyword(self):
        print(f'This class {type(self)} don\'t have a keyword, please use get_information instead')
        return self.Information

    # edit function is already assigned from CHIBI object 
    @abstractmethod
    def generate_next_state(self):
        'Some important things happens such as some observation changed the goal of the CHIBI, or some important progress are made for the CHIBI so the CHIBI made it and goes to the next new state'
        pass

    @abstractmethod
    def add_attempt_action(self):
        pass

    def destory(self):
        self.Belongs_to.Attemptations.remove(self)
        self.Belongs_to = None

# --------------------------- Basic components for plan system --------------------------------
# --------------------------- Basic components for plan system --------------------------------

# Attemtptation actions ----------------------------------------------------------
class Attemptation_Perceptual_Action(Attemptation_Action_Base):
    def __init__(self,
                 Information:str,
                 Belongs_to:'State_Simple',
                 Perceptual_object:List[Union['CHIBI.CHIBI_Base',
                                          blocks.Thing,
                                          blocks.Thing_container]], # real object
                 Cost:Optional[int] = None,
                 FOD_score:Optional[int] = None,
                 Embedding:Optional[List[float]] = None,
                 Init_time:Optional[datetime.datetime] = None,
                 Usage:Optional[Dict[str,int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Impression_object:Optional[blocks.Object_Impression] = None,
                ):
        
        super().__init__(Information, Belongs_to, Cost = Cost, 
                         FOD_score = FOD_score,Embedding = Embedding,
                         Init_time = Init_time, Usage = Usage, Model_name = Model_name)
        self.Perceptual_object = Perceptual_object
        self.Host_CHIBI = self.Belongs_to.Belongs_to.Host_CHIBI
        if Impression_object is None:
            self.Impression_object = self.Host_CHIBI.Space_manager.find_impression_object(self.Perceptual_object)
        else:
            self.Impression_object = Impression_object

    #------------ Plan attemptation interfaces
    def get_relative_objects(self)->List[blocks.CHIBI_Object]:
        '''only return REAL object that can be edited or delete'''
        return [self.Perceptual_object]
        
    def cost_function(self)->int:
        CHIBI_cur_position = self.Host_CHIBI.Space_manager.Cur_position
        impression_space = self.Impression_object.Impression_space.Space_name
        if self.Cost is None:
            #init_cost
            self.Cost = 0
            #calculate real cost for current situation
            
            if CHIBI_cur_position != impression_space:
                
                path_penalty_cost = len(self.Host_CHIBI.Space_Manager_System_Global.find_path(CHIBI_cur_position,impression_space)) * self.path_penalty_cost_weight 
                return (self.Cost + path_penalty_cost)/10
                
            elif CHIBI_cur_position == impression_space:
                return self.Cost/10
            else:
                assert False, f'{self.get_information()} error occured when calculating the cost of this object'
        else:
            if CHIBI_cur_position != impression_space:
                path_penalty_cost = len(self.Host_CHIBI.Space_Manager_System_Global.find_path(CHIBI_cur_position,impression_space)) * self.path_penalty_cost_weight 
                return (self.Cost + path_penalty_cost)/10
            elif CHIBI_cur_position == impression_space:
                return self.Cost/10
            else:
                assert False, f'{self.get_information()} error occured when calculating the cost of this object'
   
    def FOD_function(self)->int:
        if self.FOD_score is None:
            # generated_result = self._get_FOD_prompt_input_function()()
            #self.FOD_score = int(generated_result['parsed_result'])
            self.FOD_score = 5
            return self.FOD_score/10
        else:
            return self.FOD_score/10
    
    #------------ CHIBI_object interfaces
    def interact_pipeline(self):
        assert False, f'''{type(self)} is not interactable'''
        
    def edit(self,
             edited_information:str):
        super().edit(edited_information)

    def show(self):
        print(f'{self.Information}')

    def get_keyword(self)->str:
        super().get_keyword()

    def destory(self):
        # TODO add deletion function in parent state node
        super().destory()
        
    def __call__(self):
        # step1: perceptual start get all information and update impression (impression not updated yet)
        # step2: want to carry or not 
        # step3: if not carry(object not as a tool) generate interactive actions
        is_callable = self.callable()
        impression_space = self.Impression_object.Impression_space.Space_name #the action space based on the impression of the object, where is the last time that CHIBI has seen this object
        self.Perceptual_object.before_perceptual_effect(self)
        def _call():
            # step1 look into this perpectual object and gather information from the object, add memory, edit state
            self.Success_fail_state = True
            self.Host_CHIBI.Cur_action = self.get_information()
            perceptual_object = self.Perceptual_object
            perceptual_result = self.Host_CHIBI.Perception.perceive(perceptual_object, mode = 'Identical')
            #print(f'perceptual_result of {perceptual_object.get_keyword()}: {perceptual_result}')
            # update thing impression
            self.Impression_object.edit(perceptual_result)
            step_1_experience = f'''{self.Host_CHIBI.Name} checked {perceptual_object.get_keyword()}, and found the result: {perceptual_result}'''
            self.Host_CHIBI.Memory_stream.memory_add(step_1_experience, Memory_type = 'Observation')

            # Step2 decide if further interact with the object if the object is free text object 
            if isinstance(self.Perceptual_object, fixed_blocks.Fixed_Interact_Pipeline_Object_Base):
                interact_with_object = True
            else:
                generate_interactive_object = self._decide_interact_function(perceptual_result)()
                interact_with_object = generate_interactive_object['parsed_result']
            if not interact_with_object:
                step_2_experience = f'''{self.Host_CHIBI.Name} think {perceptual_object.get_keyword()} is not helpful, so he decide to move on.'''
                self.Host_CHIBI.Memory_stream.memory_add(step_2_experience)# Do we need this to be recorded in the memory stream?
                self.Tried_times += 1
                return  # CHIBI think this object is not useful he won't interact with object further more but could generate action if the 
            else:

                if isinstance(self.Perceptual_object, fixed_blocks.Fixed_Interact_Pipeline_Object_Base):
                    #assert len(perceptual_object.Parse_pipeline_dict) != 0 , f'''{perceptual_object.get_keyword()} should have at lease one parse_dict or this object is not able to interact with'''
                    if len(perceptual_object.Parse_pipeline_dict) == 0:
                        pass
                    else:
                        for action_string in perceptual_object.Parse_pipeline_dict.keys():
                            # Add all actions into the action pool
                            new_interactive_action = Attemptation_Fixed_Interactive_Pipeline_Action(action_string, self.Belongs_to,perceptual_object)
                            self.Belongs_to.add_attempt_action(new_interactive_action)
                    self.Perceptual_object.after_perceptual_effect(self)
                        
                else:
                    generated_interact_intention_result = self._interact_plan_function(perceptual_result)()
                    interact_intention_str = generated_interact_intention_result['parsed_result']
                    step_3_experience = f'''{self.Host_CHIBI.Name} think {perceptual_object.get_keyword()} is helpful, here is {self.Host_CHIBI.Name}'s thought: {interact_intention_str}'''
                    self.Belongs_to.edit(step_3_experience)
                    self.Host_CHIBI.Memory_stream.memory_add(step_3_experience)
                    new_interactive_action = Attemptation_Interactive_Action(interact_intention_str, self.Belongs_to,perceptual_object) #TODO: do we need the impression object? yes, and the action space is based on the impression object
                    self.Belongs_to.add_attempt_action(new_interactive_action)
                self.Tried_times += 1
            

        if is_callable[0]:
            _call()
        else:
            if is_callable[1] == 'Space_error':
                # first move then take action
                self.Host_CHIBI.move(impression_space,mode = 'GO')
                # then take action
                _call()

            elif is_callable[1] == 'Object_error':
                assert False, f'Not implemented:{type(self)}'
                
    def callable(self)->tuple[bool, str]:
        CHIBI_cur_position = self.Host_CHIBI.Space_manager.Cur_position
        impression_space = self.Impression_object.Impression_space.Space_name
        Cur_space_real = self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict[CHIBI_cur_position]
        all_objects_in_this_space = []
        all_objects_in_this_space.extend(Cur_space_real.retrieve_item_in_this_space('Things'))
        all_objects_in_this_space.extend(Cur_space_real.retrieve_item_in_this_space('Edges'))
        all_objects_in_this_space.extend(Cur_space_real.retrieve_item_in_this_space('CHIBIs'))
        if CHIBI_cur_position != impression_space:
            return (False, f'Space_error')
        if self.Perceptual_object not in all_objects_in_this_space and self.Perceptual_object.Belongs_to not in all_objects_in_this_space:
            return (False, f'Object_error')
        return (True,) 
    # --------------------------------------prompt functions ***
    def _decide_interact_function(self, perceptual_result:str): # Currently not in use
        '''Used after the CHIBI get the information of the perceptual action and decide whether go further'''
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''Given the background information:\n{self.Belongs_to.get_information()}\nAnd considering the findings from {self.Host_CHIBI.Name}'s investigation of {self.Perceptual_object.get_keyword()}, where the detailed information is as follows:\n {perceptual_result}. Assess whether {self.Perceptual_object.get_keyword()} is relevant to {self.Host_CHIBI.Name}'s current situation. Please proceed with the following steps for your answer'''
            Input = f'''**Step1**: Evaluate if it is logical and rational for {self.Host_CHIBI.Name} to continue interacting with {self.Perceptual_object.get_keyword()}. Interaction may include, but is not limited to, passing through, using, taking away, opening, etc.\n**Step2** Consider a typical response in {self.Host_CHIBI.Name}'s situation. Would a person in this situation move forward and ignore the object, or continue to interact with it?\n**Finally** Summarize your answer. If you conclude that {self.Host_CHIBI.Name} will continue to interact with {self.Perceptual_object.get_keyword()}, please respond with: (True). If you believe {self.Host_CHIBI.Name} will tend to ignore the object or that {self.Perceptual_object.get_keyword()} is irrelevant to {self.Host_CHIBI.Name}'s situation, respond with: (False)'''
            return Prompt, Input
        return _prompt_and_input

    def _interact_plan_function(self, perceptual_result:str): # Currently not in use
        '''Need to give a brief plan of how to use this object and what is the effect you want to achieve'''
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = None)
        def _prompt_and_input():
            Prompt = f'''Given the background information: {self.Belongs_to.get_information()}, after get more detailed information about {self.Perceptual_object.get_keyword()}:{perceptual_result}\n{self.Host_CHIBI.Name} decide to interact with {self.Perceptual_object.get_keyword()}.'''
            Input = f'''Considering the perspective of {self.Host_CHIBI.Name}, could you provide a decisive and confident explanation of {self.Host_CHIBI.Name}'s intentions and motivations for choosing to interact with the specified object? You only need to generate one simple sentence to mention how {self.Host_CHIBI.Name} will interact with {self.Perceptual_object.get_keyword()} and {self.Host_CHIBI.Name}'s intention. And the action should strongly align with the {self.Host_CHIBI.Name}'s goal!'''
            return Prompt, Input
        return _prompt_and_input

    def _choose_interact_plan_function(self, perceptual_result:str): # for fixed pipeline object
        possible_actions = '\n'.join([f'{index+1}th action: '+ action for index,action in enumerate(self.Perceptual_object.Parse_pipeline_dict.keys())])
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_square_bracket')
        def _prompt_and_input():
            Prompt = f'''Given the background information: {self.Belongs_to.get_information(viewer = self.Host_CHIBI)}, after get more detailed information about {self.Perceptual_object.get_keyword()}:{perceptual_result}\n{self.Host_CHIBI.Name} decide to interact with {self.Perceptual_object.get_keyword()}. And the following are all optional actions {self.Host_CHIBI.Name} can take:\n{possible_actions}'''
            Input = f'''**Step1:** Review all the provided actions. Reflect on {self.Host_CHIBI.Name}'s current situation and goal to assess if each action is logical and appropriate.\n**Step2:** Choose the most logical action. Explain why this action is the best choice compared to the others, focusing on how it aligns with {self.Host_CHIBI.Name}'s goals and situation.\n**Finally** Indicate your selected action by placing its corresponding Arabic numeral in square bracket at the end. For example, if the third action is chosen, write [3].'''
            return Prompt, Input
        return _prompt_and_input

class Attemptation_Interactive_Action(Attemptation_Action_Base): # Free text interaction action(currently not in use)
    '''Decide to interact with the object, during the process will generate detailed action string and decide belonings need to use'''
    def __init__(self,
                 Information:str,
                 Belongs_to:'State_Simple',
                 Interactive_object:Union['CHIBI.CHIBI_Base',
                                     blocks.Thing,
                                     blocks.Thing_container],
                 Cost:Optional[int] = None,
                 FOD_score:Optional[int] = None,
                 Embedding:Optional[List[float]] = None,
                 Init_time:Optional[datetime.datetime] = None,
                 Model_name:Optional[str] = 'gpt-3.5-turbo-0125',
                 Usage:Optional[Dict[str,int]] = None, 
                 Impression_object:Optional[blocks.Object_Impression] = None,
                ):
        if Init_time is None:
            Init_time = datetime.datetime.now()
        super().__init__(Information, Belongs_to,Embedding = Embedding,
                         Cost = Cost, FOD_score = FOD_score,
                         Init_time = Init_time,  Usage = Usage, Model_name = Model_name)
        self.Interactive_object = Interactive_object
        self.Host_CHIBI = self.Belongs_to.Belongs_to.Host_CHIBI
        self.Used_belongings = []
        self.detailed_action_string = None
        self.Success_fail_state = None
        if Impression_object is None:
            self.Impression_object = self.Host_CHIBI.Space_manager.find_impression_object(self.Interactive_object)
        else:
            self.Impression_object = Impression_object

    #------------ State attemptation interfaces
    def get_relative_objects(self)->List[blocks.CHIBI_Object]:
        '''only return REAL object that can be edited or delete'''
        '''This should be called after interact'''
        return_object_list = []
        return_object_list.append(self.Interactive_object)
        return_object_list.extend(self.Used_belongings)
        return return_object_list
        
    def cost_function(self)->int:
        CHIBI_cur_position = self.Host_CHIBI.Space_manager.Cur_position
        impression_space = self.Impression_object.Impression_space.Space_name
        if self.Cost is None:
            # init cost value
            generated_result = self._get_cost_prompt_input_function()()
            self.Cost = int(generated_result['parsed_result'])

            #calculate real cost for current situation
            if CHIBI_cur_position != impression_space:
                path_penalty_cost = len(self.Host_CHIBI.Space_Manager_System_Global.find_path(CHIBI_cur_position,impression_space)) * self.path_penalty_cost_weight # 后续这个Cost应该要根据 Edge的信息来计算
                return (self.Cost + path_penalty_cost)/10
                
            elif CHIBI_cur_position == impression_space:
                return self.Cost/10
            else:
                assert False, f'{self.get_information()} error occurred when calculating the cost of this action'
        else:
            if CHIBI_cur_position != impression_space:
                path_penalty_cost = len(self.Host_CHIBI.Space_Manager_System_Global.find_path(CHIBI_cur_position,impression_space)) * self.path_penalty_cost_weight # 后续这个Cost应该要根据 Edge的信息来计算
                return (self.Cost + path_penalty_cost)/10
                
            elif CHIBI_cur_position == impression_space:
                return self.Cost/10
            else:
                assert False, f'{self.get_information()} error occurred when calculating the cost of this action'

    def FOD_function(self)->int:
        if self.FOD_score is None:
            generated_result = self._get_FOD_prompt_input_function()()
            self.FOD_score = int(generated_result['parsed_result'])
            return self.FOD_score/10
        else:
            return self.FOD_score/10
    
    #------------ CHIBI_object interfaces
    def edit(self,
             edited_information:str):
        super().edit(edited_information)

    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''

    def show(self):
        print(f'{self.Information}')

    def get_keyword(self)->str:
        super().get_keyword()

    def destory(self):
        # TODO add deletion function in parent state node
        super().destory()

    def __call__(self):
        '''when an action is been taken(called)'''
        '''should be called only in CHIBI_Base.take_action()'''
        '''Should follow exact the following steps
        # step1, decide if the action need interation, if need interaction call the interact() function the intend 
        # step2, based on judge result, if action is success then parse the enviromnment information based on the object, if fail use prompt to decide if generating a new attempt based on fail experience
        # step3, generate a sentence of the action, and edit the information to the state, also handle the memory strem and other stuffs'''
        is_callable = self.callable()
        impression_space = self.Impression_object.Impression_space.Space_name
        def _call():
            self.interact() # generate detailed action string and used belongings saved
            ###
            ###
            ###
            # Time used here should return a list of atom actions!
            ###
            ###
            ###
            # update memorystream and state
            action_success_fail_state, action_success_fail_reason = Judger.judge(self)
            self.Success_fail_state = action_success_fail_state
            self.Success_fail_reason = action_success_fail_reason
            experience_str = f'''You decide to {self.get_information()} and you {'Succeed!' if action_success_fail_state else 'failed due to the following reason:' + action_success_fail_reason}'''
            self.Host_CHIBI.Memory_stream.memory_add(experience_str)  # need to make a summarization
            self.Tried_times += 1

            ### for all related objects parse their result
            for single_related_object in self.get_relative_objects():
                single_related_object.interact_pipeline(self)

            if not self.Success_fail_state: # if the action fails, generate new follow up action, could be other type of actions like creative action or something else
                still_try = self._get_still_try_func(action_success_fail_reason)()['parsed_result']
                if still_try:
                    new_follow_up_interactive_action_str = self._get_new_attempt_func(action_success_fail_reason)()['parsed_result']
                    new_interactive_action = Attemptation_Interactive_Action(new_follow_up_interactive_action_str,
                                                                             self.Belongs_to,# State
                                                                             self.Interactive_object)
                    self.Belongs_to.add_attempt_action(new_interactive_action)
                else:
                    experience_string = f'''{self.Host_CHIBI.Name} think he lack enough tool or prerequisite to conduct his plan, {self.Host_CHIBI.Name} will leave {self.Interactive_object.get_keyword()} behind for now until he get useful tools or prerequisite.'''
                    self.Belongs_to.edit(experience_string)
                    
            self.Success_fail_state = True # Currently in freetext action, if the action fail will generate a new interact action based on the experience, but in the future this will need modification, so currently keep this Success_fail_state logic, If the Success_fail_state is True, then this action will not be called in the future anymore
            
        if is_callable[0]:
            _call()
        else:
            if is_callable[1] == 'Space_error':
                # first move
                self.Host_CHIBI.move(impression_space,mode = 'GO')
                #then call
                
                _call()
            elif is_callable[1] == 'Object_error':
                assert False,f'Object disbarred is not handled. disappeared object:{is_callable[2]} {type(self)}'
            else:
                assert False,f'callable return false information,{is_callable}'
        
    def callable(self):
        '''Only when CHIBI at that space and that space have this object can CHIBI call this action'''
        impression_space = self.Impression_object.Impression_space.Space_name
        CHIBI_cur_position = self.Host_CHIBI.Space_manager.Cur_position
        Cur_space_real = self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict[CHIBI_cur_position]
        all_objects_in_this_space = []
        all_objects_in_this_space.extend(Cur_space_real.retrieve_item_in_this_space('Things'))
        all_objects_in_this_space.extend(Cur_space_real.retrieve_item_in_this_space('Edges'))
        all_objects_in_this_space.extend(Cur_space_real.retrieve_item_in_this_space('CHIBIs'))
        if CHIBI_cur_position != impression_space:
            return (False, f'Space_error')
        if self.Interactive_object not in all_objects_in_this_space:
            return (False, f'Object_error', single_item) 
        return (True,)

    #prompt based functions--------------------------
    def interact(self)->Dict[str,Any]:
        '''This is a fine grind interact action, only action need to interact with other object could call this function, in the interaction Host_CHIBI should generate detailed action and decide to use belonings, and Judger will predict the result of the action and every object related to this action will parse based on the action'''
        # Step1: generate detailed action string verb
        backpack_objects = self.Host_CHIBI.Profile.Items.object_retrieve()
        generated_detailed_action_str = self._get_detailed_interaction_string(backpack_objects)()
        generated_action_result = generated_detailed_action_str['parsed_result']
        # TODO need a better way to find the items related to the process
        # if generated action with tools used the return should be tuple, if only action description, then the return is just string
        if len(backpack_objects) > 0:
            # Save all items selected to use in the interaction to Used_belongings and can be extracted by get_relative_objects function
            if isinstance(generated_action_result, tuple):
                detailed_action_str = generated_action_result[0]
                item_strs = generated_action_result[1:]
                for item_str in item_strs:
                    for item in backpack_objects:
                        object_keyword_str = item.get_keyword()
                        if item_str.lower() in object_keyword_str.lower():
                            self.Used_belongings.append(item)
            elif isinstance(generated_action_result, str):
                detailed_action_str = generated_action_result['parsed_result']
            else:
                assert False, f'generated action is not in the correct format {generated_action_result}: {type(generated_action_result)}' 
        else:
            # there are no object in backpack so the parsed result should not involve any object
            detailed_action_str = generated_action_result
        assert isinstance(detailed_action_str, str), f'detailed_action_str passed into judge should be a string but got {type(detailed_action_str)} instead!'
        
        # judge_result = Judger.judge(self,detailed_action_str)
        self.detailed_action_string = detailed_action_str
        # # TODO generate new item and other parse things
        # return judge_result a
        
    def _get_detailed_interaction_string(self, backpack_objects:List[blocks.Thing]):
        interactive_object_information = self.Interactive_object.get_information(self.Host_CHIBI)
            
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            if len(backpack_objects) > 0:
                backpack_str = '\n'.join([i.get_keyword()+':'+i.get_information() for i in backpack_objects])
            else:
                backpack_str = f'{self.Host_CHIBI.Name} currently do not have any items with him.'
            
            Prompt = f'''Considering the background information: {self.Belongs_to.get_information()}\nAnd the current plan of {self.Host_CHIBI.Name}: {self.get_information()}\nPlease carefully think about your next steps. Describe in detail the physical actions required to execute your plan. For each action, specify the amount of force needed, the gestures involved, and how you will use any tools at your disposal. Focus on the physical aspects, such as where and how you place your hands and the exertion of strength required. Remember, your description should only cover the actions you intend to perform, without assuming any outcomes and physicial feedback of these actions. Just focus on the motions! You generated action should closely refer to the format and style of the sample answers. If the action is trivial and simple: like pickup something you could just describe it with a simple sentence.'''

            Prompt += f'''Step 1: Review the object you need to interact with: {self.Impression_object.get_keyword()}: {self.Impression_object.get_information()}, your description should not fake detailes mentioned above. Think reasonablely about the {self.Impression_object.get_keyword()} reasonablely and think how common human would normally do to achieve your intention.\n\n'''
            Prompt += f'''Step 2: Consider the items you have with you: {backpack_str}. Identify key items from your backpack that will most effectively help you achieve your objective if you need. How will you use this single item to fulfill your goal? If you think none of them are helpful you can just generate a tuple with only the action string. Please make sure the item name in the tuple should exactly the same in the angle brackets "<>".\n\n'''
            if len(backpack_objects) > 0:
                Prompt += f'''Finally, articulate your plan as a single, detailed action. Describe this action in a tuple where the first element of the tuple is the generated action, and the following element of the tuple is the selected items with {self.Host_CHIBI.Name}. The action should only be one attempt action nor rely on any feedback, please don't imagine the outcome. Your description should strictly avoids any mention of psychological activities, intentions, and plans – essentially, anything that can't be directly observed. Don't use adjectives whose physical meaning is unclear e.g. careful, curious.'''
            else:
                Prompt += f'''Finally, articulate your plan as a single, detailed action. Describe this action in a tuple with a single element that describing your action. The action should only be one attempt action nor rely on any feedback, please don't imagine the outcome. Your description should strictly avoids any mention of psychological activities, intentions, and plans – essentially, anything that can't be directly observed. Don't use adjectives whose physical meaning is unclear e.g. careful, curious.'''

            # sample answer 1
            Input = f'''Example answers:\n\nExample1:\n **Step 1: Review the object currently interacting with**:'''
            Input += f'''The object is a <Wooden cabin door> it has a solid, wooden structure with a reliable handle and a lock, capable of providing a secure barrier against the storm and wildlife. Since the door is well equipped with reliable handle and a lock and lock, there is no extra effort needed to secure the door. Just need to close to the door and manipulate the parts on the door should achieve my goal.\n'''
            if len(backpack_objects) > 0:
                Input += f'''**Step 2:Check Personal belongings**:'''
                Input += f'''Although I have <Sleeping bag>, <Portable water filter>, <Flashlight>, <Emergency snacks>, <First aid kit> with me, I don't need them in this case because this door has a lock, I don't need any tools to secure it.\n\n'''
            else:
                Input += f'''**Step 2:Check Personal belongings**: I don't have any personal belonings so I will try secure the door without any form of tools.\n\n'''
                
            Input += f'''**Finally generate the tuple with action**: ("Emily reaches out to the handle of the wooden cabin door. Her grip is firm as she pulls the door towards her. Then she shifts her hand position to the lock. Her thumb and forefinger turn the lock clockwise with steady pressure until it clicks into place")'''

            # sample answer 2
            Input += f'''\n\nExample2:\n **Step 1: Review the object currently interacting with**:\n'''
            Input += f'''The object is an <Iron Chest>, it is a solid iron chest with strange symbols stamped on its surface, and, with its rounded surface, there was only a lock on the bulge in the middle of the chest, and this lock looked very strong. Commonly I think I won't open it without a tool, the chest itself is solid, so the only chance is to get rid of the lock.\n\n'''
            if len(backpack_objects) > 0:
                Input += f'''**Step 2:Check Personal belongings**:
                Currently, I have <Thin iron rod>, <Portable water filter>, <Flashlight>, <Emergency snacks>, <First aid kit> with me, I think I can insert my <Thin iron rod> into the gao between the lock and the <Iron Chest>, and then I should be able to force the lock open.'''
                Input += f'''**Finally generate the tuple with action**: ("I hold the Thin Iron Rod with both hands, angling it to slide precisely into the gap near the chest's lock. Gripping the rod tightly, I start applying downward pressure. The force is gradual but firm, increasing as I lean my body weight into the action. My wrists remain steady, guiding the rod, while my arms push downwards in a smooth, consistent motion, exerting controlled force to manipulate the lock.", "<Thin iron rod>")'''
            else:
                Input += f'''**Step 2:Check Personal belongings** I don't have any personal belongings so I will try open this <Iron Chest> without any form of tools, alough the chance is little, at least I need to give it a try.\n'''
                Input += f'''**Finally generate the tuple with action**: ("Bob grasped the sides of the iron box tightly and lifted it with both hands. He positioned himself and used his strength to lift the iron box above his head. Then, with his greatest strength, he slammed the locked side to the ground.")'''
            Input += f'''Now please generate action for {self.Host_CHIBI.Name}.'''

            if len(backpack_objects) > 0:
                Input += f'''Currently, {self.Host_CHIBI.Name} only have the following object with him, he can only choose tools or object from what he had, if none of them are useful you can not fake tools or items.'''
            else:
                Input += f'''Currently, {self.Host_CHIBI.Name} can't find any tools or items. So he decide to proceed without any form of tools.'''
            return Prompt, Input
        return _prompt_and_input

    def _get_still_try_func(self, fail_reason:str):
        '''Need to give a brief plan of how to use this object and what is the effect you want to achieve'''
        @utils.Prompt_constructor_for_system(self.Model_name,
                                             Usage = self.Usage,
                                             parse_function_str = 'str_with_tuple')
        def _prompt_and_input():
            Prompt = f'''Background information: {self.Belongs_to.get_information()}. Previously, {self.Host_CHIBI.Name}'s plan was: {self.get_information()}.
However, {self.Host_CHIBI.Name} encountered failure due to: {fail_reason}.\nGiven {self.Host_CHIBI.Name}'s previous experiences, do you think {self.Host_CHIBI.Name} will continuing interact with {self.Interactive_object.get_keyword()} or will he leave this behind for now? Please give your answer based on the following steps.'''

            Input = f'''**Step1** Analyze whether {self.Host_CHIBI.Name}'s failure was due to a lack of necessary prerequisites. Are there any indications that another attempt by {self.Host_CHIBI.Name} might lead to success?\n**Finally** Summarize your answer in a tuple with only one element. If {self.Host_CHIBI.Name} will give it another try, please answer (True). If {self.Host_CHIBI.Name} will leave {self.Interactive_object.get_keyword()} behind for now, please answer (False).'''

            return Prompt, Input
        return _prompt_and_input

    def _get_new_attempt_func(self, fail_reason:str):
        @utils.Prompt_constructor_for_system(self.Model_name, 
                                             Usage = self.Usage,
                                             parse_function_str = None)
        def _prompt_and_input():
            Prompt = f'''Background information: {self.Belongs_to.get_information()}. Previously, {self.Host_CHIBI.Name}'s plan was: {self.get_information()}.
However, {self.Host_CHIBI.Name} encountered failure due to: {fail_reason}.\nNow {self.Host_CHIBI.Name} decide to give it another try, please modify {self.Host_CHIBI.Name}'s previous plan a bit and generate a new plan.'''
            Input = f'''Please mention this is another attempt and modifiy previous plan based on the fail reason, and use one sentence to summarize the modified plan.'''
            return Prompt, Input
        
    def interact_pipeline(self):
        assert False, f'''{type(self)} is not interactable'''
        
    # def _get_new_interactive_action_plan_func(self):
    #     @utils.Prompt_constructor_for_system(self.Model_name,
    #                                          parse_function_str = 'str_with_tuple',
    #                                          Usage = self.Usage):
    # TODO


class Attemptation_Fixed_Interactive_Pipeline_Action(Attemptation_Action_Base):
    '''Decide to interact with the object, during the process will generate detailed action string and decide belonings need to use'''
    def __init__(self,
                 Information:str,
                 Belongs_to:'State_Simple',
                 Interactive_object:'fixed_blocks.Fixed_Action_Pipeline_Object',
                 Embedding:Optional[List[float]] = None,
                 Init_time:Optional[datetime.datetime] = None,
                 Model_name:Optional[str] = 'gpt-3.5-turbo-0125',
                 Usage:Optional[Dict[str,int]] = None, 
                 Impression_object:Optional[blocks.Object_Impression] = None,
                ):
        assert isinstance(Interactive_object,fixed_blocks.Fixed_Interact_Pipeline_Object_Base),f'''{type(self)} is only for fixed pipeline objects'''
        if Init_time is None:
            Init_time = datetime.datetime.now()
        super().__init__(Information, Belongs_to,Embedding = Embedding,
                         Init_time = Init_time,  Usage = Usage, Model_name = Model_name)
        self.Interactive_object = Interactive_object
        self.Host_CHIBI = self.Belongs_to.Belongs_to.Host_CHIBI
        self.Used_belongings = []
        self.detailed_action_string = self.Information
        self.Success_fail_state = None
        self.Success_fail_reason = None
        self.Selected_action_interactive_pipeline = self.Interactive_object.Parse_pipeline_dict[self.Information]
        
        if Impression_object is None:
            self.Impression_object = self.Host_CHIBI.Space_manager.find_impression_object(self.Interactive_object)
        else:
            self.Impression_object = Impression_object
        # Scores here will not affect the result
        self.Cost = 5#self.Selected_action_interactive_pipeline['Cost']
        self.FOD_score = 5#self.Selected_action_interactive_pipeline['FOD_score']

    def get_relative_objects(self)->List[blocks.CHIBI_Object]:
        '''only return REAL object that can be edited or delete'''
        '''This should be called after interact'''
        return_object_list = []
        # return_object_list.append(self.Interactive_object)
        # return_object_list.extend(self.Used_belongings) Currently don't need to parse tools used in interaction
        return return_object_list
        
    def cost_function(self)->int:
        CHIBI_cur_position = self.Host_CHIBI.Space_manager.Cur_position
        impression_space = self.Impression_object.Impression_space.Space_name
        if CHIBI_cur_position != impression_space:
            path_penalty_cost = len(self.Host_CHIBI.Space_Manager_System_Global.find_path(CHIBI_cur_position,impression_space)) * self.path_penalty_cost_weight 
            return (self.Cost + path_penalty_cost)/10
        else:
            return self.Cost/10


    def FOD_function(self)->int:
        return self.FOD_score/10
        
    #------------ CHIBI_object interfaces
    def get_information(self, viewer:Optional[Union[str,'CHIBI_base']] = None):
        return super().get_information(viewer = viewer)
        
    def edit(self,
             edited_information:str):
        super().edit(edited_information)

    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''

    def show(self):
        print(f'{self.Information}')

    def get_keyword(self)->str:
        super().get_keyword()

    def destory(self):
        # TODO add deletion function in parent state node
        super().destory()

    def __call__(self, memory_use:Optional[int] = None):
        '''when an fixed pipeline action is been taken(called)'''
        '''Should follow exact the following steps
        Step1,
        '''
        is_callable = self.callable()
        impression_space = self._get_impression_space()
        def _call():
            destory_bool = self.Interactive_object.interact_pipeline(self, memory_use = memory_use)
            if destory_bool and self.Success_fail_state:
                # delete previous impression:
                previous_impression_space = self.Impression_object.Impression_space
                previous_impression_space.object_delete(self.Impression_object)

                # if the object is an edge might have impression in two space: delete that impression as well
                all_impression_objects = self.Host_CHIBI.Space_manager.get_cur_space(space_type = 'impression').All_objects
                for impression_object in all_impression_objects:
                    if self.Interactive_object is impression_object.Impression_of:
                        self.Host_CHIBI.Space_manager.get_cur_space(space_type = 'impression').object_delete(impression_object)                    
                
                # delete attemptation action related to this action
                action_need_to_remove = []
                for single_action in self.Belongs_to.Attemptations:
                    if single_action.Impression_object is self.Impression_object:
                        action_need_to_remove.append(single_action)
                for single_action in action_need_to_remove:
                    self.Belongs_to.Attemptations.remove(single_action)
                        
                
            self.Tried_times +=1
            
        if is_callable[0]:
            _call()
        else:
            if is_callable[1] == 'Space_error':
                # first move
                self.Host_CHIBI.move(impression_space,mode = 'GO')
                #then call
                _call()
            elif is_callable[1] == 'Object_error':
                assert False,f'Object disparred :{is_callable[2]} {type(self)}, and this should not happen in single agent setting'
            else:
                assert False,f'callable return false information,{is_callable}'
        
    def callable(self):
        impression_space = self._get_impression_space()
        # Only when CHIBI is at that space and that space have this object can CHIBI call this action
        CHIBI_cur_position = self.Host_CHIBI.Space_manager.Cur_position
        Cur_space_real = self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict[CHIBI_cur_position]
        all_objects_in_this_space = []
        all_objects_in_this_space.extend(Cur_space_real.retrieve_item_in_this_space('Things'))
        all_objects_in_this_space.extend(Cur_space_real.retrieve_item_in_this_space('Edges'))
        all_objects_in_this_space.extend(Cur_space_real.retrieve_item_in_this_space('CHIBIs'))
        if CHIBI_cur_position != impression_space.Space_name:
            return (False, f'Space_error')
        if self.Interactive_object not in all_objects_in_this_space:
            return (False, f'Object_error') # TODO 这里可能只有部分的物品 但是没有全部的物品 现在先不考虑这么复杂的情况
        return (True,)
        
    def _get_impression_space(self):
        # Since edge exist in two spaces, so when decide the impression of the space always go to the closest space as impression space. (The path length is calculated by CHIBI's Space Impression path)
        if isinstance(self.Interactive_object, fixed_blocks.Fixed_pipeline_Simple_Edge):
            cur_space_str = self.Host_CHIBI.Space_manager.Cur_position
            vertex_1 = self.Interactive_object.Connected_two_space[0]
            vertex_2 = self.Interactive_object.Connected_two_space[1]
            if vertex_1.Space_name in self.Host_CHIBI.Space_manager.Vertices_dict and vertex_2.Space_name in self.Host_CHIBI.Space_manager.Vertices_dict:
                path_length_1 = self.Host_CHIBI.Space_manager.find_path(cur_space_str, vertex_1.Space_name)
                path_length_2 = self.Host_CHIBI.Space_manager.find_path(cur_space_str, vertex_2.Space_name)
                impression_space = vertex_1 if path_length_1 <= path_length_2 else vertex_2
            else:
                impression_space = self.Impression_object.Impression_space
        else:
            impression_space = self.Impression_object.Impression_space
        return impression_space

    #prompt based functions--------------------------

class Attemptation_Abduction_Action(Attemptation_Action_Base):
    '''CHIBI can call this action infinitely many times, and each time using most relevant observations to reflect'''
    def __init__(self,
                 Information:str,
                 Belongs_to:'State_Simple',
                 Embedding:Optional[List[float]] = None,
                 Init_time:Optional[datetime.datetime] = None,
                 Model_name:Optional[str] = 'gpt-3.5-turbo-0125',
                 Usage:Optional[Dict[str,int]] = None, 
                ):
        if Init_time is None:
            Init_time = datetime.datetime.now()
        super().__init__(Information, Belongs_to,Embedding = Embedding,
                         Init_time = Init_time,  Usage = Usage, Model_name = Model_name)

        self.Host_CHIBI = self.Belongs_to.Belongs_to.Host_CHIBI
        self.detailed_action_string = self.Information
        self.Success_fail_state = False # reflection action will not be finished.
        
        self.Cost = 5#self.Selected_action_interactive_pipeline['Cost'] (Currently not in use)
        self.FOD_score = 5#self.Selected_action_interactive_pipeline['FOD_score'] (Currently not in use)
        
    def cost_function(self)->int:
        
        return self.Cost/10

    def get_relative_objects(self):
        # This action do not need to have object to interact with
        pass


    def FOD_function(self)->int:
        return self.FOD_score/10
        
    #------------ CHIBI_object interfaces
    def get_information(self, viewer:Optional[Union[str,'CHIBI_base']] = None):
        return super().get_information(viewer = viewer)
        
    def edit(self,
             edited_information:str):
        super().edit(edited_information)

    def interact_pipeline(self):
        assert False, f'''currently {type(self)} is not interactable'''

    def show(self):
        print(f'{self.Information}')

    def get_keyword(self)->str:
        super().get_keyword()

    def destory(self):
        # TODO add deletion function in parent state node
        super().destory()

    def __call__(self):
        self.Host_CHIBI.Memory_stream.abduction_loop()

    def callable(self):
        # This action is always callable
        return (True,)
        
class Attemptation_Creative_Action(Attemptation_Action_Base):
    def __init__(self):
        pass
    # TODO need to decide when to generated these actions and 
# Plan nodes ----------------------------------------------------------
    
class State_Simple(State_Base):
    '''Stores information of current situation including who is this character why it is here what is it's goal some important history actions'''
    def __init__(self,
                 Information:str,
                 Belongs_to:'Plan_system_Base',
                 Parent_state:Optional['State_Simple'] = None,
                 Child_states:Optional[List['State_Simple']] = None,
                 Editable:bool = True,
                 Cost:Optional[str] = None,
                 FOD_score:Optional[int] = None,
                 Attemptations:Optional[List[Attemptation_Action_Base]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:Dict[str,int] = None,
                 Embedding:Optional[List[float]] = None,
                 Tried_action_str:Optional[str] = None,
                 Narrative_action_str:Optional[str] = None
                ):
        self.Information = Information
        self.Parent_state = Parent_state
        if Child_states is None:
            self.Child_states = []
        else:
            self.Child_states = Child_states
            
        if Attemptations is None:
            self.Attemptations = []
        else:
            self.Attemptations = Attemptations
        self.Editable = Editable
        self.Belongs_to = Belongs_to
        self.Cost = Cost
        self.FOD_score = FOD_score
        self.Model_name = Model_name
        if Usage is None:
            self.Usage = {'completion_tokens':0,
                          'prompt_tokens':0,
                          'total_tokens':0}
        else:
            self.Usage = Usage
        self.Tried_action_str = '' if Tried_action_str is None else Tried_action_str
        self.Narrative_action_str = '' if Narrative_action_str is None else Narrative_action_str
        self.Host_CHIBI = self.Belongs_to.Host_CHIBI
        if not isinstance(self.Belongs_to, Plan_System_Human):
            if Embedding is None:
                self.Embedding = utils.get_embedding(self.get_information())[0]
            else:
                self.Embedding = Embedding
        if self.Host_CHIBI.Do_abduction:
            abduction_action_str = f'''Modify previous hypothesis and make a new plan: (Take this action when your current observations contradict your previous hypothesis or your current plan is fulfilled.)'''
            abduction_action = Attemptation_Abduction_Action(abduction_action_str, self)
            self.Attemptations.append(abduction_action)
        
    #------------ Custom functions
        
    #------------ State node interfaces
    def add_attempt_action(self,
                           Attemptation_action:Attemptation_Action_Base):
        assert isinstance(Attemptation_action, Attemptation_Action_Base), f'{Attemptation_action} is not a type of action'
        self.Attemptations.append(Attemptation_action)
        self.Belongs_to.update_state_graph()
        
    def generate_next_state(self):
        pass

    def FOD_function(self)->int:
        '''Generating a FOD estimation for the distance to final goal after achieving this sub-goal'''
        return 0
        
    def cost_function(self)->int:
        '''generate the cost of this action'''
        return 0

    def generate_perceptual_action_based_on_impressions(self, top_n:Optional[int] = None):
        self.Host_CHIBI.look_around() # update impression in current space

        cur_space_name = self.Host_CHIBI.Space_manager.Cur_position
        cur_space = self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict[cur_space_name]
        relevant_impressions = self.Host_CHIBI.recall_impressions(top_n = top_n)
        if top_n is None: # this means return all impression and create perceptual action for each of them
            top_n = len(relevant_impressions)
        
        for impression in relevant_impressions:
            # if currently, in the attemptation list have action generated based on the impression then do not create
            created = False
            for attempt_action in self.Attemptations:
                if isinstance(attempt_action, Attemptation_Abduction_Action):
                    continue
                if attempt_action.Impression_object.Impression_of is impression.Impression_of:# Two impression have the same real object
                    if isinstance(impression.Impression_of, blocks.Edge_Double_Side): 
                        # if this is a Edge double side, one real object can have two impression
                        if attempt_action.Impression_object.Impression_space != impression.Impression_space and impression.Impression_space not in [i.Impression_object.Impression_space for i in self.Attemptations]: # only when this is a edge and created action space is not the same space will we create another acion.
                            print(f'''Creat a new perceptual action for {impression.get_keyword}''')
                            new_perceptual_action_other_side_edge = Attemptation_Perceptual_Action(f'Investigate other side of {impression.Impression_of.get_keyword()}, you know that you just came from the other side of the {impression.Impression_of.get_keyword()}', self, impression.Impression_of, Impression_object = impression)
                            self.Attemptations.append(new_perceptual_action_other_side_edge)
                    created = True
                    break
            if not created:
                new_perceptual_action = Attemptation_Perceptual_Action(f'Investigate {impression.Impression_of.get_keyword()}.',self,
                                                         impression.Impression_of, Impression_object = impression)
                self.Attemptations.append(new_perceptual_action)
        
    #------------ CHIBI Object interfaces
    def edit(self, new_action_str:str):
        # Information is root background will not be changed unless we generate a new state
        self.Tried_action_str += new_action_str + '\n\n' #system action str
        #narrated_information = narrators.State_edit_narrator.narrate(self, Model_name = self.Host_CHIBI.Model_name)
        # update embedding
        #self.Narrative_action_str = narrated_information
        if self.Embedding is not None:
            self.Embedding = utils.get_embedding(self.get_information())[0]

    def show(self):
        print(f'{self.Information}')

    def get_information(self, viewer:Optional[Any] = None)->str:
        return self.Information + self.Narrative_action_str

    def get_keyword(self)->str:
        super().get_keyword()

    def destory(self):
        # TODO add deletion function in parent plan node
        assert False, f'{type(self)} destory not implemented'
        pass

    def interact_pipeline(self):
        assert False, f'''{type(self)} is not interactable'''


class State_Simple_Human(State_Base):
    '''Stores information of current situation including who is this character why it is here what is it's goal some important history actions'''
    def __init__(self,
                 Information:str,
                 Belongs_to:'Plan_system_Base',
                 Parent_state:Optional['State_Simple_Human'] = None,
                 Child_states:Optional[List['State_Simple_Human']] = None,
                 Editable:bool = True,
                 Cost:Optional[str] = None,
                 FOD_score:Optional[int] = None,
                 Attemptations:Optional[List[Attemptation_Action_Base]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:Dict[str,int] = None,
                 Embedding:Optional[List[float]] = None,
                 Tried_action_str:Optional[str] = None,
                 Narrative_action_str:Optional[str] = None,
                 Do_abduction:bool = False,
                ):
        self.Information = Information
        self.Parent_state = Parent_state
        if Child_states is None:
            self.Child_states = []
        else:
            self.Child_states = Child_states
            
        if Attemptations is None:
            self.Attemptations = []
        else:
            self.Attemptations = Attemptations
        self.Editable = Editable
        self.Belongs_to = Belongs_to
        self.Cost = Cost
        self.FOD_score = FOD_score
        self.Model_name = Model_name
        if Usage is None:
            self.Usage = {'completion_tokens':0,
                          'prompt_tokens':0,
                          'total_tokens':0}
        else:
            self.Usage = Usage
        self.Tried_action_str = '' if Tried_action_str is None else Tried_action_str
        self.Narrative_action_str = '' if Narrative_action_str is None else Narrative_action_str
        self.Host_CHIBI = self.Belongs_to.Host_CHIBI

        if self.Host_CHIBI.Do_abduction:
            abduction_action_str = f'''Modify previous hypothesis and make a new plan: (Take this action when your current observations contradict your previous hypothesis or your current plan is fulfilled.)'''

            abduction_action = Attemptation_Reflect_Action(abduction_action_str, self)
            self.Attemptations.append(abduction_action)
        
        
    #------------ Custom functions
        
    #------------ State node interfaces
    def add_attempt_action(self,
                           Attemptation_action:Attemptation_Action_Base):
        assert isinstance(Attemptation_action, Attemptation_Action_Base), f'{Attemptation_action} is not a type of action'
        self.Attemptations.append(Attemptation_action)
        self.Belongs_to.update_state_graph()
        
    def generate_next_state(self):
        pass

    def FOD_function(self)->int:
        '''Generating a FOD estimation for the distance to final goal after achieving this sub-goal'''
        return 0
        
    def cost_function(self)->int:
        '''generate the cost of this action'''
        return 0

    def generate_perceptual_action_based_on_impressions(self, top_n:Optional[int] = None):
        self.Host_CHIBI.look_around() # update impression in current space

        cur_space_name = self.Host_CHIBI.Space_manager.Cur_position
        cur_space = self.Host_CHIBI.Space_Manager_System_Global.Vertices_dict[cur_space_name]
        relevant_impressions = self.Host_CHIBI.recall_impressions(top_n = top_n)
        if top_n is None: # this means return all impression and create perceptual action for each of them
            top_n = len(relevant_impressions)
        
        for impression in relevant_impressions:
            # if currently, in the attemptation list have action generated based on the impression then do not create
            created = False
            for attempt_action in self.Attemptations:
                if isinstance(attempt_action, Attemptation_Abduction_Action):
                    continue
                if attempt_action.Impression_object.Impression_of is impression.Impression_of:
                    if isinstance(impression.Impression_of, blocks.Edge_Double_Side): 
                        # if this is a Edge double side, one real object can have two impression
                        if attempt_action.Impression_object.Impression_space != impression.Impression_space and impression.Impression_space not in [i.Impression_object.Impression_space for i in self.Attemptations]: # only when this is a edge and created action space is not the same space will we create another acion.
                            print(f'''Creat a new perceptual action for {impression.get_keyword}''')
                            new_perceptual_action_other_side_edge = Attemptation_Perceptual_Action(f'Investigate other side of {impression.Impression_of.get_keyword()}, you know that you just came from the other side of the {impression.Impression_of.get_keyword()}', self, impression.Impression_of, Impression_object = impression)
                            self.Attemptations.append(new_perceptual_action_other_side_edge)
                    created = True
                    break
            if not created:
                new_perceptual_action = Attemptation_Perceptual_Action(f'Investigate {impression.Impression_of.get_keyword()}.',self,
                                                         impression.Impression_of, Impression_object = impression)
                self.Attemptations.append(new_perceptual_action)
        
    #------------ CHIBI Object interfaces
    def edit(self, new_action_str:str):
        # Information is root background will not be changed unless we generate a new state
        self.Tried_action_str += new_action_str + '\n\n' #system action str
        #narrated_information = narrators.State_edit_narrator.narrate(self, Model_name = self.Host_CHIBI.Model_name)
        # update embedding
        #self.Narrative_action_str = narrated_information
        #self.Embedding = utils.get_embedding(self.get_information())[0]

    def show(self):
        print(f'{self.Information}')

    def get_information(self, viewer:Optional[Any] = None)->str:
        return self.Information + self.Narrative_action_str

    def get_keyword(self)->str:
        super().get_keyword()

    def destory(self):
        # TODO add deletion function in parent plan node
        assert False, f'{type(self)} destory is not implemented'
        pass

    def interact_pipeline(self):
        assert False, f'''{type(self)} is not interactable'''

# ----------------------------------Basic ABC plan system ---------------------------------------
# ----------------------------------Basic ABC plan system --------------------------------------- 
class Plan_System_Base(ABC):
    def __init__(self,
                 Host_CHIBI:'CHIBI.CHIBI_Base',
                 State_dict:Optional[Dict[str,State_Base]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:Dict[str,int] = None
                ):
        self.label_colors = {
                        'State_node': 'blue',
                        'Attemptation_action_untried': 'orange',
                        'Attemptation_action_tried': 'gray'
                        }
        self.Host_CHIBI = Host_CHIBI
        if State_dict is None:
            self.State_dict = {}
        else:
            self.State_dict = State_dict
        self.update_state_graph()
        self.Model_name = Model_name
        self.Usage = Usage
        
    @abstractmethod
    def update_state_graph(self,
                          State_dict:Optional[Dict[str,State_Base]] = None,
                         ):
        # makesure self.State_dict is up to date
        if State_dict is None:
            pass
        else:
            self.State_dict = State_dict
        G = nx.DiGraph()  # Plan nodes are a directed graph
        # update all generated
        for state_node in self.State_dict.values():
            G.add_node(state_node, label = 'State_node')
            for child_state in state_node.Child_states:
                G.add_edge(state_node, child_state, label = 'State_node')
            if state_node.Parent_state is not None:
                G.add_edge(state_node, state_node.Parent_state, label = 'State_node')
            for attempt_action in state_node.Attemptations:
                if attempt_action.Success_fail_state:
                    G.add_node(attempt_action, label = 'Attemptation_action')
                    G.add_edge(attempt_action, state_node, label = 'Attemptation_action')
                else:
                    G.add_node(attempt_action, label = 'Attemptation_action_untried')
                    G.add_edge(attempt_action, state_node, label = 'Attemptation_action_untried')
        self.G = G
    @abstractmethod
    def update_state(self)->str:
        pass
        
    @abstractmethod
    def get_action(self)->Attemptation_Action_Base:
        '''using all the context information to decide what to do nxt return a callable action'''
        return None
    
    @abstractmethod
    def finish_state_node(self):
        '''Something happens when a node is finished from the graph'''
        # when plan is finish it will not generate successor attemptations anymore 
        # and will do some positive effect for node in this path
        pass

    @abstractmethod
    def giveup_state_node(self):
        '''Something happens when a node is given up by Hose'''
        # when plan is given up, it will genera
        pass

    @abstractmethod
    def show(self,
             node_size:float = 500, 
             font_size:float = 8,
             figure_size:tuple = (10,8),
             node_color:str = 'lightblue',
             edge_color:str = 'gray'):
        '''print and show all plans in this plan system'''
        if len(self.State_dict) == 0:
            print('Currently no plans in this plan system')
        else:
            plt.figure(figsize=figure_size)
            pos = nx.spring_layout(self.G) 
            edge_colors = [self.label_colors.get(self.G.edges[edge]['label'], 'gray') for edge in self.G.edges()]
            node_colors = [self.label_colors.get(self.G.nodes[node]['label'], 'lightblue') for node in self.G.nodes()]
            node_labels = {node: node.get_information() for node in self.G.nodes()}
            nx.draw(self.G, pos, with_labels=True, labels=node_labels,node_color=node_colors, 
                    edge_color=edge_colors, node_size=node_size, font_size=font_size)
            plt.show()

# ---------------------------------- plan system ---------------------------------------
# ---------------------------------- plan system ---------------------------------------
class Plan_System_CHIBI_main_character(Plan_System_Base):
    def __init__(self, 
                 Host_CHIBI:"CHIBI.CHIBI_main_character",
                 State_dict:Optional[Dict[str,State_Simple]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Usage:Dict[str,int] = None,
                 forced_abduction:bool = False,
                ):
        super().__init__(Host_CHIBI, 
                         State_dict = State_dict, 
                         Model_name = Model_name,
                         Usage = Usage)
        self.Cur_state = None
        self.Previous_called_action = None # should be assigned in the experiment loop incase the action failed
        self.forced_abduction = forced_abduction
        
    #------------ Base class interfaces
    def update_state(self,
                     plan:tuple=None,
                     mode = 'Generate',):
        '''need to generate a feasible plan for CHIBI with prompt combined with current situation and goal'''
        if plan is None:
            if mode == 'Generate':
                print('Currently not implemented')
                return
            elif mode == 'Decompose':
                print('not implemented')
                return 
            else:
                pass
        else: 
            # provided a plan, just add it to the plan system
            pass
        self.update_state_graph()

    def print_all_action_scores(self,
                   weight:List[float] = None):
        if weight is None:
            weight = [1,1,-1,1] #relevance_weight, recency_weight, cost_weight, FOD_weight
        # current all these values are not in use, becasue action space is small and will show all actions but listed in order of scores caluted by these weights
        relevance_weight = weight[0]  # This should be positive value 
        recency_weight = weight[1] # This should be positive value  
        cost_weight = weight[2]   # this should be a negative value
        FOD_weight = weight[3] 
        available_actions = [attempt for attempt in self.Cur_state.Attemptations if not attempt.Success_fail_state] 
        for action in available_actions:
            print(f'cur_action_information:{action.get_information()}')
            # time recency_score  range(0,1)
            hour_passed = int((self.Host_CHIBI.Cur_time - action.Init_time).total_seconds()/3600)
            recency_score = np.power(action.recency_decay_factor,hour_passed)*recency_weight
            
            # state relevance score
            relevance_score = utils.calculate_cosine_similiarity(action.Embedding, self.Cur_state.Embedding)*relevance_weight 
            # cost score
            cost_score = action.cost_function() * cost_weight
            # FOD_score
            FOD_score = action.FOD_function() * FOD_weight
            print(f'''The Scores of action:{action.get_information()} relevance_score:{relevance_score}, recency_score:{recency_score}, cost_score: {cost_score}, FOD_score: {FOD_score}, Final_score = {relevance_score+recency_score+cost_score+FOD_score}''')
            print('-----------------------------------------------------------------------------------')
        
    def get_action(self,
                   weight:List[float] = None,
                   top_n:Optional[int] = None,
                   use_prompt_choose:bool = True,
                   memory_use:Optional[int] = None,
                   memory_retrieve_type:str = 'Most_recent',
                   return_list:bool = False, )->Attemptation_Action_Base:
        if weight is None:
            weight = [1,1,-1,1] #relevance_weight, recency_weight, cost_weight, FOD_weight
        relevance_weight = weight[0]  # This should be positive value
        recency_weight = weight[1] # This should be positive value  
        cost_weight = weight[2]   # this should be a negative value
        FOD_weight = weight[3]
        available_actions = [attempt for attempt in self.Cur_state.Attemptations] # All these matric are not in use currently, and all the action will be listed.
        filter_invisible_actions = []
        # remove actions that can not be called by defination in the database
        for action in available_actions:
            if isinstance(action, Attemptation_Fixed_Interactive_Pipeline_Action):
                visible_condition = action.Selected_action_interactive_pipeline['Show_condition']
                if not action.Interactive_object.action_visible(visible_condition, action): # by defined condition this action is not visible for now
                    filter_invisible_actions.append(action)
                elif not action.Selected_action_interactive_pipeline['Repeat_action'] and action.Success_fail_state: # This action only allowed to call once if succeed.
                    filter_invisible_actions.append(action)
            else:
                if action.Success_fail_state:
                    filter_invisible_actions.append(action)
                    
        action_scores = []
        for action in filter_invisible_actions:
            available_actions.remove(action)
        for action in available_actions:
            if isinstance(action, Attemptation_Abduction_Action):
                pass
            elif not isinstance(action, Attemptation_Perceptual_Action): # if it is interacting with an object, then do the following to update detailed action information
                new_action_information = action.Interactive_object.update_str_with_variable(action, get_action_decision = True)
                if new_action_information != action.Information:
                    action.Information = new_action_information
                    action.update_embedding()

            #action.decoreate_space_move_information()

            # time recency_score  range(0,1)
            hour_passed = int((self.Host_CHIBI.Cur_time - action.Init_time).total_seconds()/3600)
            recency_score = np.power(action.recency_decay_factor,hour_passed)*recency_weight
            
            # state relevance score
            relevance_score = utils.calculate_cosine_similiarity(action.Embedding, self.Cur_state.Embedding)*relevance_weight 
            # cost score
            cost_score = action.cost_function() * cost_weight
            # FOD_score
            FOD_score = action.FOD_function() * FOD_weight
            #print(f'''The Scores of action:{action.get_information()} relevance_score:{relevance_score}, recency_score:{recency_score}, cost_score: {cost_score}, FOD_score: {FOD_score}''')
            action_scores.append(recency_score+relevance_score+cost_score+FOD_score)

        # sort
        if top_n is None:
            top_n = len(available_actions)
        final_score_dict = {i[0]:i[1] for i in zip(available_actions, action_scores)}
        top_n_action_list = sorted(final_score_dict.items(), key=lambda item: item[1],reverse = True)[:top_n]

        #Auto call all perceptual action and get all informaiton: (In current experiment setting the order to investigate all action does not matter and these perceptual action are providede as initial observations)
        for action in top_n_action_list:
            if isinstance(action[0], Attemptation_Perceptual_Action):
                return action[0]

        if return_list:
            # Currently only for constructing the prompt for abductive action so remove abduction action before return 
            remove_action = None
            for action in top_n_action_list:
                if isinstance(action[0], Attemptation_Abduction_Action):
                    remove_action = action
            if remove_action is not None:
                top_n_action_list.remove(remove_action)
            return top_n_action_list
                
        #First abduction and generate an initial hypothesis and plan at the begining of the puzzle
        if len(self.Host_CHIBI.Memory_stream.Buffer_memories) > self.Host_CHIBI.Memory_stream.Buffer_size:
            if self.Host_CHIBI.Do_abduction:
                if self.Host_CHIBI.Memory_stream.Cur_assumption_and_plan is None: # only when Agent do not have an hypothesis to begin with will it forced to call this action, otherwize agent will only do abduction when he select this action
                    for action in top_n_action_list:
                        if isinstance(action[0], Attemptation_Abduction_Action):
                            return action[0]

        if self.Host_CHIBI.Do_abduction and self.forced_abduction:
            assert self.Host_CHIBI.Memory_stream.Cur_assumption_and_plan is not None, f'''Agent should have a hypothesis at this point.'''
            if not isinstance(self.Previous_called_action, Attemptation_Abduction_Action): # if previous action is not abduction action then force agent to do one step of abduction
                for action in top_n_action_list:
                    if isinstance(action[0], Attemptation_Abduction_Action):
                        return action[0]
                    
        if self.Host_CHIBI.Do_abduction:
            if self.Host_CHIBI.Special_label != 'Reactor_puzzles': # we find out that LLMs are not very sensible to character level operations, how ever when setting to low temperature, they are able to tell whether current hypothesis is consistant with observations, let them repeatedly call this may addup to cost but will turned out with better hypothesis. Decide add this or not before running the experiment.
                if isinstance(self.Previous_called_action, Attemptation_Abduction_Action): # avoid calling the induction repeatedly
                    return_action_list = []
                    for action in top_n_action_list:
                        if isinstance(action[0], Attemptation_Abduction_Action):
                            pass
                        else:
                            return_action_list.append(action)
                    top_n_action_list = return_action_list
                    
                
        
        #If there is no hypothesis then do not show the the modify inductive to agent
        #If the abduction is set to forced call after each interaction, then do not need to show inductive action
        if self.Host_CHIBI.Do_abduction:
            if self.Host_CHIBI.Memory_stream.Cur_assumption_and_plan is None or self.forced_abduction:
                remove_revise_assumption_action = None
                for action in top_n_action_list:
                    if isinstance(action[0], Attemptation_Abduction_Action):
                        remove_revise_assumption_action = action
                assert remove_revise_assumption_action is not None, f'''No hypothesis action founded in the available action list.'''
                top_n_action_list.remove(remove_revise_assumption_action)

        # prompt selected action
        if use_prompt_choose and len(top_n_action_list) > 1:
            Prompt, Input, parse_function_str, logging_label = self._select_action_prompt(top_n_action_list, previous_memory_use = memory_use)
            generated_result = self.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = logging_label)
            try:
                selected_action_index = int(generated_result) - 1
            except TypeError:
                self.Host_CHIBI.Memory_stream.memory_add(f'''You should select an action and use one integer to respond your selection. But your input {generated_result} is not an integer selection''')
                raise TypeError(f'''You should select an action and use one integer to respond your selection. But your input {generated_result} is not an integer selection''')
            return top_n_action_list[selected_action_index][0]
        else:
            return top_n_action_list[0][0]
    
    def _select_action_prompt(self, top_n_action_list:List[Attemptation_Action_Base], previous_memory_use:Optional[int] = None): # Deduction step
        previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str = self.Host_CHIBI.retrieve_prompt_information(memory_use = previous_memory_use)
        action_str_list = [f'{index+1}th action: '+action[0].get_information() for index,action in enumerate(top_n_action_list)]
            
        Prompt = f'''{self.Cur_state.get_information()} \n{self.Host_CHIBI.Name} now decide to choose one of the actions provided to achieve his goal. Please think in the aspect of {self.Host_CHIBI.Name}, and use the following information to select your action:\n\n{previous_memories_str}{cur_assumption_str}{cur_plan_str}{most_recent_memories_str}{storage_information}What is the most suitable next action for {self.Host_CHIBI.Name} based on above given information? Below are the available actions:\n\n''' + '\n'.join(action_str_list) + f'''\n\nAbove {len(top_n_action_list)} provided actions are not yet performed by {self.Host_CHIBI.Name} don't assume its outcome, please follow the steps to generate your final answer. You MUST select one of the provided actions. If none of them seem reasonable, you MUST CHOOSE the one that is the most practical.'''

        Input = f'''**Step1:** Review all the provided actions. Reflect on {self.Host_CHIBI.Name}'s current situation and goal to assess if each action is logical and appropriate.\n**Step2:** Choose the most logical action. Explain why this action is the best choice compared to the others, focusing on how it aligns with {self.Host_CHIBI.Name}'s goals and situation.\n**Finally** Indicate your selected action by placing its corresponding Arabic numeral in square bracket at the end. For example, if the third action is chosen, write [3]. Please do not use square bracket anywhere else other than final answer.'''
        return Prompt, Input, 'str_with_square_bracket', 'Action_select'

    def update_state_graph(self):
        super().update_state_graph()
        
    def finish_state_node(self):
        pass

    def giveup_state_node(self):
        pass
        
    def show(self):
        super().show()
        
    #------------ Custom functions 
    def generate_actions(self):
        '''return the Attempt action that have the highest score'''
        self.Cur_state.generate_perceptual_action_based_on_impressions()
        self.update_state_graph()

    def add_state(self,
                 plan_information:str,
                 Parent_state:Optional['State_Simple'] = None,
                 Child_states:Optional[List['State_Simple']] = None,
                 Attemptations:Optional[List[Attemptation_Action_Base]] = None,
                ):
        new_state = State_Simple(plan_information, self, Parent_state = Parent_state,
                                   Child_states = Child_states, Attemptations = Attemptations, Model_name = self.Model_name)
        self.State_dict.update({new_state.Information:new_state})
        self.update_state_graph()
        self.Cur_state = new_state
    
    def _get_main_plan_func(self)->Callable[...,Any]:

        pass

    def action(self,
               task = 'Retrieve all', # 'Preceptial action', 'move', 'Interactive action'
              ) -> Dict[str,Union[List[blocks.CHIBI_Object],str,List[str],'space_manager.Space_Base']]:
        pass

    
class Plan_System_Human(Plan_System_Base):
    def __init__(self, 
                 Host_CHIBI:"CHIBI.CHIBI_human",
                 State_dict:Optional[Dict[str,State_Simple]] = None,
                ):
        super().__init__(Host_CHIBI, 
                         State_dict = State_dict,)
        self.Cur_state = None
        
    #------------ Base class interfaces
    def update_state(self,
                     plan:tuple=None,
                     mode = 'Generate',): # Currently not in use
        '''need to generate a feasible plan for CHIBI with prompt combined with current situation and goal'''
        if plan is None:
            if mode == 'Generate':
                print('Generate not implemented')
                return
            elif mode == 'Decompose':
                print('Decompose not implemented')
                return 
            else:
                pass
        else: 
            # provided a plan, just add it to the plan system
            pass
        self.update_state_graph()
        
    def get_action(self,
                   weight:List[float] = None,
                   top_n:Optional[int] = None,
                   use_prompt_choose:bool = True,
                   memory_use:Optional[int] = None,
                   memory_retrieve_type:str = 'Most_recent',
                   return_list:bool = False, )->Attemptation_Action_Base:

        available_actions = [attempt for attempt in self.Cur_state.Attemptations]
        filter_invisible_actions = []
        # remove actions that can not be called by defination in the database
        for action in available_actions:
            if isinstance(action, Attemptation_Fixed_Interactive_Pipeline_Action):
                visible_condition = action.Selected_action_interactive_pipeline['Show_condition']
                if not action.Interactive_object.action_visible(visible_condition, action): # by defined condition this action is not visible for now
                    filter_invisible_actions.append(action)
                elif not action.Selected_action_interactive_pipeline['Repeat_action'] and action.Success_fail_state: # This action only allowed to call once if succeed.
                    filter_invisible_actions.append(action)
            else:
                if action.Success_fail_state:
                    filter_invisible_actions.append(action)
                    
        for action in filter_invisible_actions:
            available_actions.remove(action)

        for action in available_actions:
            if isinstance(action, Attemptation_Abduction_Action):
                pass
            elif not isinstance(action, Attemptation_Perceptual_Action): # if it is interacting with an object, then do the following to update detailed action information
                new_action_information = action.Interactive_object.update_str_with_variable(action, get_action_decision = True)
                if new_action_information != action.Information:
                    action.Information = new_action_information

        for action in available_actions:
            if isinstance(action, Attemptation_Perceptual_Action):
                return action
                
        if return_list:
            # Currently only for constructing the prompt for abductive action so remove abduction action before return 
            remove_action = None
            for action in available_actions:
                if isinstance(action, Attemptation_Abduction_Action):
                    remove_action = action
            if remove_action is not None:
                available_actions.remove(remove_action)
            return [(single_action,) for single_action in available_actions] # since human plan system do not have weights to rank the action need to keep the format the same as LLMs
                
        if self.Host_CHIBI.Memory_stream.Cur_assumption_and_plan is None:
            for action in available_actions:
                if isinstance(action, Attemptation_Abduction_Action):
                    return action

        # prompt selected action
        if use_prompt_choose and len(available_actions) > 1:
            generated_result = self._select_action_prompt(available_actions, previous_memory_use = memory_use)
            selected_action_index = int(generated_result) - 1
            return available_actions[selected_action_index]
        else:
            return available_actions[0]
    
    def _select_action_prompt(self, top_n_action_list:List[Attemptation_Action_Base], previous_memory_use:Optional[int] = None): # Deduction step
        previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str = self.Host_CHIBI.retrieve_prompt_information(memory_use = previous_memory_use)
        parse_function_str = None
        action_str_list = [f'{index+1}th action: '+action.get_information() for index,action in enumerate(top_n_action_list)]
        Prompt = f'''{self.Cur_state.get_information()}\n{self.Host_CHIBI.Name} now decide to choose one of the actions provided to achieve his goal. Please think in the aspect of {self.Host_CHIBI.Name}, and use the following information to select your action:\n\n{previous_memories_str}{storage_information}{cur_assumption_str}{cur_plan_str}{most_recent_memories_str}What is the most suitable next action for {self.Host_CHIBI.Name} based on above given information? Below are the available actions:\n\n''' + '\n'.join(action_str_list) 
        Input = utils.decorate_text_with_color('''\n\nPlease indicate your next action by providing a single arabic numerals, eg if you want to take the second action please input 2''','cyan',bold = True,deep = True)
        logging_label = 'Action_select'
        selected_action = self.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = logging_label)
        return selected_action

    def update_state_graph(self):
        super().update_state_graph()
        
    def finish_state_node(self):
        pass

    def giveup_state_node(self):
        pass
        
    def show(self):
        super().show()
        
    #------------ Custom functions 
    def generate_actions(self):
        '''return the Attempt action that have the highest score'''
        self.Cur_state.generate_perceptual_action_based_on_impressions()
        self.update_state_graph()

    def add_state(self,
                 plan_information:str,
                 Parent_state:Optional['State_Simple'] = None,
                 Child_states:Optional[List['State_Simple']] = None,
                 Attemptations:Optional[List[Attemptation_Action_Base]] = None,
                ):
        new_state = State_Simple(plan_information, self, Parent_state = Parent_state,
                                   Child_states = Child_states, Attemptations = Attemptations, Model_name = self.Model_name)
        self.State_dict.update({new_state.Information:new_state})
        self.update_state_graph()
        self.Cur_state = new_state
    
    def _get_main_plan_func(self)->Callable[...,Any]:

        pass

    def action(self,
               task = 'Retrieve all', # 'Preceptial action', 'move', 'Interactive action'
              ) -> Dict[str,Union[List[blocks.CHIBI_Object],str,List[str],'space_manager.Space_Base']]:
        pass
    
    

