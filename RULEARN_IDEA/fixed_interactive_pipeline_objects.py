# basics
import world_basic_blocks as blocks
from abc import ABC, abstractmethod
from typing import *
import pandas as pd
import random
import ast
import re
import math

# CHIBI framework components
import utils
import world_basic_blocks as blocks

# Fixed interact pipeline
# Step1, Get object keyword as perceptual action
# Step2, Get all possible actions and generate candidate interactive action and put them into action pool
# Step3, Select one perferred interactive action and call it
# Interactive pipeline
# Step4, CHIBI decide input: could be a tool or key information, this is decided by predefined object, and such object should have prompt to formulate input format(If the action do not need input, the Success_condition will be True or False) and this decide whether do something further written in systemic parse function. If this region is False: The action simple don't call systemic parse and won't provide useful information, the action is a trick and will waste CHIBI's time
# Step5, Do predefined semantic parse pipeline, including 1, change into another object, or change information, (no information change if the object will be deleted after the interaction) 2, generate new object. 3 delete or not. 
# Step6, Decide if action finishes, Currently there are only one possible way that the action not finish after being called, The action need input and CHIBI input incorrect result.


# The core of interaction is the information feedback, or clues gained from the interaction.
# There are four ways that an interaction will provided information.

# 1, The information of the object gained from perceptual action. Simply read the information of the object.
# 2, The information gained from the change of the object, once CHIBI interact with the object, the object change will provied information as well.(Or event changed into another object) (This part of information should be stored in the action databased based on different action)
# 3, The information gained from the systemic parse. (This is provided in the hard coded pipeline in systemic parse)
# 4, The information gained from action it self. '''Action_return_information''' (Add another degree of freedom that let agent can choose the way(tool) to take the action) This is for action that aim to gain information, All action that have this part should let Systemic parse to be False, because this action only provide the information, and not mean do systemic parse and try finish interacting with the object， This is not the same as before so the select, the action will always return the infromation given no matter success or not.


class Fixed_Interact_Pipeline_Object_Base(blocks.CHIBI_Object,ABC):
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Parse_pipeline_dict:Dict[str,Dict[str,Union[str,int]]],
                 Usage:Optional[Dict[str,int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',#gpt-4-1106-preview,
                 Belongs_to:Optional['Object'] = None,
                 Special_label = None, # only used in predefined objects
                ):
        super().__init__(Keyword, Information, 
                         Usage = Usage,
                         Model_name = Model_name,
                         Belongs_to = Belongs_to,)
        self.Parse_pipeline_dict = Parse_pipeline_dict
        self.Object_slot_1 = []
        self.Object_slot_2 = []
        self.Special_label = Special_label

    #CHIBI object interfaces:
    def edit(self, edited_information:str):
        self.Information = edited_information

    def before_perceptual_effect(self, attemptation_action:Optional['plan_system.Attemptation_Perceptual_Action'] = None):
        pass
    def after_perceptual_effect(self, attemptation_action:Optional['plan_system.Attemptation_Perceptual_Action'] = None):
        pass
    @abstractmethod
    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_recent')->str:  # add a format for GPT to follow the format of generated input
        # by default this function will only select tools acquired by current CHIBI, if the action require items but CHIBI don't have any items, this will return with Bare hands.
        # In order to let CHIBI have formated output, need to override this prompt function for different type of object. Basically, a new object means new Systemic parse and new input format
        ### This action can also be used in other parts 
        CHIBI_Objects = attemptation_action.Host_CHIBI.Profile.Items.object_retrieve()
        if len(CHIBI_Objects) == 0:
            return None # Since only when action need a tool or input will this function be called, so select None as input will lead to False, and the action will fail

        # default decide_input function do not need to use memory to generate input, so return empty list instead
        used_memories = []
        
        Prompt, Input, parse_function_str = self._prompt_template(attemptation_action)
        generated_result = attemptation_action.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = f'Interact_input_{self.Keyword}')
        return generated_result
        
    def _prompt_template(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
    # If your decide input function also need new input or format of input, please also override this function 
        CHIBI_Objects = attemptation_action.Host_CHIBI.Profile.Items.object_retrieve()
        if len(CHIBI_Objects) == 0:
            return None
        CHIBI_object_string = f'''{attemptation_action.Host_CHIBI.Name}'s current personal belongings:\n'''+'\n'.join([i.get_keyword()+":"+i.get_information() for i in CHIBI_Objects])
        if attemptation_action.Host_CHIBI.Memory_stream.Cur_assumption is None:
            cur_assumption_str = ''
        else:
            cur_assumption_str = f'''After previous exploration, you have the following assumption, which you think is the rule behind the problem you want to solve: \n{attemptation_action.Host_CHIBI.Memory_stream.get_assumption()}, please select proper item based on this assumption.\n\n'''

        Prompt = f'''Based on the following recent experience of {attemptation_action.Host_CHIBI.Name}:{attemptation_action.Belongs_to.get_information()}\n\n{attemptation_action.Host_CHIBI.Name}'s current action is: {attemptation_action.get_information()}\n\n{cur_assumption_str}Now you have the following items with you:{CHIBI_object_string}.\nIn order to success, you need to select proper tool or object that can help you fullfill your intention. Please follow the following steps to think what object you will need to success.'''
        Input = f'''**Step1** Compare all the items you have(if any), select one proper item and state your reason. **Final Step** Please copy the name of selected item in Step1 and Paste the name into a pair of parentheses, the name should be exactly the same provided previously. eg, Previously there are two items, <Note book> and <Gun>, and you think <Gun> is helpful, then you should answer ("Gun")'''
        return Prompt, Input, 'str_with_tuple'

    def judge_action_success(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        CHIBI_input = None
        selected_action_pipeline = attemptation_action.Selected_action_interactive_pipeline
        if isinstance(selected_action_pipeline['Success_condition'],bool):
            '''if do systemic parse we consider this action is success else this action is fail'''
            success_fail_state = selected_action_pipeline['Success_condition']
            # If this Value is false means we set a action that will never be success (trap), don't know if this necessary, so currently, all action that set the Success_condition to bool should be called once and will doomed to be success
        else:
            CHIBI_input = self.decide_input(attemptation_action, memory_use = memory_use)
            if CHIBI_input is None:
                success_fail_state = False
            else:
                success_fail_state = True if CHIBI_input == selected_action_pipeline['Success_condition'] else False
            attemptation_action.CHIBI_input = CHIBI_input
        attemptation_action.Success_fail_state = success_fail_state
        if CHIBI_input is not None:
            success_fail_reason = f'''{attemptation_action.Host_CHIBI.Name} tried the following action: {attemptation_action.Information}, {attemptation_action.Host_CHIBI.Name}'s decision is {CHIBI_input}, and {attemptation_action.Host_CHIBI.Name} {'succeed' if success_fail_state else 'failed'}.'''
            success_fail_reason += f'''{CHIBI_input} is {'' if success_fail_state else 'not'} the correct input.'''
            attemptation_action.Host_CHIBI.Memory_stream.memory_add(success_fail_reason, Memory_type = 'Observation')
        
    @abstractmethod
    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        '''Including systemic parse, egObject postion move, CHIBI move to a new space......'''
        '''This parse focus on the systemic level change, other than the information change of the object during the action. Once you need a completly new systemic_parse_function eg. the space structure change, some new event happens, you need to define a new object'''
        # Here we should define some universal systemic parse pipelines. (that can be used for future new objects)
        selected_action_pipeline = attemptation_action.Selected_action_interactive_pipeline
        if attemptation_action.Success_fail_state: # Semantic parse 
            if selected_action_pipeline['Systemic_parse_id'] == 'transport': # use class that have transport method to tranport the item
                # send this animal into other side of the related transportation
                cur_space = attemptation_action.Host_CHIBI.Space_manager.get_cur_space()
                linked_objects = self.find_linked_object(attemptation_action)
                transport_object = list(linked_objects.values())[0] # since only have one object
                carry_to_space = transport_object.Connected_two_space[0] if cur_space is transport_object.Connected_two_space[1] else transport_object.Connected_two_space[1]
                transport_object.transport(self, carry_to_space.Space_name, attemptation_action)
            elif selected_action_pipeline['Systemic_parse_id'] == 'grab':
                attemptation_action.Host_CHIBI.grab_item(self)
            

    @abstractmethod
    def return_action_information_construct(self, attemptation_action:Union[str,'plan_system.Attemptation_Interactive_Action'])->Dict[str,str]:
        '''Some obejct should have some unique attribute when CHIBI take certain action the action return information relys on systemic level attributes, eg the action returns how many object inside the Box......'''
        Action_return_information_dict = {'Keyword': self.Keyword,
                                                        'Information': self.Information,
                                                        'Extra_information': None}
        if isinstance(attemptation_action, str): # for statmachine action broadcast
            pass
            
        else: # input is attemptation_action instance (CHIBI interactive actions)
            Action_return_information_dict.update({'Host_CHIBI': attemptation_action.Host_CHIBI.Name})
            Action_return_information_dict.update({'Space_name': attemptation_action.Host_CHIBI.Space_manager.Cur_position})
            storage_items = list(attemptation_action.Host_CHIBI.Profile.Items.All_objects.values())
            if len(storage_items) == 0:
                storage_items_str = ''
            else:
                storage_items_str = f'''{attemptation_action.Host_CHIBI.Name} have the following items in {attemptation_action.Host_CHIBI.Name}'s pocket:''' + ', '.join([i.get_keyword() for i in storage_items]) + '\n\n'
            Action_return_information_dict.update({'CHIBI_storage':storage_items_str})
            if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'transport':
                linked_objects = self.find_linked_object(attemptation_action)
                transport_object = list(linked_objects.values())[0] # since only have one object
                CHIBI_cur_space = attemptation_action.Host_CHIBI.Space_manager.get_cur_space()
                destination_space = transport_object.Connected_two_space[0] if CHIBI_cur_space is transport_object.Connected_two_space[1] else transport_object.Connected_two_space[1]
                Action_return_information_dict.update({'Destination':destination_space.Space_name})
            success_condition = attemptation_action.Selected_action_interactive_pipeline['Success_condition']
            if not isinstance(success_condition,bool):
                Action_return_information_dict.update({'Puzzle_answer':success_condition})

        # By default the return action information will have 4 attribute variable to use. If some Object need more than one Extra_information field, then you need to align database and code.
        return Action_return_information_dict

    def update_str_with_variable(self, input_str_or_action:Union[str, 'plan_system.Attemptation_Interactive_Action'], get_action_decision:bool = False):
        if isinstance(input_str_or_action, str): # for statemachine broad action only
            action_return_str = input_str_or_action
        else: # for CHIBI interactive actions
            if get_action_decision: # the action str is used in index when selecting the actions
                action_return_str = input_str_or_action.Selected_action_interactive_pipeline['Action_str']
            else:
                action_return_str = input_str_or_action.Selected_action_interactive_pipeline['Action_return_information']
            
        if '{' not in action_return_str and '}' not in action_return_str: #  if there is no place holder
            return action_return_str
        else:
            Action_return_information_parse_dict = self.return_action_information_construct(input_str_or_action)
            return action_return_str.format(**Action_return_information_parse_dict)

    def find_linked_object(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        # find the linked object in the current Space, if don't find one then return none
        return_dict = {}
        linked_object_keyword = attemptation_action.Selected_action_interactive_pipeline['Linked_objects']
        space_manager_global = attemptation_action.Host_CHIBI.Space_Manager_System_Global
        for keyword in linked_object_keyword:
            is_found = False
            for space in space_manager_global.Vertices_dict.values():
                all_objects = space.retrieve_item_in_this_space(object_type = 'All')
                for single_object in all_objects:
                    if keyword == single_object.Keyword:
                        return_dict.update({keyword:single_object})
                        break
                        is_found = True
                if is_found is True:
                    break
        return return_dict
        
    def semantic_parse(self,
                       attemptation_action:'plan_system.Attemptation_Interactive_Action')->bool:
        '''This function should do all sorts of semantic parse including the information edit create some new objects in this steps. This parse focus on the information change on the object current interacting with'''

        # In most case this function do not need to be override since this function only handles the information change or Object change, and this part of information is predefined in the action
        
        selected_action_pipeline = attemptation_action.Selected_action_interactive_pipeline

        # edit semantic information for this object including information keyword edit and generate subparts
        information_edit_data = selected_action_pipeline['Information_edit']
        if isinstance(information_edit_data, int):
            # Create new object and delete current object and replace impression
            self.replace_object(information_edit_data, attemptation_action)
        else:
            # the edit only changes the information, and do not need to generate new object, just change the objects' information and update impression data
            print(f'information edit data: {information_edit_data}')
            self.Information = information_edit_data
            attemptation_action.Impression_object.Information = information_edit_data
        
        generated_subparts = selected_action_pipeline['Generated_subparts']
        if generated_subparts is not None:
            for subpart_id in generated_subparts:
                new_object = Fixed_Block_helper.create_fixed_object_with_database(subpart_id)
                new_object.Belongs_to = attemptation_action.Action_space
                attemptation_action.Action_space.object_add(new_object)
                # add new object into the scene
        
    def interact_pipeline(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):

        # Step1, if the action need chibi input to decide if the action is success or not.
        self.judge_action_success(attemptation_action, memory_use = memory_use)

        selected_action_pipeline = attemptation_action.Selected_action_interactive_pipeline.copy() # Need to copy the pipeline in case the object could be changed into another object 

        if attemptation_action.Success_fail_state:
            if selected_action_pipeline['Information_edit'] is not None:
                self.semantic_parse(attemptation_action)
                
            if selected_action_pipeline['Action_return_information'] is not None:
                action_return_information = self.update_str_with_variable(attemptation_action)
                attemptation_action.Host_CHIBI.Memory_stream.memory_add(action_return_information, Memory_type = selected_action_pipeline['Marker'])
                
            if selected_action_pipeline['Systemic_parse_id'] is not None:
                self.systemic_parse(attemptation_action, memory_use = memory_use)

        destory_bool = selected_action_pipeline['Destory_bool']
        if destory_bool and attemptation_action.Success_fail_state:
            self.destory()
            
        return destory_bool # will be used to modified the state of attemptations related to this object
        
    
    def replace_object(self, 
                       replace_object_id:int,
                       attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        # If the object changed its state(replaced with another object with different actions), this function will delete current object make a new object, and handling all impressions, actions
        # Step1 create new object,
        new_object = Fixed_Block_helper.create_fixed_object_with_database(replace_object_id)
        new_object.Belongs_to = self.Belongs_to
        if new_object.Belongs_to is not None:
            new_object.Belongs_to.object_add(new_object)
        if isinstance(new_object, Fixed_pipeline_Simple_Edge):
            vertices = self.Connected_two_space
            for vertex in vertices:
                vertex.object_add(new_object)
                new_object.Connected_two_space = vertices
        
        # Step2, change impression 
        previous_impression = attemptation_action.Impression_object
        previous_impression.Impression_of = new_object
        new_impression = blocks.Object_Impression(new_object.Keyword, 
                                                  new_object.Information,
                                                  new_object, 
                                                  previous_impression.Belongs_to,
                                                  previous_impression.Impression_space,
                                                  Usage = previous_impression.Usage,
                                                  Model_name = previous_impression.Model_name
                                                  )
        if isinstance(previous_impression.Belongs_to, blocks.Space_CHIBI_impression):
            previous_impression.Belongs_to.object_add(new_impression)
        previous_impression.destory()
        # Step3, delete actions related to previous imperssion and create new perceptual action
        all_actions = attemptation_action.Belongs_to.Attemptations
        # remove actions that are not finished(not tried) and created based on old impression
        actions_to_be_removed = []
        for single_action in all_actions:
            if single_action.Impression_object is previous_impression and not single_action.Success_fail_state:
                # will add action
                actions_to_be_removed.append(single_action)

        for remove_action in actions_to_be_removed:
            remove_action.destory()
        # Step4, create new impression and generate new perceptual action based on this new object, plan_system will handle this part

    def get_variable_value(self, 
                           variable_expression:str, 
                           attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return_variable_value = None
        '''Current support variables in Base class:
           Object_exist: If there is an object that have the same keyword in the current space
           Object_slot_1/2: If there is an object in the object slot, or exact the object in the slot'''
        # Since some of the object need sepefic unique class variable to decide when to call this action, so each subclass need to make sure if all variables can successfully parsed into True or False
        if ':' not in variable_expression:
            if variable_expression == 'Len_object_slot_1':
                return len(self.Object_slot_1)
            elif variable_expression == 'Len_object_slot_2':
                return len(self.Object_slot_2)
            elif variable_expression == 'Len_CHIBI_items':
                return len(attemptation_action.Host_CHIBI.Profile.get_all_items())
        else:
            variable_name = variable_expression.split(':')[0]
            variable_value = variable_expression.split(':')[1]
            if variable_name == 'Object_exist':
                all_objects_in_the_same_scope = attemptation_action.Host_CHIBI.Space_manager.get_cur_space().All_objects
                if isinstance(all_objects_in_the_same_scope, dict):
                    all_object_list = []
                    for value in all_objects_in_the_same_scope.values():
                        all_object_list.extend(value)
                    all_objects_in_the_same_scope = all_object_list
                    
                object_name_list = [i.Keyword for i in all_objects_in_the_same_scope]
                return_variable_value = variable_value in object_name_list
            elif variable_name == 'Object_slot_1':
                if len(self.Object_slot_1) == 0:
                    return_variable_value = False
                else:
                    if variable_value == '1':
                        return_variable_value = True
                    else:
                        return_variable_value = self.Object_slot_1[0].Keyword == variable_value
                        
            elif variable_name == 'Object_slot_2':
                if len(self.Object_slot_2) == 0:
                    return_variable_value = False
                else:
                    if variable_value == '1':
                        return_variable_value = True
                    else:
                        return_variable_value = self.Object_slot_1[0].Keyword == variable_value
            
            elif variable_name == 'Investigated':
                tried_action_list = [i for i in  attemptation_action.Host_CHIBI.Plan_system.Cur_state.Attemptations if i.Success_fail_state == True]
                if variable_value in [i.Impression_object.Keyword for i in tried_action_list]:
                    return_variable_value = True
                else:
                    return_variable_value = False
            elif variable_name == 'CHIBI_has':
                CHIBI_belongings = attemptation_action.Host_CHIBI.Profile.get_all_items()
                CHIBI_belongings_str_list = [i.Keyword for i in CHIBI_belongings]
                return_variable_value = True if variable_value in CHIBI_belongings_str_list else False

        # if return_variable_value is not None:
        #     print(f'condition: {variable_expression} is {return_variable_value}')

        return return_variable_value

    def action_visible(self, condition_str:str, attemptation_action:'plan_system.Attemptation_Interactive_Action')->bool:
        condition_str = attemptation_action.Selected_action_interactive_pipeline['Show_condition']
        if isinstance(condition_str, bool):
            return condition_str
        pattern = re.compile(r'\{([^}]+)\}')
        matches = pattern.findall(condition_str)
        
        for match in matches:
            result = self.get_variable_value(match, attemptation_action)
            condition_str = condition_str.replace(f'{{{match}}}', str(result))
        try:
            return eval(condition_str)
        except Exception as e:
            return f"Error evaluating expression: {e}"

# -------------------------------------------------------------Regular Objects---------------------------------------------------------------
# -------------------------------------------------------------Regular Objects---------------------------------------------------------------
# -------------------------------------------------------------Regular Objects---------------------------------------------------------------
class Fixed_pipeline_Thing(Fixed_Interact_Pipeline_Object_Base):
    # Can only be used as tools or get information or need to generate a new sub class for typical object
    # So the systemic parse only put the item into CHIBI's storage
    # Object interfaces
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    # Fixed object pipeline interfaces
    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_recent')->str:  # add a format for GPT to follow the format of generated input
        return super().decide_input(attemptation_action, memory_use = memory_use, memory_retrieve_type= memory_retrieve_type) # Regular Fixed pipeline thing will only need to dec

    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        '''some systemic level information should be changed in this step, eg object type, CHIBI open the box so the contained stuff should be exposed to every body, if CHIBI pass through the edge, CHIBI should move to the other side'''
        '''Can only be called if the action is success by judger'''
        '''Each object should have different systemic_parse pipeline'''
        '''Simple thing can only be used as tool'''
        super().systemic_parse(attemptation_action, memory_use = memory_use)

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return super().return_action_information_construct(attemptation_action)

class State_machine_object_task_monitor(Fixed_Interact_Pipeline_Object_Base):
    # CHIBI Object interfaces-----------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Required_materials = self.Special_label.split(',')
        self.Information = f'''It's a quest display that shows you the current quest you need to complete, which currently reads, "Please synthesise a unit of <{utils.sms(self.Required_materials[0])}> materials and put it into the slot under the monitor."'''

    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)

    def destory(self):
        super().destory()

    def get_keyword(self):
        return f'<{self.Keyword}>'

    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    # Regular fixed pipeline object interfaces------------------------------------------------
    def get_variable_value(self, variable_expression:str, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return_variable_value = super().get_variable_value(variable_expression, attemptation_action)
        # add some new unique variables for this class when needed
        if return_variable_value is None:
            if variable_expression == 'CHIBI_has_task_object':
                if self.Required_materials[0] in attemptation_action.Host_CHIBI.Profile.Items.All_objects.keys(): 
                    return_variable_value = True
                else: 
                    return_variable_value = False
        return return_variable_value

    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_recent')->str:  # add a format for GPT to follow the format of generated input
        return super().decide_input(attemptation_action, memory_use = memory_use, memory_retrieve_type = memory_retrieve_type) # Regular Fixed pipeline thing will only need to dec

    def return_action_information_construct(self, attemptation_action:Union['plan_system.Attemptation_Interactive_Action',str])->Dict[str,str]:
        return_information_dict = super().return_action_information_construct(attemptation_action) 
        if isinstance(attemptation_action, str):
            pass # Statemachine used variables
        else:    # Interactive action used variables
            if self.Required_materials[0] in attemptation_action.Host_CHIBI.Profile.Items.All_objects.keys(): 
                Action_select_information_task_monitor = f'''You currently have <{utils.sms(self.Required_materials[0])}> in your storage, you can submit the <{self.Required_materials[0]}> to the slot under the monitor and finish current task.'''
            else: 
                Action_select_information_task_monitor = f'''You currently do not have <{utils.sms(self.Required_materials[0])}> to submit. Please do not select this action before you successfully synthesis {utils.sms(self.Required_materials[0])}.'''
            return_information_dict.update({'Action_select_information_task_monitor':Action_select_information_task_monitor})

        return return_information_dict

    def action_visible(self, condition_str:str, attemptation_action:'plan_system.Attemptation_Interactive_Action')->bool:
        condition_str = attemptation_action.Selected_action_interactive_pipeline['Show_condition']
        if isinstance(condition_str, bool):
            return condition_str
        pattern = re.compile(r'\{([^}]+)\}')
        matches = pattern.findall(condition_str)
        
        for match in matches:
            result = self.get_variable_value(match, attemptation_action)
            condition_str = condition_str.replace(f'{{{match}}}', str(result))
        try:
            return eval(condition_str)
        except Exception as e:
            return f"Error evaluating expression: {e}"
    
    def judge_action_success(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        CHIBI_input = None
        selected_action_pipeline = attemptation_action.Selected_action_interactive_pipeline
        if isinstance(selected_action_pipeline['Success_condition'],bool):
            '''if do systemic parse we consider this action is success else this action is fail'''
            success_fail_state = selected_action_pipeline['Success_condition']
            # If this Value is false means we set a action that will never be success (trap), don't know if this necessary, so currently, all action that set the Success_condition to bool should be called once and will doomed to be success
        else:
            if self.Required_materials[0] in attemptation_action.Host_CHIBI.Profile.Items.All_objects.keys():
                success_fail_state = True
            else:
                success_fail_state = False
            #attemptation_action.CHIBI_input = CHIBI_input
        attemptation_action.Success_fail_state = success_fail_state

        if attemptation_action.Success_fail_state:
            # If CHIBI call this Function and he did have the required material
            if len(self.Required_materials) == 1:
                print(f"Success! {attemptation_action.Host_CHIBI.Name} successfully synthesised all required materials!")
                raise utils.TaskCompletedException(f"Success! {attemptation_action.Host_CHIBI.Name} successfully synthesised all required materials!")
            else:
                return_information = f'''Well done {attemptation_action.Host_CHIBI.Name}! You successfully submitted the material <{self.Required_materials[0]}>, now your next task is to synthesise and submit a unit of <{self.Required_materials[1]}>, please finish this new task and submit the required material into the slot under {self.get_keyword()}.'''
                self.Required_materials.pop(0)
                attemptation_action.Host_CHIBI.Memory_stream.memory_add(return_information, Memory_type = 'Observation', Importance_score = 20)
                
        else:
            # CHIBI call this Function but he does not have the required material
            return_information = f'''You currently do not have <{self.Required_materials[0]}> to submit, please synthesis {self.Required_materials[0]} first.'''
            attemptation_action.Host_CHIBI.Memory_stream.memory_add(return_information, Importance_score = '10')

    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        # Currently this class do not need systemic parse
        pass
        

class Fixed_pipeline_Reactor(Fixed_Interact_Pipeline_Object_Base): # Currently not in use!!!! 
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    # Fixed object pipeline interfaces
    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_recent')->str:  # add a format for GPT to follow the format of generated input
        template = self._prompt_template(attemptation_action,  memory_use = memory_use)
        if isinstance(template, str):
            print(f'''{attemptation_action.Host_CHIBI.Name} only have one object <{template}> in storage, so <{template}> is been put into the storage.''')
            return template
        else:
            Prompt, Input, parse_function_str = template
            generated_result = attemptation_action.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str)
            return generated_result

    def _prompt_template(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[str] = None):
        CHIBI_Objects = attemptation_action.Host_CHIBI.Profile.get_all_items()
        assert len(CHIBI_Objects) > 0, f'''When calling this action chibi should have at least one object in {attemptation_action.Host_CHIBI.Name}'s storage.'''
        previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str = attemptation_action.Host_CHIBI.retrieve_prompt_information(memory_use = memory_use)
        if len(CHIBI_Objects) == 1:
            return CHIBI_Objects[0].Keyword

        if len(self.Object_slot_1) == 0:
            reactor_cur_content = 'Currently the reactor is empty.'
        elif len(self.Object_slot_1) == 1:
            reactor_cur_content = f'''Currently, there is a unit of {self.Object_slot_1[0].get_keyword()} in the reactor.'''
        else:
            reactor_cur_content = f'''Currently, there are {' and '.join([i.get_keyword() for i in self.Object_slot_1])} in the reactor.'''
        Prompt = f'''{attemptation_action.Belongs_to.get_information()}\n\n{attemptation_action.Host_CHIBI.Name}'s current action is: {attemptation_action.get_information()} And you have the following information to decide what material you put into the reactor:\n\n{previous_memories_str}{most_recent_memories_str}{storage_information}{cur_assumption_str}{cur_plan_str}Please follow the following steps to generate your input.'''
        Input = f'''Please follows the following steps to decide what material should you put into the reactor.\n\n**Step1** {reactor_cur_content} Describe the synthesis formula you intend to use for this step, specify needed material you need.\n\n**Step2** {storage_information}, please ONLY SELECT ONE Material from your storage and put it into the reactor.\n\n**Step3** Please copy the name of selected material and Paste the name into a pair of angle brackets, the name should be exactly the same provided in the angle bracket "<>". For example, if you want put a unit of <CDF> into the reactor please answer <"CDF">, you can only select materials from your storage, you cannot select items that you don't have currently.'''
        return Prompt, Input, 'str_with_angle_bracket'
        
    def react(self)->List[str]:
        assert len(self.Object_slot_1) in [1,2], f'Currently there are {len(self.Object_slot_1)} items in the ractor!! there should only be 1 or 2 items!'
        if len(self.Object_slot_1) == 1:
            object_element_list = list(self.Object_slot_1[0].Keyword)
            if len(object_element_list) == 1:
                return object_element_list
            else:  # random shuffle the elements and split into two parts (decomposition reaction)
                split_point = random.randint(0, len(object_element_list))
                random.shuffle(object_element_list)
                if split_point == 0 or split_point == len(object_element_list):
                    return [''.join(object_element_list)]
                else:
                    return [''.join(object_element_list[:split_point]), ''.join(object_element_list[split_point:])]
            
        else: # there are two items in the reactor 
            react_element_1_list = list(self.Object_slot_1[0].Keyword)
            react_element_2_list = list(self.Object_slot_1[1].Keyword)
            all_elements = []
            all_elements.extend(react_element_1_list)
            all_elements.extend(react_element_2_list)
            all_element_same = True
            for i in all_elements:
                if i != all_elements[0]:
                    all_element_same = False
            element_count_same = len(react_element_1_list) == len(react_element_2_list)
            
            
            if all_element_same or element_count_same: # Currently there are two special casese, when there are only one element envolved or the length of two material is the same simply join them togather
                return [''.join(all_elements)]
            else: #Keep the head element of the longer material and replace the tail with shorter material
                longer_element_list = react_element_1_list if len(react_element_1_list)>len(react_element_2_list) else react_element_2_list
                shorter_element_list = react_element_1_list if len(react_element_1_list)<len(react_element_2_list) else react_element_2_list
                product_1 = [longer_element_list[0]]
                product_1.extend(shorter_element_list)
                product_1 = ''.join(product_1)
                product_2 = ''.join(longer_element_list[1:])
                return [product_1, product_2]

    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        super().systemic_parse(attemptation_action, memory_use = memory_use)
        selected_action_pipeline = attemptation_action.Selected_action_interactive_pipeline
        if selected_action_pipeline['Systemic_parse_id'] == 'put in slot_1':
            keyword_of_input_material = self.decide_input(attemptation_action)
            CHIBI_all_items = attemptation_action.Host_CHIBI.Profile.get_all_items()
            if keyword_of_input_material not in [i.Keyword for i in CHIBI_all_items]:
                print('----------------------------------------')
                print(keyword_of_input_material)
                print([i.Keyword for i in CHIBI_all_items])
                print('----------------------------------------')
                exception_string = f'''{attemptation_action.Host_CHIBI.Name} want to put {keyword_of_input_material} into the Reactor, but currently {attemptation_action.Host_CHIBI.Name} do not have {keyword_of_input_material} in {attemptation_action.Host_CHIBI.Name}'s storage, he need to pick it from material box or create it using the Reactor.'''
                attemptation_action.Host_CHIBI.Memory_stream.memory_add(exception_string)
                return 
            for item in attemptation_action.Host_CHIBI.Profile.get_all_items():
                if item.Keyword == keyword_of_input_material:
                    selected_material = item
                    break
            attemptation_action.Host_CHIBI.Profile.Items.object_delete(selected_material)
            self.Object_slot_1.append(selected_material)
            material_information = self.Object_slot_1[0].Keyword if len(self.Object_slot_1) <2 else self.Object_slot_1[0].Keyword + ' and ' + self.Object_slot_1[1].Keyword
            input_information = f'''You just put one unit of {keyword_of_input_material} material into the reactor, and the material in the reactor now is: {material_information}'''
            attemptation_action.Host_CHIBI.Memory_stream.memory_add(input_information)
            
        if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'react':
            assert len(self.Object_slot_1)>0, f'Currently there are no items in the ractor!! this action should not be called!'
            react_result = self.react()
            self.Object_slot_1 = [] # empty the material
            def _create_new_object(new_keyword:str)->Fixed_pipeline_Thing:
                return Fixed_Block_helper.create_fixed_object_with_database({'Object_id':0,
                                                                    'Keyword':new_keyword,
                                                                    'Information':f'''One unit of material "{new_keyword}"'''})

            for keyword in react_result:
                product = _create_new_object(keyword)
                attemptation_action.Host_CHIBI.Profile.Items.object_add(product)

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        assert not isinstance(attemptation_action, str), f'Currently reactor do not have statemachine actions, so the input shouldn\'t be a string, please debug'
        return_dict = super().return_action_information_construct(attemptation_action)
        if len(self.Object_slot_1) == 0:
            reactor_cur_content = 'Currently the reactor is empty.'
        elif len(self.Object_slot_1) == 1:
            reactor_cur_content = f'''Currently, there is a unit of {self.Object_slot_1[0].get_keyword()} in the reactor. To avoid unstable reaction, you can still put one unit of material into it.'''
        else:
            reactor_cur_content = f'''Currently, there are {' and '.join([i.get_keyword() for i in self.Object_slot_1])} in the reactor.'''
        return_dict.update({'Reactor_content_react':reactor_cur_content})

        if len(self.Object_slot_1) == 0:
            reactor_cur_content = 'Currently the reactor is empty.'
        elif len(self.Object_slot_1) == 1:
            reactor_cur_content = f'''Currently, there is a unit of {self.Object_slot_1[0].get_keyword()} in the reactor.'''
        else:
            reactor_cur_content = f'''Currently, there are {' and '.join([i.get_keyword() for i in self.Object_slot_1])} in the reactor.'''
        return_dict.update({'Reactor_content_input':reactor_cur_content+f'''Consider current item you have in your storage, and the formula you want to use in current step of reaction, do you have the item you want to put in? If not please first pick items from material box.'''})
        
        if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'react':
            reactants = [i.Keyword for i in self.Object_slot_1]
            react_result = self.react()
            
            React_result = f'''By turning on the reactor {' and '.join(reactants)} turned into {' and '.join(react_result)} after the reaction. And you put the products into your storage for later use.'''
            return_dict.update({'React_result':React_result})

        # elif attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'put in slot_1':
        #     material_information = self.Object_slot_1[0].Keyword if len(self.Object_slot_1) <2 else self.Object_slot_1[0].Keyword + 'and' + self.Objet_slot_2[1].Keyword
        #     input_information = f'''You just put one unit of material into the reactor, and currently the material in the reactor is: {material_information}'''
        #     return_dict.update({'Input_information':input_information})
        
        return return_dict

class Fixed_pipeline_Reactor_Simple(Fixed_Interact_Pipeline_Object_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # read special label, which is the basic providede raw material
        self.React_rule = int(self.Special_label.split(',')[0])
        self.Raw_materials = self.Special_label.split(',')[1:]
        
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)

    def after_perceptual_effect(self, attemptation_action:Optional['plan_system.Attemptation_Perceptual_Action']):
        # Will add all Special Object into the CHIBI's storage
        for object_keyword in self.Raw_materials:
            new_object = Fixed_Block_helper.create_fixed_object_with_database({'Object_id':0,
                                                                               'Keyword':object_keyword,
                                                                               'Information':f'''One unit of material "{object_keyword}"'''})
            attemptation_action.Host_CHIBI.Profile.Items.object_add(new_object)
            

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    # Fixed object pipeline interfaces
    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_recent')->str:  # add a format for GPT to follow the format of generated input
        template = self._prompt_template(attemptation_action,  memory_use = memory_use)
        if isinstance(template, str):
            print(f'''{attemptation_action.Host_CHIBI.Name} only have one object <{template}> in storage, so <{template}> is been put into the storage.''')
            return template
        else:
            Prompt, Input, parse_function_str, logging_label = template
            generated_result = attemptation_action.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = logging_label)
            if isinstance(generated_result, str):
                generated_result = (generated_result,)
            generated_result_list = list(generated_result)
            
            for index in range(len(generated_result_list)):
                generated_result_list[index] = utils.rsms(generated_result_list[index])
                
            return tuple(generated_result_list)

    def _prompt_template(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[str] = None):
        CHIBI_Objects = attemptation_action.Host_CHIBI.Profile.get_all_items()
        assert len(CHIBI_Objects) > 0, f'''When calling this action chibi should have at least one object in {attemptation_action.Host_CHIBI.Name}'s storage.'''
        previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str = attemptation_action.Host_CHIBI.retrieve_prompt_information(memory_use = memory_use) 
        
        Prompt = f'''{attemptation_action.Belongs_to.get_information()}\n\n{attemptation_action.Host_CHIBI.Name}'s current action is: {attemptation_action.get_information()} And you have the following information to decide what material you put into the reactor:\n\n{previous_memories_str}{most_recent_memories_str}{storage_information}{cur_assumption_str}{cur_plan_str}Please follow the following steps to decide the input.'''
        Input = f'''Please follows the steps below to decide which materials you should put into the reactor.\n\n**Step 1:**Given all the material in the storage you can use and synthetics you require to create:{storage_information}Decide which (one or two) materials you want to put into the reactor this time you can select any material from your storage, you need to clear specify the reaction you excepted and state the reason based on all given information.\n\n**Step 2:** Please copy the name of the selected material paste the name into a pair of parentheses, and separate two different materials with a comma. The name should be exactly as provided, enclosed in parentheses, for example, if you want to put a unit of X and a unit of Y into the reactor and make a reaction, please answer (X, Y), if you want to see what comes out the reactor with material <{utils.sms('XY')}> and <Z> you should answer({utils.sms('XY')}, Z). You can only choose the material that is listed in your storage. Please do not forget the parentheses!'''
        return Prompt, Input, 'str_with_tuple', 'Interact_input'
        
    def react(self)->List[str]:
        assert len(self.Object_slot_1) in [1,2], f'Currently there are {len(self.Object_slot_1)} items in the ractor!! there should only be 1 or 2 items!'
        if len(self.Object_slot_1) == 1:
            object_element_list = list(self.Object_slot_1[0].Keyword)
            if len(object_element_list) == 1:
                return object_element_list
            else:  # random shuffle the elements and split into two parts (decomposition reaction)
                split_point = random.randint(0, len(object_element_list))
                random.shuffle(object_element_list)
                if split_point == 0 or split_point == len(object_element_list):
                    return [''.join(object_element_list)]
                else:
                    return [''.join(object_element_list[:split_point]), ''.join(object_element_list[split_point:])]
            
        else: # there are two items in the reactor 
            if self.React_rule == 1:
                # Simply combine two element with the sequence put into the reactor A+B = AB
                react_element_1_list = list(self.Object_slot_1[0].Keyword)
                react_element_2_list = list(self.Object_slot_1[1].Keyword)
                return_element_list = []
                return_element_list.extend(react_element_1_list)
                return_element_list.extend(react_element_2_list)
                return [''.join(return_element_list)]
            elif self.React_rule == 2:
                # Simply combine two elements but reverse the sequence that putted into the reactor A+B = BA
                react_element_1_list = list(self.Object_slot_1[0].Keyword)
                react_element_2_list = list(self.Object_slot_1[1].Keyword)
                return_element_list = []
                return_element_list.extend(react_element_2_list)
                return_element_list.extend(react_element_1_list)
                return [''.join(return_element_list)]
                
            elif self.React_rule == 3:
                # If two element have the same length: add the second element into the middle of the first element
                # If two element is not at the same length, add the shorter element into the middle of the longer element
                # If two element i
                def _get_middle_index(input_sequence:List[str]):
                    if not len(input_sequence)%2 == 0: # input length is odd
                        return int(len(input_sequence)/2) + 1
                    else:
                        return int(len(input_sequence)/2)
                react_element_1_list = list(self.Object_slot_1[0].Keyword)
                react_element_2_list = list(self.Object_slot_1[1].Keyword)
                if len(react_element_1_list) == len(react_element_2_list):

                    add_index = _get_middle_index(react_element_1_list)
                    result_material = []
                    result_material.extend(react_element_1_list[:add_index])
                    result_material.extend(react_element_2_list)
                    result_material.extend(react_element_1_list[add_index:])
                    return [''.join(result_material)]
                else: # two material is not at the same length
                    longer_element_list = react_element_1_list if len(react_element_1_list)>len(react_element_2_list) else react_element_2_list
                    shorter_element_list = react_element_1_list if len(react_element_1_list)<len(react_element_2_list) else react_element_2_list
                    add_index = _get_middle_index(longer_element_list)
                    result_material = []
                    result_material.extend(longer_element_list[:add_index])
                    result_material.extend(shorter_element_list)
                    result_material.extend(longer_element_list[add_index:])
                    return [''.join(result_material)]
                
            elif self.React_rule == 4:
                # If two element have the same length combine them
                # If two element have different length, keep the head of the longer element and change the rest with the shorter element
                # If two element have different length, but only have one letter material, combine them
                react_element_1_list = list(self.Object_slot_1[0].Keyword)
                react_element_2_list = list(self.Object_slot_1[1].Keyword)
                all_elements = []
                all_elements.extend(react_element_1_list)
                all_elements.extend(react_element_2_list)
                all_element_same = True
                for i in all_elements:
                    if i != all_elements[0]:
                        all_element_same = False
                element_count_same = len(react_element_1_list) == len(react_element_2_list)
                
                
                if all_element_same or element_count_same: # Currently there are two special casese, when there are only one element envolved or the length of two material is the same simply join them togather
                    return [''.join(all_elements)]
                else: #Keep the head element of the longer material and replace the tail with shorter material
                    longer_element_list = react_element_1_list if len(react_element_1_list)>len(react_element_2_list) else react_element_2_list
                    shorter_element_list = react_element_1_list if len(react_element_1_list)<len(react_element_2_list) else react_element_2_list
                    product_1 = [longer_element_list[0]]
                    product_1.extend(shorter_element_list)
                    product_1 = ''.join(product_1)
                    product_2 = ''.join(longer_element_list[1:])
                    return [product_1, product_2]

    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        super().systemic_parse(attemptation_action, memory_use = memory_use)
        def _create_new_object(new_keyword:str)->Fixed_pipeline_Thing:
                return Fixed_Block_helper.create_fixed_object_with_database({'Object_id':0,
                                                                    'Keyword':new_keyword,
                                                                    'Information':f'''One unit of material "{new_keyword}"'''})
        selected_action_pipeline = attemptation_action.Selected_action_interactive_pipeline
            
        if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'react':
            react_decide_material_tuple = self.decide_input(attemptation_action)
            #print(react_decide_material_tuple)
            for selected_material_keyword in react_decide_material_tuple:
                if selected_material_keyword not in attemptation_action.Host_CHIBI.Profile.Items.All_objects:
                    error_experience = f'''You tried to put <{utils.sms(selected_material_keyword)}> into the <Reactor>, but you currently do not have such material in your storage.'''
                    attemptation_action.Host_CHIBI.Memory_stream.memory_add(error_experience, Importance_score = 15)
                    self.Object_slot_1 = []
                    return
                new_object = _create_new_object(selected_material_keyword)
                self.Object_slot_1.append(new_object)

            
            reactants = [i.Keyword for i in self.Object_slot_1]

            if len(reactants) >2:
                React_result = f'''You selected {','.join([i.get_keyword() for i in self.Object_slot_1])} to react, there are more than 2 reactants in the reactor, you should only put at most two reactants per reaction.'''
                attemptation_action.Host_CHIBI.Memory_stream.memory_add(React_result, Importance_score = 15)
                self.Object_slot_1 = []
                return 

            
            react_result = self.react()
            react_result_show = []
            for single_material in react_result:
                react_result_show.append(utils.sms(single_material))
            reactants_show = []
            for single_material in reactants:
                reactants_show.append(utils.sms(single_material))
            
            React_result = f'''By turning on the reactor {' and '.join(reactants_show)} turned into {' and '.join(react_result_show)} after the reaction. And you put the products into your storage for later use.'''
            attemptation_action.Host_CHIBI.Memory_stream.memory_add(React_result, Memory_type = 'Observation', Importance_score = 15)
            
            self.Object_slot_1 = [] # empty the material
            
            for keyword in react_result:
                product = _create_new_object(keyword)
                attemptation_action.Host_CHIBI.Profile.Items.object_add(product)

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        assert not isinstance(attemptation_action, str), f'Currently reactor do not have statemachine actions, so the input shouldn\'t be a string, please debug'
        return_dict = super().return_action_information_construct(attemptation_action)
        if len(self.Object_slot_1) == 0:
            reactor_cur_content = 'Currently the reactor is empty.'
        elif len(self.Object_slot_1) == 1:
            reactor_cur_content = f'''Currently, there is a unit of {self.Object_slot_1[0].get_keyword()} in the reactor. To avoid unstable reaction, you can still put one unit of material into it.'''
        else:
            reactor_cur_content = f'''Currently, there are {' and '.join([i.get_keyword() for i in self.Object_slot_1])} in the reactor.'''
        return_dict.update({'Reactor_content_react':reactor_cur_content})

        if len(self.Object_slot_1) == 0:
            reactor_cur_content = 'Currently the reactor is empty.'
        elif len(self.Object_slot_1) == 1:
            reactor_cur_content = f'''Currently, there is a unit of {self.Object_slot_1[0].get_keyword()} in the reactor.'''
        else:
            reactor_cur_content = f'''Currently, there are {' and '.join([i.get_keyword() for i in self.Object_slot_1])} in the reactor.'''
        return_dict.update({'Reactor_content_input':reactor_cur_content+f'''Consider current item you have in your storage, and the formula you want to use in current step of reaction, do you have the item you want to put in? If not please first pick items from material box.'''})
        
        if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'react':
            pass

        # elif attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'put in slot_1':
        #     material_information = self.Object_slot_1[0].Keyword if len(self.Object_slot_1) <2 else self.Object_slot_1[0].Keyword + 'and' + self.Objet_slot_2[1].Keyword
        #     input_information = f'''You just put one unit of material into the reactor, and currently the material in the reactor is: {material_information}'''
        #     return_dict.update({'Input_information':input_information})
        return return_dict

class Fixed_pipeline_Function_Operator_one_variable(Fixed_Interact_Pipeline_Object_Base):
    # all functions are consisted with constant and the following monomials: x, sin(x), x^2, abs(x), x^-1, so there are a,b,c,d,e,f variables
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_functions()
        function_information = self._init_action_pipelines()
        all_parameters = []
        for coefficient_list in self.Function_coefficient_map.values():
            for coefficient_name in coefficient_list:
                if coefficient_name not in all_parameters:
                    all_parameters.append(coefficient_name)
        all_parameter_str = ', '.join(all_parameters)
        monomials_str_dict = {'sin(x)' : "'sin(x)'",
                              '1/x' : "'1/x'",
                              'x^2' : '''"x^2" (square of x)''',
                              '|x|' : "'|x|' (absolute value of x)",
                              'x' : "'x'",
                              '-x' : "'-x' negative x"}
        monomials_information = ','.join([monomials_str_dict[monomial] for monomial in list(self.Final_used_monomials.keys())[:-1]])
        if len(self.Final_used_monomials) >= 2:
            monomials_information += ' and ' + monomials_str_dict[list(self.Final_used_monomials.keys())[-1]]
        elif len(self.Final_used_monomials) == 1: # only one monomail is used
            monomials_information = f'''Only {monomials_str_dict[list(self.Final_used_monomials.keys())[0]]} appears in these functions.'''
        elif len(self.Final_used_monomials) == 0:
            monomials_information = f'''There are only {len(self.Coefficient_map)} constants in current puzzle, no monomials.'''
            
        if len(self.Function_map) == 1:
            self.Information = f'''This {self.get_keyword()} consists of one univariate function, with one variable, x. The function is composed of one or more of the following monomials: {monomials_information}. All functions share a set of constant parameters, which are integers ranging from 1 to 9 (inclusive) and are listed here: {{{all_parameter_str}}}, Each parameter is unique and can appear multiple times in different functions either a coefficient or an additive constant(eg, Function #x could have term b*x, and Function #y could have a constant term b*x + b, and Function #z could have a term b*sin(x)). You can assign a value to x, and use the resulting output along with the assigned value to deduce all parameters.'''
        else:  
            self.Information = f'''This {self.get_keyword()} consists of {len(self.Function_map)} different univariate functions, each with one variable, x. Each function is composed of one or more of the following monomials: {monomials_information}. All functions share a set of constant parameters, which are integers range from 1 to 9 (inclusive) and are listed here: {{{all_parameter_str}}}, Each parameter is unique and can appear multiple times in different function either a coefficient or an additive constant(eg, Function #x could have term b*x, and Function #y could have a constant term b*x + b, and Function #z could have a term b*sin(x)). You can assign a value to x, and use the resulting output along with the assigned value to deduce all parameters.'''
        self.Information += f'''And the following are the information of all the functions you need to solve: \n{function_information}'''

    def _init_action_pipelines(self):
        pipeline_action_template = {'Action_str': 'Read the {Keyword}.', 
                                    'Destory_bool': False, 
                                    'Show_condition': "{Investigated:Code secured door}", 
                                    'Success_condition': True, 
                                    'Systemic_parse_id': None, 
                                    'Information_edit': None, 
                                    'Action_return_information': '', 
                                    'Generated_subparts': None, 
                                    'Linked_objects': [], 
                                    'Repeat_action': True, 
                                    'Marker': 'Observation'}
        function_information = ''
        for index, item in enumerate(self.Function_map.items()):
            show_agent_function_str = item[0]
            if len(self.Function_coefficient_map[show_agent_function_str]) == 0:
                assert False, f'''This should not happen, check coefficients and constants in the Special_label'''
            coefficient_informaiton = ', '.join(self.Function_coefficient_map[show_agent_function_str])
            term_count = len(self.Function_coefficient_map[show_agent_function_str])
            
            if self.Show_formula_bool:
                Action_str = f'''Assign a value to x variable and get the output of the following function: f(x) = {show_agent_function_str}. This function have {term_count} terms and following parameters: {coefficient_informaiton}.'''
                Action_return_information = f'''You assign the value {{CHIBI_input}} to x of the function: f(x) = {show_agent_function_str}, and then the function outputs {{Function_output}}'''
                Function_str = show_agent_function_str
                function_information += f'''Function No.{index+1}: f(x) = {show_agent_function_str}.\n'''
            else:
                Action_str = f'''Assign a value to the variable of Function #{index+1} and see the output. Function #{index+1} have {term_count} terms and the following parameters(Could be constant or coefficients): {self.Function_coefficient_map[show_agent_function_str]}.'''
                Action_return_information = f'''You assign the value {{CHIBI_input}} to x of the function #{index+1}, and then the function outputs {{Function_output}}. (Function #{index+1} have {term_count} terms and the following parameters(Could be constant or coefficients): {self.Function_coefficient_map[show_agent_function_str]}.)'''
                function_information += f'''Function #{index+1} have {term_count} terms and the following parameters(Could be constant or coefficients): {self.Function_coefficient_map[show_agent_function_str]}.\n'''
                Function_str = f'Function #{index+1}'
            cur_action_pipeline_dict = pipeline_action_template.copy()
            cur_action_pipeline_dict.update({'Action_str':Action_str})
            cur_action_pipeline_dict.update({'Action_return_information':Action_return_information})
            cur_action_pipeline_dict.update({'Function_str':Function_str})
            cur_action_pipeline_dict.update({'Calculate_function_str':item[1]})
            self.Parse_pipeline_dict.update({Action_str:cur_action_pipeline_dict}) 
        return function_information
    
    def _init_functions(self):
        function_strs = self.Special_label.split(',')
        self.Show_formula_bool = ast.literal_eval(function_strs[0]) # If show this function to agent this is oracle setting
        function_strs = function_strs[1:]
        monomials_mapping_dict = {'sin(x)' : 'math.sin({x})',
                                  '1/x' : '1/{x}',
                                  'x^2' : '({x})**2',
                                  '|x|' : 'abs({x})',
                                  'x' : '{x}',
                                  '-x' : '(-{x})'}
        monomials_used_dict = {}
        alphabetic_letters = [chr(i) for i in range(97, 123)]
        coefficients_map = {} #'letter':int
        functions_map = {} #function_show:function_calculate
        functions_coefficient_map = {} #function_show: list[coefficients]
        for function_str in function_strs: # for every Function
            function_str = re.sub(r'[\n]+', '', function_str)
            function_str_clean = re.sub(r'[ \n]+', '', function_str)
            calcualte_function_str = ''
            show_agent_function_str = ''
            coefficient_names = []
            for term in function_str_clean.split('+'): 
                if '*' in term:
                    coefficient, literial = term.split('*')
                    if int(coefficient) not in coefficients_map.values():
                        coefficient_name = alphabetic_letters.pop(0)
                        coefficients_map.update({coefficient_name: int(coefficient)})
                    else:
                        for item in coefficients_map.items():
                            if item[1] == int(coefficient):
                                coefficient_name = item[0]
                                break 
                    calcualte_function_str += '{' + coefficient_name + '}' + '*' + monomials_mapping_dict[literial]
                    monomials_used_dict.update({literial:monomials_mapping_dict[literial]})
                    show_agent_function_str += coefficient_name + '*' + literial
                    coefficient_names.append(coefficient_name)
                else:
                    try:
                        coefficient = int(term)
                        if int(coefficient) not in coefficients_map.values():
                            coefficient_name = alphabetic_letters.pop(0)
                            coefficients_map.update({coefficient_name: int(coefficient)})
                        else:
                            for item in coefficients_map.items():
                                if item[1] == int(coefficient):
                                    coefficient_name = item[0]
                                    break 
                        show_term = coefficient_name
                        calculate_term = '{' + coefficient_name + '}'
                        coefficient_names.append(coefficient_name)
                    except ValueError:
                        show_term = term
                        calculate_term = monomials_mapping_dict[term]
                        monomials_used_dict.update({term:monomials_mapping_dict[term]})

                    calcualte_function_str += calculate_term
                    show_agent_function_str += show_term
                calcualte_function_str += '+'
                show_agent_function_str += ' + '
            functions_map.update({show_agent_function_str[:-3]:calcualte_function_str[:-1]})
            functions_coefficient_map.update({show_agent_function_str[:-3]:coefficient_names})
        self.Function_map = functions_map
        self.Coefficient_map = coefficients_map
        self.Function_coefficient_map = functions_coefficient_map
        self.Final_used_monomials = monomials_used_dict

    def reassign_variable_values(self):
        # to avoide the puzzle is calculated by brute force, there is a limit of fail, if exceed the limit, will assign all of the variables into new value
        parameter_count = len(self.Coefficient_map)
        new_values = random.sample(range(1, 10), parameter_count)
        new_coefficients_map = {}
        for index, parameter_name in enumerate(self.Coefficient_map.keys()):
            new_coefficients_map.update({parameter_name:new_values[index]})
        self.Coefficient_map = new_coefficients_map

    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    # Fixed object pipeline interfaces
    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_relevant')->str:  # add a format for GPT to follow the format of generated input
        Prompt, Input, parse_function_str, logging_label = self._prompt_template(attemptation_action, memory_use = memory_use)
        decided_input = attemptation_action.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = logging_label)
        return str(decided_input) # simple thing do not need to have input and the success and out are fixed

    def _prompt_template(self, 
                         attemptation_action:'plan_system.Attemptation_Interactive_Action', 
                         memory_use:Optional[int] = None):
        previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str = attemptation_action.Host_CHIBI.retrieve_prompt_information(memory_use = memory_use)
        Prompt = f'''Based on the following recent experience of {attemptation_action.Host_CHIBI.Name}:{attemptation_action.Belongs_to.get_information()}\n\n{attemptation_action.Host_CHIBI.Name}'s current action is: {attemptation_action.get_information()} And you have the following information to decide what is the value you want to assign to the variable in {attemptation_action.Selected_action_interactive_pipeline['Function_str']}:\n\n{previous_memories_str}{most_recent_memories_str}{cur_assumption_str}{cur_plan_str}Please follow the following steps to generate your final answer.'''
        Input = f'''**Step 1:** Reflect on your recent experience and consider which value for the variable 'x' would be most helpful in decoding the parameters. State your reason of choosing this value. **Step 2:** Please enter the value you wish to assign to 'x', enclosed in square brackets, you can input 'pi', '0.5*pi', 'any_integer*pi', 'any_decimal*pi', any decimal, and any integer. For example, if you want to assign the value 12 to 'x', you should type ['12']. If you want to assign the value 0.5pi to 'x', you should type ['0.5*pi']. Please do not use fraction and "/" mark, please use decimal multiplication instead.'''
        return Prompt, Input, 'str_with_square_bracket', 'Interact_input'

    def generate_function_output(self, function_expression:str, CHIBI_input:str):
        CHIBI_input = CHIBI_input.lower()
        try:
            value_dict = self.Coefficient_map.copy()
            if CHIBI_input == 'pi':
                CHIBI_input = math.pi
            elif 'pi' in CHIBI_input:
                if '*' not in CHIBI_input:
                    CHIBI_input = CHIBI_input.replace('pi','*math.pi')
                    CHIBI_input = eval(CHIBI_input)
                else: #'*' in CHIBI_input
                    CHIBI_input = CHIBI_input.replace('pi','math.pi')
                    CHIBI_input = eval(CHIBI_input)
            elif 'π' in CHIBI_input:
                if '*' not in CHIBI_input:
                    CHIBI_input = CHIBI_input.replace('π','*math.pi')
                    CHIBI_input = eval(CHIBI_input)
                else: #'*' in CHIBI_input
                    CHIBI_input = CHIBI_input.replace('π','math.pi')
                    CHIBI_input = eval(CHIBI_input)
    
            CHIBI_input_dict = {'x':CHIBI_input}
            value_dict.update(CHIBI_input_dict)
            return_value = eval(function_expression.format(**value_dict))
            if abs(float(return_value) - 0) < 0.000001:
                return_value = 0
        except ZeroDivisionError:
            return_value = f'''Error, your input cause the function divided by zero!'''
        except SyntaxError:
            return_value = f'''Error, your input "{CHIBI_input}" is not valid. Please only input decimals, integers, and random decimal * pi'''
            raise SyntaxError(f'''Error, your input "{CHIBI_input}" is not valid. Please only input decimals, integers, and random decimal * pi''')
        except NameError:
            print(value_dict)
            return_value = f'''Error, your input "{CHIBI_input}" is not valid. Please only input decimals, integers, and random decimal * pi'''
            raise SyntaxError(f'''Error, your input "{CHIBI_input}" is not valid. Please only input decimals, integers, and random decimal * pi''')
        except TypeError:
            return_value = f'''Error, your input "{CHIBI_input}" is not a valid input, please only input decimals, integers, and random decimal * pi'''
            raise SyntaxError(f'''Error, your input "{CHIBI_input}" is not valid. Please only input decimals, integers, and random decimal * pi''')
            
        return return_value
    
    def return_action_information_construct(self, attemptation_action:Union[str, 'plan_system.Attemptation_Interactive_Action']):
        action_return_information_dict = super().return_action_information_construct(attemptation_action)
        CHIBI_input = self.decide_input(attemptation_action)
        action_return_information_dict.update({'CHIBI_input':CHIBI_input})
        calcualte_function_str = attemptation_action.Selected_action_interactive_pipeline['Calculate_function_str']
        Function_output = self.generate_function_output(calcualte_function_str,CHIBI_input)
        action_return_information_dict.update({'CHIBI_input':CHIBI_input})
        action_return_information_dict.update({'Function_output':Function_output})
        return action_return_information_dict

        
    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        '''some systemic level information should be changed in this step, eg object type, CHIBI open the box so the contained stuff should be exposed to every body, if CHIBI pass through the edge, CHIBI should move to the other side'''
        '''Can only be called if the action is success by judger'''
        '''Each object should have different systemic_parse pipeline'''
        '''Simple thing can only be used as tool'''
        super().systemic_parse(attemptation_action, memory_use = memory_use)


class Fixed_pipeline_Thing_Creator(Fixed_Interact_Pipeline_Object_Base):
    # 'd better not create contianer object and edge object 
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    # Fixed object pipeline interfaces
    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type = 'Most_recent')->str:  # add a format for GPT to follow the format of generated input
        return super().decide_input(attemptation_action, memory_use = memory_use) # Regular Fixed pipeline thing will only need to dec

        
    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        '''some systemic level information should be changed in this step, eg object type, CHIBI open the box so the contained stuff should be exposed to every body, if CHIBI pass through the edge, CHIBI should move to the other side'''
        '''Can only be called if the action is success by judger'''
        '''Each object should have different systemic_parse pipeline'''
        '''Simple thing can only be used as tool'''
        super().systemic_parse(attemptation_action, memory_use = memory_use)
        if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'get item':
            # currently only create first item
            object_keyword = attemptation_action.Selected_action_interactive_pipeline['Linked_objects'][0]
            new_object = Fixed_Block_helper.create_fixed_object_with_database({'Object_id':0,
                                                                               'Keyword':object_keyword,
                                                                               'Information':f'''One unit of material "{object_keyword}"'''})
            attemptation_action.Host_CHIBI.Profile.Items.object_add(new_object)
            

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return super().return_action_information_construct(attemptation_action)
        
# -------------------------------------------------------------Container Objects-------------------------------------------------------------
# -------------------------------------------------------------Container Objects-------------------------------------------------------------
# -------------------------------------------------------------Container Objects-------------------------------------------------------------
class Fixed_pipeline_Simple_Container(Fixed_Interact_Pipeline_Object_Base):
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Parse_pipeline_dict:Dict[str,Dict[str,Union[str,int]]],
                 All_objects:Optional[Dict[str,Any]] = None,
                 Usage:Optional[Dict[str,int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',#gpt-4-1106-preview,
                 Belongs_to:Optional['Object'] = None,
                ):
        super().__init__(Keyword, Information, 
                         Usage = Usage,
                         Model_name = Model_name,
                         Belongs_to = Belongs_to,
                         Parse_pipeline_dict = Parse_pipeline_dict)
        if All_objects is None:
            self.All_objects = []
        else:
            self.All_objects = All_objects
    
    # Object interfaces
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        # relase_all_container_stuff
        release_objects = self.All_objects
        for single_container_stuff in release_objects:
            self.Belongs_to.object_add(single_container_stuff)
            single_container_stuff.Belongs_to = self.Belongs_to
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_recent')->str:
        return super().decide_input(attemptation_action, memory_use =memory_use, memory_retrieve_type = memory_retrieve_type)

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return super().return_action_information_construct(attemptation_action)
        
    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None)->str:
        # for a simple container if action require systemic parse means action successfully opened the container and reveal all items inside
        super.systemic_parse(attemptation_action, memory_use = memory_use)
        if attemptation_action.selected_action_pipeline['Systemic_parse_id'] == 'open':
            contained_stuffs = self.All_objects
            if len(contained_stuffs) == 0:
                open_experience = f'You opened the {self.get_keyword()}, unfortunately the {self.get_keyword()} is empty, there nothing in it!'
            else:
                keywords = [i.get_keyword() for i in contained_stuffs]
                open_experience = f'''You opened the {self.get_keyword()}, you find the following items in the {self.get_keyword()}: {','.join(keywords)}.'''
                for container_stuff in contained_stuffs:
                    # create impression for the object
                    impression_space = attemptation_action.Host_CHIBI.Space_manager.get_cur_space(space_type = 'impression')
                    information = f'You find {container_stuff.get_keyword()} from {self.get_keyword()}, you haven\'t investigated it closely yet'
                    impression_object = blocks.Object_Impression(container_stuff.Keyword,information,container_stuff,self,impression_space)
                attemptation_action.Host_CHIBI.Memory_stream.memory_add(open_experience)

    def object_add(self, Object_to_be_aded):
        self.All_objects.append(Object_to_be_aded)
        Object_to_be_aded.Belongs_to = self
        
    def object_delete(self, Object_to_be_delete):
        self.All_objects.remove(Object_to_be_delete)
        Object_to_be_delete.Belongs_to = None


# -------------------------------------------------------------Edge Objects---------------------------------------------------------------
# -------------------------------------------------------------Edge Objects---------------------------------------------------------------
# -------------------------------------------------------------Edge Objects---------------------------------------------------------------
class Fixed_pipeline_Simple_Edge(Fixed_Interact_Pipeline_Object_Base):
    # this is an simple edge
    # Simple edge 下联通的两个edge都会加上这个edge object， CHIBI只会有一个impression 生成的action space 也会根据之前的方向来决定 # 移动中增加新的内容的东西先不考虑
    # 如果是后续需要拓展的情况，两边的edge object对应的不同的可能的function。而且是一个pair。那样的话就会创建两个impression
    # 首先这个Edge存在的意义就是设置中间谜题，增加空间复杂度，设计关卡上多了一些自由度更加贴近现实
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Parse_pipeline_dict:Dict[str,Dict[str,Union[str,int]]],
                 Connected_two_space:Optional[List[blocks.Space_System_global]] = None,
                 Usage:Optional[Dict[str,int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',#gpt-4-1106-preview,
                 Belongs_to:Optional['Object'] = None,
                 Special_label:Optional[str] = None,
                ):
        super().__init__(Keyword, Information, 
                         Usage = Usage,
                         Model_name = Model_name,
                         Belongs_to = Belongs_to,
                         Parse_pipeline_dict = Parse_pipeline_dict,
                         Special_label = Special_label)
        #assert len(Connected_two_space) == 2, f'length of connedted space should be exactly 2 but {len(Connected_two_space)} get'
        if Connected_two_space is None:
            self.Connected_two_space = []
            #print(f'When creating {self.get_keyword()}, the connected space is not assigned, please make sure each edge do have connected to 2 spaces by adding them later')
        else:
            self.Connected_two_space = Connected_two_space

    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        # relase_all_container_stuff
        space_1 = self.Connected_two_space[0]
        space_2 = self.Connected_two_space[1]
        space_1.object_delete(self)
        space_2.object_delete(self)
        
    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_recent')->str:
        return super().decide_input(attemptation_action, memory_use = memory_use, memory_retrieve_type = memory_retrieve_type)

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return_dict =  super().return_action_information_construct(attemptation_action)
        if isinstance(attemptation_action, str): # for state machie state action
            pass
        else: # attemptation_action is a action taken by CHIBI
            cur_space_name = attemptation_action.Impression_object.Impression_space.Space_name
            other_side_name = self.Connected_two_space[1] if self.Connected_two_space[0].Space_name == cur_space_name else self.Connected_two_space[0]
            return_dict.update({'Other_side_space_name':other_side_name.Space_name})
        return return_dict
        
    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None)->str:
        # for a simple edge if action require systemic parse means CHIBI successfully passed the edge and explore the new space
        super().systemic_parse(attemptation_action, memory_use = memory_use)
        if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'try pass':
            cur_space = attemptation_action.Host_CHIBI.Space_manager.get_cur_space()
            move_to_space = self.Connected_two_space[0] if cur_space is self.Connected_two_space[1] else self.Connected_two_space[1]
            move_to_str = move_to_space.Space_name
            attemptation_action.Host_CHIBI.move(move_to_str,)# already created an space_impression here if the space is a new space

class Fixed_pipeline_Simple_Boat(Fixed_pipeline_Simple_Edge):
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Parse_pipeline_dict:Dict[str,Dict[str,Union[str,int]]],
                 Connected_two_space:Optional[List[blocks.Space_System_global]] = None,
                 Usage:Optional[Dict[str,int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',#gpt-4-1106-preview,
                 Belongs_to:Optional['Object'] = None,
                ):
        super().__init__(Keyword, Information, 
                         Usage = Usage,
                         Model_name = Model_name,
                         Belongs_to = Belongs_to,
                         Parse_pipeline_dict = Parse_pipeline_dict)
        #assert len(Connected_two_space) == 2, f'length of connedted space should be exactly 2 but {len(Connected_two_space)} get'
        if Connected_two_space is None:
            self.Connected_two_space = []
            #print(f'When creating {self.get_keyword()}, the connected space is not assigned, please make sure each edge do have connected to 2 spaces by adding them later')
        else:
            self.Connected_two_space = Connected_two_space

    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        # relase_all_container_stuff
        space_1 = self.Connected_two_space[0]
        space_2 = self.Connected_two_space[1]
        space_1.object_delete(self)
        space_2.object_delete(self)
        
    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_recent')->str:
        return super().decide_input(attemptation_action, memory_use = memory_use, memory_retrieve_type = memory_retrieve_type)

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return super().return_action_information_construct(attemptation_action)

    def transport(self, carry_object:'CHIBI_object', carry_to_space:str, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        to_space = self.Connected_two_space[0] if carry_to_space==self.Connected_two_space[0].Space_name else self.Connected_two_space[1]
        from_space = self.Connected_two_space[1] if carry_to_space==self.Connected_two_space[0].Space_name else self.Connected_two_space[0]
        to_space.object_add(carry_object)

        # handle impression objects
        attemptation_action.Host_CHIBI.move(to_space.Space_name)
        impression_object = attemptation_action.Impression_object
        from_space_impression = attemptation_action.Host_CHIBI.Space_manager.Vertices_dict[from_space.Space_name]
        to_space_impression = attemptation_action.Host_CHIBI.Space_manager.Vertices_dict[to_space.Space_name]
        from_space_impression.object_delete(impression_object)
        to_space_impression.object_add(impression_object)
        impression_object.Impression_space = to_space_impression
        
    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None)->str:
        # for a simple edge if action require systemic parse means CHIBI successfully passed the edge and explore the new space
        if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'try pass':
            cur_space = attemptation_action.Host_CHIBI.Space_manager.get_cur_space()
            move_to_space = self.Connected_two_space[0] if cur_space is self.Connected_two_space[1] else self.Connected_two_space[1]
            move_to_str = move_to_space.Space_name
            attemptation_action.Host_CHIBI.move(move_to_str,)# already created an space_impression here if the space is a new space
        else:
            assert False, f'{attemptation_action.get_information()} required systemic parse do not exist.'


# -------------------------------------------------------------Puzzle Objects---------------------------------------------------------------
# -------------------------------------------------------------Puzzle Objects---------------------------------------------------------------
# -------------------------------------------------------------Puzzle Objects---------------------------------------------------------------
class Fixed_Puzzle_Object_Base(Fixed_Interact_Pipeline_Object_Base):
    '''This is a class of object that have one action have systemic parse that marked as finish of the whole test'''
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Parse_pipeline_dict:Dict[str,Dict[str,Union[str, int]]],
                 Usage:Optional[Dict[str, int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Belongs_to:Optional['Object'] = None,
                 Puzzle_answer:Optional[str] = None,
                 Special_label:Optional[str] = None, 
                ):
        super().__init__(Keyword, Information,
                         Usage = Usage,
                         Model_name = Model_name,
                         Belongs_to = Belongs_to,
                         Parse_pipeline_dict = Parse_pipeline_dict,
                         Special_label = Special_label)

        raise_error_flag = False
        if Puzzle_answer is not None:
            for single_action in self.Parse_pipeline_dict.values():
                if not isinstance(single_action['Success_condition'], bool): # Puzzle action that leads to finish state will have same action
                    single_action['Success_condition'] = Puzzle_answer
                    raise_error_flag = True
        assert raise_error_flag or Puzzle_answer is None, f'''The puzzle object have override puzzle answer, but do not have action that need the variable.'''
    
class Fixed_pipeline_code_secured_box(Fixed_Puzzle_Object_Base):
    # Object interfaces
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_relevant')->str:  # add a format for GPT to follow the format of generated input
        
        Prompt, Input, parse_function_str, logging_label = self._prompt_template(attemptation_action, memory_use = memory_use)
        decided_input = attemptation_action.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = logging_label)
        return str(decided_input) # simple thing do not need to have input and the success and out are fixed

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return super().return_action_information_construct(attemptation_action)
        
    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        '''some systemic level information should be changed in this step, eg object type, CHIBI open the box so the contained stuff should be exposed to every body, if CHIBI pass through the edge, CHIBI should move to the other side'''
        '''Can only be called if the action is success by judger'''
        '''Each object should have different systemic_parse pipeline'''
        '''Simple thing can only be used as tool'''
        super().systemic_parse(attemptation_action, memory_use = memory_use)
        pass
        # this object do not need systemic parse

    def _prompt_template(self, 
                         attemptation_action:'plan_system.Attemptation_Interactive_Action', 
                         memory_use:Optional[int] = None):
        previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str = attemptation_action.Host_CHIBI.retrieve_prompt_information(memory_use = memory_use)
        Prompt = f'''Based on the following recent experience of {attemptation_action.Host_CHIBI.Name}:{attemptation_action.Belongs_to.get_information()}\n\n{attemptation_action.Host_CHIBI.Name}'s current action is: {attemptation_action.get_information()} And you have the following information to decide what is the correct password:\n\n{previous_memories_str}{most_recent_memories_str}{storage_information}{cur_assumption_str}{cur_plan_str}Please follow the following steps to generate your final answer.'''
        Input = f'''**Step1** refelect the recent experience, what do you think is the password to {self.get_keyword()} is? Please only use information provided to do inference and give your reason. **Final Step** Please generate your final answer in a pair of square brackets. eg, if you think the final pass word is '112233' you should output ['112233'], if you think the output is '3a1b45v' please output ['3a1b45v'].'''
        return Prompt, Input, 'str_with_square_bracket', 'Interact_input'
        
class Fixed_pipeline_code_secured_door(Fixed_Puzzle_Object_Base):

    # Object interfaces
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_relevant')->str:  # add a format for GPT to follow the format of generated input
        Prompt, Input, parse_function_str = self._prompt_template(attemptation_action, memory_use = memory_use)
        decided_input = attemptation_action.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str)
        return str(decided_input) # simple thing do not need to have input and the success and out are fixed

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return super().return_action_information_construct(attemptation_action)
        
    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        '''some systemic level information should be changed in this step, eg object type, CHIBI open the box so the contained stuff should be exposed to every body, if CHIBI pass through the edge, CHIBI should move to the other side'''
        '''Can only be called if the action is success by judger'''
        '''Each object should have different systemic_parse pipeline'''
        '''Simple thing can only be used as tool'''
        super().systemic_parse(attemptation_action, memory_use = memory_use)
        if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'try finish':
            raise utils.TaskCompletedException(f'''Mission Complete, CHIBI found the correct code for the door!''')

    def _prompt_template(self, 
                        attemptation_action:'plan_system.Attemptation_Interactive_Action', 
                        memory_use:Optional[int] = None):
        previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str = attemptation_action.Host_CHIBI.retrieve_prompt_information(memory_use = memory_use)
        Prompt = f'''Based on the following recent experience of {attemptation_action.Host_CHIBI.Name}:{attemptation_action.Belongs_to.get_information()}\n\n{attemptation_action.Host_CHIBI.Name}'s current action is: {attemptation_action.get_information()} And you have the following information to decide what is the correct password:\n\n{previous_memories_str}{most_recent_memories_str}{storage_information}{cur_assumption_str}{cur_plan_str}Please follow the following steps to generate your final answer.'''
        Input = f'''**Step1** refelect the recent experience, what do you think is the password to {self.get_keyword()} is? Please only use information provided to do inference and give your reason. **Final Step** Please generate your final answer in a pair of square brackets. eg, if you think the final pass word is '112233' you should output ['112233'], if you think the output is '3a1b45v' please output ['3a1b45v'].'''
        return Prompt, Input, 'str_with_square_bracket'


class Fixed_pipeline_code_secured_door_changeable(Fixed_Puzzle_Object_Base):
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Parse_pipeline_dict:Dict[str,Dict[str,Union[str, int]]],
                 Usage:Optional[Dict[str, int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Belongs_to:Optional['Object'] = None,
                 Puzzle_answer:Optional[str] = None,
                 Special_label:Optional[str] = None,
                ):
        super().__init__(Keyword,
                         Information,
                         Parse_pipeline_dict,
                         Usage = Usage,
                         Model_name = Model_name,
                         Belongs_to = Belongs_to,
                         Puzzle_answer = Puzzle_answer,
                         Special_label = Special_label,)
        
            
        self.Try_limit = 3 # after this fail attemptation the password will change. 6 is the permutation of 3 digit
        self.Answer_generate_rule = 'blue'
        self.All_colors = ['blue', 'yellow', 'black', 'green']
        self.All_paintings = None
        # 3 digit password is always oil, acrylic, watercolor 
        
    def change_puzzle_answer(self):
        puzzle_failed = False
        previous_color_index = self.All_colors.index(self.Answer_generate_rule)
        cur_color_index = previous_color_index + 1
        try:
            self.Answer_generate_rule = self.All_colors[cur_color_index]
        except IndexError:
            puzzle_failed = True
        return puzzle_failed
        
    def get_password(self):
        first_digit = 0
        second_digit = 0
        third_digit = 0
        password_dict = {'oil':first_digit, 
                         'acrylic':second_digit,
                         'watercolor':third_digit}
        for paint in self.All_paintings:
            paint_color, paint_type, _ = paint.Special_label.split()
            if self.Answer_generate_rule == paint_color:
                password_dict[paint_type] += 1
        return f'''{password_dict['oil']}{password_dict['acrylic']}{password_dict['watercolor']}'''
        
        
    def judge_action_success(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        if self.All_paintings is None:
            all_spaces = attemptation_action.Host_CHIBI.Space_Manager_System_Global.Vertices_dict.values()
            self.All_paintings = []
            for space in all_spaces:
                for Thing_object in space.All_objects['Things']:
                    if Thing_object.Special_label.endswith('paint'):
                        self.All_paintings.append(Thing_object)
            
        CHIBI_input = None
        selected_action_pipeline = attemptation_action.Selected_action_interactive_pipeline
        if isinstance(selected_action_pipeline['Success_condition'],bool):
            '''if do systemic parse we consider this action is success else this action is fail'''
            success_fail_state = selected_action_pipeline['Success_condition']
            # If this Value is false means we set a action that will never be success (trap), don't know if this necessary, so currently, all action that set the Success_condition to bool should be called once and will doomed to be success
        else:
            CHIBI_input = self.decide_input(attemptation_action, memory_use = memory_use)
            CHIBI_input = re.sub(r'[^0-9]', '', CHIBI_input)
            if CHIBI_input is None:
                success_fail_state = False
            else:
                success_fail_state = True if CHIBI_input == self.get_password() else False
            attemptation_action.CHIBI_input = CHIBI_input
        attemptation_action.Success_fail_state = success_fail_state
        
        if CHIBI_input is not None:
            if not attemptation_action.Success_fail_state:
                success_fail_reason = f'''{attemptation_action.Host_CHIBI.Name} tried the following action: {attemptation_action.Information}, {attemptation_action.Host_CHIBI.Name}'s decision is {CHIBI_input}, but the password was incorrect.'''
                feedback_information = ''
                if len(CHIBI_input) != 3:
                    feedback_information = f'The {self.get_keyword()} requires 3 digit password, but your input have {len(CHIBI_input)} digits, which is not correct.'
                else: # len(CHIBI_input) == 3
                    cur_password = self.get_password()
                    if CHIBI_input[0] == cur_password[0]:
                        feedback_information += f'''{attemptation_action.Host_CHIBI.Name}'s first digit {CHIBI_input[0]} is correct. '''
                    else:
                        feedback_information += f'''{attemptation_action.Host_CHIBI.Name}'s first digit {CHIBI_input[0]} is incorrect. '''
                    if CHIBI_input[1] == cur_password[1]:
                        feedback_information += f'''{attemptation_action.Host_CHIBI.Name}'s second digit {CHIBI_input[1]} is correct. '''
                    else:
                        feedback_information += f'''{attemptation_action.Host_CHIBI.Name}'s second digit {CHIBI_input[1]} is incorrect. '''
                    if CHIBI_input[2] == cur_password[2]:
                        feedback_information += f'''{attemptation_action.Host_CHIBI.Name}'s third digit {CHIBI_input[2]} is correct. '''
                    else:
                        feedback_information += f'''{attemptation_action.Host_CHIBI.Name}'s third digit {CHIBI_input[2]} is incorrect. '''
                success_fail_reason += feedback_information
            else:
                success_fail_reason = f'''{CHIBI_input} is {'' if success_fail_state else 'not'} the correct input for {self.get_keyword()}.'''
            attemptation_action.Host_CHIBI.Memory_stream.memory_add(success_fail_reason, Memory_type = 'Observation', Importance_score = 15)
            
            if attemptation_action.Tried_times >= self.Try_limit:
                # to avoid brute force, the puzzle password will change
                puzzle_failed = self.change_puzzle_answer()
                if puzzle_failed:
                    raise utils.TaskFailedException(f'You tried too many times, and you failed this puzzle')
                else:
                    if self.Special_label == 'Level1':
                        experience_str = f'''You tried too many times, the password is now changed. The lines on the screen says: "The code from left to right the first digit is the number of {self.Answer_generate_rule} oil paintings, the second digit is the number of blue acrylic paintings and the third digit is the number of blue Watercolour paintings""'''
                    elif self.Special_label == 'Level2':
                        experience_str = f'''You tried too many times, the password is now changed. The lines on the screen says: "Focus on {self.Answer_generate_rule} it hides the truth."'''
                    else:
                        assert False, f'''Not known Special label {self.Special_label}'''
                    attemptation_action.Tried_times = 0
            else:
                experience_str = f'You can still try {self.Try_limit - attemptation_action.Tried_times} times before the password change!'
            attemptation_action.Host_CHIBI.Memory_stream.memory_add(experience_str, Memory_type = 'Observation', Importance_score = 20)

    # Object interfaces
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_relevant')->str:  # add a format for GPT to follow the format of generated input
        Prompt, Input, parse_function_str, logging_label = self._prompt_template(attemptation_action, memory_use = memory_use)
        decided_input = attemptation_action.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = logging_label)
        return str(decided_input) # simple thing do not need to have input and the success and out are fixed

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return super().return_action_information_construct(attemptation_action)
        
    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        '''some systemic level information should be changed in this step, eg object type, CHIBI open the box so the contained stuff should be exposed to every body, if CHIBI pass through the edge, CHIBI should move to the other side'''
        '''Can only be called if the action is success by judger'''
        '''Each object should have different systemic_parse pipeline'''
        '''Simple thing can only be used as tool'''
        super().systemic_parse(attemptation_action, memory_use = memory_use)
        if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'try finish':
            print(f'''Mission Complete, CHIBI found the correct code for the door!''')
            raise utils.TaskCompletedException(f'''Mission Complete, CHIBI found the correct code for the door!''')

    def _prompt_template(self, 
                        attemptation_action:'plan_system.Attemptation_Interactive_Action', 
                        memory_use:Optional[int] = None):
        previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str = attemptation_action.Host_CHIBI.retrieve_prompt_information(memory_use = memory_use)
        Prompt = f'''Based on the following recent experience of {attemptation_action.Host_CHIBI.Name}:{attemptation_action.Belongs_to.get_information()}\n\n{attemptation_action.Host_CHIBI.Name}'s current action is: {attemptation_action.get_information()} And you have the following information to decide what is the correct password:\n\n{previous_memories_str}{most_recent_memories_str}{storage_information}{cur_assumption_str}{cur_plan_str}Please follow the following steps to generate your final answer.'''
        Input = f'''**Step1** refelect the recent experience, what do you think is the password to {self.get_keyword()} is? Please only use information provided to do inference and give your reason. **Final Step** Please generate your final answer in a pair of square brackets. eg, if you think the final pass word is '112233' you should output ['112233'], if you think the output is '3a1b45v' please output ['3a1b45v'].'''
        return Prompt, Input, 'str_with_square_bracket', 'Interact_input'

class Fixed_pipeline_code_secured_door_changeable_function_operator(Fixed_Puzzle_Object_Base):
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Parse_pipeline_dict:Dict[str,Dict[str,Union[str, int]]],
                 Usage:Optional[Dict[str, int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',
                 Belongs_to:Optional['Object'] = None,
                 Puzzle_answer:Optional[str] = None,
                 Special_label:Optional[str] = None,
                ):
        super().__init__(Keyword,
                         Information,
                         Parse_pipeline_dict,
                         Usage = Usage,
                         Model_name = Model_name,
                         Belongs_to = Belongs_to,
                         Puzzle_answer = Puzzle_answer,
                         Special_label = Special_label,)
        self.Try_limit = 4 # after this fail attemptation the password will change. value of coefficient will randomly reassign
        # in current setting the code door must placed with Fixed_pipeline_Function_Operator_one_variable in the same space
        self.Function_operator = None
        
    def change_puzzle_answer(self):
        self.Function_operator.reassign_variable_values()

    def before_perceptual_effect(self, attemptation_action:'plan_system:Attemptation_Perceptual_Action'):
        if self.Function_operator is not None: # only assign once
            pass
        else:
            cur_space = self.Belongs_to
            for single_thing in cur_space.All_objects['Things']:
                if isinstance(single_thing, Fixed_pipeline_Function_Operator_one_variable):
                    self.Function_operator = single_thing
            assert self.Function_operator is not None, f'''Function operator is not found in current space, please check puzzle setting, place this object behind the function operator.'''
            self.Information = f'''This {self.get_keyword()} is the only exit from this room. To unlock it and leave, you must enter the correct code. The code is {len(self.Function_operator.Coefficient_map)} digits long, with each digit corresponding to the value of the parameter: {', '.join(list(self.Function_operator.Coefficient_map.keys()))}. You can discover the values of these parameters by interacting with the {self.Function_operator.get_keyword()}. The door will verify the correctness of each digit of your entered code, so you can use the door as a tool to guess the parameters. However, if you fail {self.Try_limit} times, the parameter values will change.'''
        Password_example_1 = random.sample(range(1,10), len(self.Function_operator.Coefficient_map))
        self.Password_example_1 = ''.join([str(i) for i in Password_example_1])
        
    def get_password(self):
        #print(self.Function_operator.Coefficient_map)
        return_password = ''
        for value in self.Function_operator.Coefficient_map.values():
            return_password += str(value)
        return return_password
        
    def judge_action_success(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
            
        CHIBI_input = None
        selected_action_pipeline = attemptation_action.Selected_action_interactive_pipeline
        if isinstance(selected_action_pipeline['Success_condition'],bool):
            '''if do systemic parse we consider this action is success else this action is fail'''
            success_fail_state = selected_action_pipeline['Success_condition']
            # If this Value is false means we set a action that will never be success (trap), don't know if this necessary, so currently, all action that set the Success_condition to bool should be called once and will doomed to be success
        else:
            CHIBI_input = self.decide_input(attemptation_action, memory_use = memory_use)
            CHIBI_input = re.sub(r'[^0-9]', '', CHIBI_input)
            if CHIBI_input is None:
                success_fail_state = False
            else:
                success_fail_state = True if CHIBI_input == self.get_password() else False
            attemptation_action.CHIBI_input = CHIBI_input
        attemptation_action.Success_fail_state = success_fail_state
        
        if CHIBI_input is not None:
            if not attemptation_action.Success_fail_state: # The action fails
                success_fail_reason = f'''{attemptation_action.Host_CHIBI.Name} tried the following action: {attemptation_action.Information}, {attemptation_action.Host_CHIBI.Name}'s decision is {CHIBI_input}, but the password was incorrect.'''
                feedback_information = ''
                if len(CHIBI_input) != len(self.Function_operator.Coefficient_map):
                    feedback_information = f'The {self.get_keyword()} requires {len(self.Function_operator.Coefficient_map)} digit password, but your input have {len(CHIBI_input)} digits, which is not correct.'
                else: # len(CHIBI_input) == len(self.Function_map.Coefficient_map)
                    cur_password = self.get_password()
                    def _get_order_str(order:int):
                        order_mapping_dict = {0:'first', 1:'second', 2:'third'}
                        if order in order_mapping_dict:
                            return f'''{order_mapping_dict[order]}'''
                        else:
                            return f'''{order+1}th'''
                    for index in range(len(cur_password)):
                        if CHIBI_input[index] == cur_password[index]:
                            feedback_information += f'Your {_get_order_str(index)} digit {CHIBI_input[index]} is correct. '
                        else:
                            feedback_information += f'Your {_get_order_str(index)} digit {CHIBI_input[index]} is incorrect. '
                        
                success_fail_reason += feedback_information
            else:
                success_fail_reason = f'''{CHIBI_input} is {'' if success_fail_state else 'not'} the correct input for {self.get_keyword()}.'''
            attemptation_action.Host_CHIBI.Memory_stream.memory_add(success_fail_reason, Memory_type = 'Observation', Importance_score = 15)
            
            if attemptation_action.Tried_times >= self.Try_limit:
                # to avoid brute force, the puzzle password will change
                self.change_puzzle_answer()
                experience_str = f'''\nYou tried too many times, the value of parameters have been reassigned(Only the value of parameters changed, the functions stays the same). And the password of the door changed accordingly, you may need to keep on interacting with the {self.Function_operator.get_keyword()} and decode the password.\n'''
                attemptation_action.Tried_times = 0
            else:
                experience_str = f'You can still try {self.Try_limit - attemptation_action.Tried_times} times before the password change!'
            attemptation_action.Host_CHIBI.Memory_stream.memory_add(experience_str, Memory_type = 'Observation', Importance_score = 20)

    # Object interfaces
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        
    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)
        
    def get_keyword(self):
        return f'<{self.Keyword}>'

    def destory(self):
        if self.Belongs_to is not None:
            self.Belongs_to.object_delete(self)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_relevant')->str:  # add a format for GPT to follow the format of generated input
        Prompt, Input, parse_function_str, logging_label = self._prompt_template(attemptation_action, memory_use = memory_use)
        decided_input = attemptation_action.Host_CHIBI.CHIBI_input(Prompt, Input, parse_function_str = parse_function_str, logging_label = logging_label)
        return str(decided_input) # simple thing do not need to have input and the success and out are fixed

    def return_action_information_construct(self, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return super().return_action_information_construct(attemptation_action)
        
    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        '''some systemic level information should be changed in this step, eg object type, CHIBI open the box so the contained stuff should be exposed to every body, if CHIBI pass through the edge, CHIBI should move to the other side'''
        '''Can only be called if the action is success by judger'''
        '''Each object should have different systemic_parse pipeline'''
        '''Simple thing can only be used as tool'''
        super().systemic_parse(attemptation_action, memory_use = memory_use)
        if attemptation_action.Selected_action_interactive_pipeline['Systemic_parse_id'] == 'try finish':
            print(f'''Mission Complete, CHIBI found the correct code for the door!''')
            raise utils.TaskCompletedException(f'''Mission Complete, CHIBI found the correct code for the door!''')

    def _prompt_template(self, 
                        attemptation_action:'plan_system.Attemptation_Interactive_Action', 
                        memory_use:Optional[int] = None):
        previous_memories_str, most_recent_memories_str, storage_information, cur_assumption_str, cur_plan_str = attemptation_action.Host_CHIBI.retrieve_prompt_information(memory_use = memory_use)
        Prompt = f'''Based on the following recent experience of {attemptation_action.Host_CHIBI.Name}:{attemptation_action.Belongs_to.get_information()}\n\n{attemptation_action.Host_CHIBI.Name}'s current action is: {attemptation_action.get_information()} And you have the following information to decide what is the correct password:\n\n{previous_memories_str}{most_recent_memories_str}{cur_assumption_str}{cur_plan_str}Please follow the following steps to generate your final answer.'''
        Input = f'''**Step 1:** Based on your interaction with {self.Function_operator.get_keyword()}, what do you believe the values of these parameters should be? Please format your guess for the password in alphabetical order of the parameters (e.g., if you think a=3, b=2, c=5, d=9, then the password should be 3259).**Final Step:** Enter your final answer within square brackets. For example, if you believe the final password is '{self.Password_example_1}', you should output ['{self.Password_example_1}'].'''
        return Prompt, Input, 'str_with_square_bracket', 'Interact_input'

# -----------------------------------------------------------State_machine_Objects---------------------------------------------------------------
# -----------------------------------------------------------State_machine_Objects---------------------------------------------------------------
# -----------------------------------------------------------State_machine_Objects---------------------------------------------------------------

class State_machine_objects_Base(Fixed_Interact_Pipeline_Object_Base, ABC):
    def __init__(self,
                 Keyword:str,
                 Information:str,
                 Parse_pipeline_dict:Dict[str,Dict[str,Union[str,int]]],
                 Usage:Optional[Dict[str,int]] = None,
                 Model_name:str = 'gpt-3.5-turbo-0125',#gpt-4-1106-preview,
                 Belongs_to:Optional['Object'] = None,
                 State_machine_actions:Optional[Dict[str,Any]] = None,
                 Special_label:Optional[str] = None,
                ):
        super().__init__(Keyword, Information, 
                         Usage = Usage,
                         Model_name = Model_name,
                         Belongs_to = Belongs_to,
                         Parse_pipeline_dict = Parse_pipeline_dict,
                         Special_label = Special_label)
        self.State_machine_actions = State_machine_actions


    # State machine interfaces-----------------------------------------
    def update(self):
        actions = self.get_state_machine_action()
        for action in actions:
            self.take_state_machine_actions(action)
            
        
    def semantic_parse_state_machine(self, state_machine_action_information:Dict[str,Any]):
        selected_action_pipeline = attemptation_action.Selected_action_interactive_pipeline

        # edit semantic information for this object including information keyword edit and generate subparts
        information_edit_data = state_machine_action_information['Information_edit']
        if isinstance(information_edit_data, int):
            # Create new object and delete current object and replace impression
            self.replace_object_state_machine(information_edit_data)
        else:
            # the edit only changes the information, and do not need to generate new object, just change the objects' information and update impression data
            #print(f'information edit data: {information_edit_data}')
            self.Information = information_edit_data
        
        generated_subparts = selected_action_pipeline['Generated_subparts']
        if generated_subparts is not None:
            for subpart_id in generated_subparts:
                new_object = Fixed_Block_helper.create_fixed_object_with_database(subpart_id)
                new_object.Belongs_to = self.Belongs_to
                self.Belongs_to.object_add(new_object)

    def replace_object_state_machine(self, information_edit_data):
        # Step1 create new object,
        new_object = Fixed_Block_helper.create_fixed_object_with_database(replace_object_id)
        new_object.Belongs_to = self.Belongs_to
        if new_object.Belongs_to is not None:
            new_object.Belongs_to.object_add(new_object)
        if isinstance(new_object, Fixed_pipeline_Simple_Edge):
            vertices = self.Connected_two_space
            for vertex in vertices:
                vertex.object_add(new_object)
                new_object.Connected_two_space = vertices
        self.destory()
                
    def broad_cast_information(self,
                               broadcast_information:str):
        broadcast_information = self.update_str_with_variable(broadcast_information)
        if isinstance(self.Belongs_to, blocks.Space_System_global):
            all_chibis = self.Belongs_to.retrieve_item_in_this_space(object_type = 'CHIBIs')
            for chibi in all_chibis:
                chibi.Memory_stream.memory_add(broadcast_information, Memory_type = 'Observation')
        else:
            # TODO perhaps the item is in chibi storage the CHIBI should know what happened
            pass
        
    @abstractmethod
    def systemic_parse_state_machine(self,state_machine_action_information:Dict[str,Any]):
        # every object should have different systemic parse
        pass


    def take_state_machine_actions(self, state_machine_action_information:Dict[str,Any]):
        # Semantic parse
        if state_machine_action_information['Information_edit'] is not None:
            self.semantic_parse_state_machine(state_machine_action_information)
        
        # Systemic parse
        if state_machine_action_information['Systemic_parse_id'] is not None:
            self.systemic_parse_state_machine(state_machine_action_information)

        # information broad cast
        if state_machine_action_information['Broadcast_information'] is not None:
            self.broad_cast_information(state_machine_action_information['Broadcast_information'])

        # destory item
        if state_machine_action_information['Destory_bool']:
            self.destory()

        if not state_machine_action_information['Repeat_action']:
            self.State_machine_actions.remove(state_machine_action_information)

    def get_variable_value_state_machine(self, variable_expression:str):
        return_variable_value = None
        '''Current support variables in Base class:
           Object_exist: If there is an object that have the same keyword in the current space
           Object_slot_1/2: If there is an object in the object slot, or exact the object in the slot'''
        # Since some of the object need sepefic unique class variable to decide when to call this action, so each subclass need to make sure if all variables can successfully parsed into True or False
        if ':' not in variable_expression:
            if variable_expression == 'Len_object_slot_1':
                return_variable_value = len(self.Object_slot_1)
            elif variable_expression == 'Len_object_slot_2':
                return_variable_value = len(self.Object_slot_2)
        else:
            variable_name = variable_expression.split(':')[0]
            variable_value = variable_expression.split(':')[1]
            if variable_name == 'Object_exist':
                all_objects_in_the_same_scope = self.Belongs_to.All_objects
                if isinstance(all_objects_in_the_same_scope, dict):
                    all_object_list = []
                    for value in all_objects_in_the_same_scope.values():
                        all_object_list.extend(value)
                    all_objects_in_the_same_scope = all_object_list
                object_name_list = [i.Keyword for i in all_objects_in_the_same_scope]
                return_variable_value = variable_value in object_name_list
            elif variable_name == 'Object_slot_1':
                if len(self.Object_slot_1) == 0:
                    return_variable_value = False
                else:
                    if variable_value == '1': # '1' means have object in this slot but no matter what object
                        return_variable_value = True
                    else:
                        return_variable_value = self.Object_slot_1[0].Keyword == variable_value
            elif variable_name == 'Object_slot_2':
                if len(self.Object_slot_2) == 0:
                    return_variable_value = False
                else:
                    if variable_value == '1':
                        return_variable_value = True
                    else:
                        return_variable_value = self.Object_slot_1[0].Keyword == variable_value

        return return_variable_value

    def check_condition_state_machine(self,condition_str:str)->bool:
        pattern = re.compile(r'\{([^}]+)\}')
        matches = pattern.findall(condition_str)

        for match in matches:
            result = self.get_variable_value_state_machine(match)
            condition_str = condition_str.replace(f'{{{match}}}', str(result))

        try:
            return eval(condition_str)
        except Exception as e:
            return f"Error evaluating expression: {e}"
        
    def get_state_machine_action(self)->List[Dict[str,str]]:# should return all callable statemachine actions
        return_action_list = []
        for state_machine_action in self.State_machine_actions:
            if self.check_condition_state_machine(state_machine_action['Auto_call_condition']):
                return_action_list.append(state_machine_action)
        return return_action_list

    def find_linked_object_state_machine(self,state_machine_action_information:Dict[str,Any]):
        # find the linked object in the current Space, if don't find one then return none
        return_dict = {}
        linked_object_keyword = state_machine_action_information['Linked_objects']
        assert isinstance(self.Belongs_to, blocks.Space_System_global), f'Currently state machine object only can find object in current space'
        all_objects = self.Belongs_to.retrieve_item_in_this_space(object_type = 'All')
        for keyword in linked_object_keyword:
            for single_object in all_objects:
                if keyword == single_object.Keyword:
                    return_dict.update({keyword:single_object})
        return return_dict
            

class State_machine_object_animal(State_machine_objects_Base): # State machine object with given systemic parse actions

    # CHIBI Object interfaces-----------------------------------------------------------------
    def show(self):
        print(self.Keyword)
        print(self.Information)
        print(self.Parse_pipeline_dict)
        print(self.State_machine_actions)

    def destory(self):
        super().destory()

    def get_keyword(self):
        return f'<{self.Keyword}>'

    def get_information(self, viewer:Optional[Union[str,'CHIBI.CHIBI_Base']] = None):
        return super().get_information(viewer)

    def edit(self):
        assert False, f'''Information of {type(self)} can only be edited via predefined pipeline'''

    # Regular fixed pipeline object interfaces------------------------------------------------
    def get_variable_value(self, variable_expression:str, attemptation_action:'plan_system.Attemptation_Interactive_Action'):
        return_variable_value = super().get_variable_value(variable_expression, attemptation_action)
        # add some new unique variables when needed
        return return_variable_value

    def decide_input(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None, memory_retrieve_type:str = 'Most_recent')->str:  # add a format for GPT to follow the format of generated input
        return super().decide_input(attemptation_action, memory_use = memory_use, memory_retrieve_type = memory_retrieve_type) # Regular Fixed pipeline thing will only need to dec

    def return_action_information_construct(self, attemptation_action:Union['plan_system.Attemptation_Interactive_Action',str])->Dict[str,str]:
        return super().return_action_information_construct(attemptation_action) # used for state machine boradcast


    def systemic_parse(self, attemptation_action:'plan_system.Attemptation_Interactive_Action', memory_use:Optional[int] = None):
        super().systemic_parse(attemptation_action, memory_use = memory_use)
            
    def systemic_parse_state_machine(self, state_machine_action_information:Dict[str,Any]):
        if state_machine_action_information['Systemic_parse_id'] == 'eat':
            linked_objects = self.find_linked_object_state_machine(state_machine_action_information)
            object_to_be_eaten = list(linked_objects.values())[0] # since only have one object for eat action
            object_to_be_eaten.destory()
            raise utils.TaskCompletedException(f'Mission failed the {self.get_keyword()} eats the {object_to_be_eaten.get_keyword()}')
        

    # state machine interfaces -----------------------------------------------------------

    # TODO state machine object should have different systemic parse and semantic parse pipeline only for state machin actions


class Fixed_Block_helper:
    def resource_path(relative_path):
        import sys
        import os
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    Model_name = 'gpt-4-1106-preview'
    Object_database = pd.read_excel(resource_path('data/CHIBI_database.xlsx'), sheet_name = 'Atom_Objects', dtype={'Object_id':int, 'Special_label':str})
    Action_database = pd.read_excel(resource_path('data/CHIBI_database.xlsx'), sheet_name = 'Predefined_Actions', keep_default_na=False, na_values=[''])
    State_machine_action_database = pd.read_excel(resource_path('data/CHIBI_database.xlsx'), sheet_name = 'State_Machine_Actions', keep_default_na=False, na_values=[''], dtype={'Repeat_action': str,'Destory_bool': str})
    @staticmethod
    def create_fixed_object_with_dict(input_dict:Dict[str,Any],
                                      Model_name:Optional[str] = None) -> Fixed_Interact_Pipeline_Object_Base:
        if Model_name is None:
            Model_name = Fixed_Block_helper.Model_name
        class_type = globals()[input_dict['Type']]
        instance = class_type(input_dict['Keyword'], input_dict['Information'], input_dict['Interative_pipeline'], Model_name = Model_name)
        return instance

    @staticmethod
    def create_fixed_object_with_database(input_object_information:Union[int,Dict[str, str]], # could be a integer or dict
                                          Object_database:Optional[pd.DataFrame] = None,
                                          Action_database:Optional[pd.DataFrame] = None,
                                          State_machine_action_database:Optional[pd.DataFrame] = None,
                                          Model_name:Optional[str] = None)->Fixed_Interact_Pipeline_Object_Base:
        # if the input_object_information is a dict, it should have object_id target to a template object, and could have object information, keyword, Puzzle_answer as keys.
        if Object_database is None:
            Object_database = Fixed_Block_helper.Object_database
        if Action_database is None:
            Action_database = Fixed_Block_helper.Action_database
        if State_machine_action_database is None:
            State_machine_action_database = Fixed_Block_helper.State_machine_action_database
        if isinstance(input_object_information, int):
            object_information = Object_database[Object_database['Object_id'] == input_object_information]
            assert len(object_information) != 0, f'No object found id == {object_id} in the database.'
            assert len(object_information) == 1, f'object_id should be unique for all objects, but the id:{object_id} do not have exact one objcet, it have {len(object_information)} objects'
            object_information = object_information.iloc[0]
            object_id = input_object_information

        else: # object information is a dict
            assert isinstance(input_object_information, dict), f'''Input object id should be dict or int'''
            assert 'Object_id' in input_object_information, f'''If input is a dict with additional or replaceable information at leaset this dict should contain an object point to which the object is modified on'''
            # Currently this allow input a dict with replaceable Keyword, Information, Puzzle answer, if you want to create an object with different actions, you need to create a new object and it's corresponding actions in the database!
            object_id = input_object_information['Object_id']
            object_information = Object_database[Object_database['Object_id'] == object_id]
            assert len(object_information) == 1, f'object_id should be unique for all objects, but the id:{object_id} do not have exact one objcet, it have {len(object_information)} objects'
            object_information = object_information.iloc[0]
            if input_object_information.get('Keyword') is not None:
                object_information['Keyword'] = input_object_information['Keyword']
            if input_object_information.get('Information') is not None:
                if input_object_information.get('Keyword') is not None:
                    object_information['Information'] = input_object_information['Information'].format(**input_object_information) 
                    # parse Keyword to the object description if needed
                else:
                    object_information['Information'] = input_object_information['Information']
            if input_object_information.get('Puzzle_answer') is not None:
                object_information['Puzzle_answer'] = input_object_information['Puzzle_answer']
            if input_object_information.get('Special_label') is not None:
                object_information['Special_label'] = input_object_information['Special_label']
        
        object_information['Information'] = object_information['Information'].format(Keyword = f'''<{object_information['Keyword']}>''') if '{Keyword}' in object_information['Information'] else object_information['Information']
        
        parse_pipeline = Fixed_Block_helper.create_parse_pipeline_dict(object_information, Object_database, Action_database)
        if Model_name is None:
            Model_name = Fixed_Block_helper.Model_name
        class_type = globals()[object_information['Object_type']]

        if issubclass(class_type, Fixed_Puzzle_Object_Base) and object_information.get('Puzzle_answer') is not None:
            instance = class_type(object_information['Keyword'], object_information['Information'], parse_pipeline, Model_name = Model_name, Puzzle_answer = object_information['Puzzle_answer'], Special_label = object_information['Special_label']) # in order to prevent create similar objects every time for a new puzzle in the database, this allow only make modification in secene py file. To change the puzzle answer input a dict with keyword ”Puzzle_answer“.
        else:
            instance = class_type(object_information['Keyword'], object_information['Information'], parse_pipeline, Model_name = Model_name, Special_label = object_information['Special_label'])

        if issubclass(class_type, State_machine_objects_Base):
            tem_state_machine_actions = Fixed_Block_helper.create_state_machine_parse_dict(object_id, 
                                                                                    Object_database = Object_database,
                                                                                    Action_database = Object_database,
                                                                                    State_machine_action_database = State_machine_action_database)
            instance.State_machine_actions = tem_state_machine_actions
        return instance

    @staticmethod
    def create_fixed_container_with_database(input_object_information:Union[int,Dict[str, str]],
                                             contained_object_id_list = List[int],
                                             Object_database:Optional[pd.DataFrame] = None,
                                             Action_database:Optional[pd.DataFrame] = None,
                                             Model_name:Optional[str] = None) -> Fixed_pipeline_Simple_Container:
        if Object_database is None:
            Object_database = Fixed_Block_helper.Object_database
        if Action_database is None:
            Action_database = Fixed_Block_helper.Action_database
        if Model_name is None:
            Model_name = Fixed_Block_helper.Model_name
        container_object = Fixed_Block_helper.create_fixed_object_with_database(input_object_information, Object_database = Object_database, Action_database = Action_database, Model_name = Model_name)
        contained_objects = []
        for contained_object_id in contained_object_id_list:
            contained_object = Fixed_Block_helper.create_fixed_object_with_database(contained_object_id, Object_database = Object_database, Action_database = Action_database)
            contained_objects.append(contained_object)
        container_object.Contained_objects.extend(contained_objects)
        for contained_object in contained_objects:
            contained_object.Belongs_to = container_object
        return container_object

    @staticmethod
    def create_fixed_edge_with_database(input_object_information:Union[int,Dict[str, str]],
                                        connected_two_space:List['Space_System_global'],
                                        Object_database:Optional[pd.DataFrame] = None,
                                        Action_database:Optional[pd.DataFrame] = None,
                                        Model_name:Optional[str] = None) -> Fixed_pipeline_Simple_Edge:
        if Object_database is None:
            Object_database = Fixed_Block_helper.Object_database
        if Action_database is None:
            Action_database = Fixed_Block_helper.Action_database
        if Model_name is None:
            Model_name = Fixed_Block_helper.Model_name

        edge_object = Fixed_Block_helper.create_fixed_object_with_database(input_object_information, Object_database = Object_database, Action_database = Action_database, Model_name = Model_name) 
        edge_object.Connected_two_space = connected_two_space
        #edge_object.show()
        return edge_object

    @staticmethod
    def parse_condition_sentence(condition_str: str,
                             Object_database: Optional[pd.DataFrame] = None,
                             Action_database: Optional[pd.DataFrame] = None,
                             State_machine_action_database: Optional[pd.DataFrame] = None):
        if Object_database is None:
            Object_database = Fixed_Block_helper.Object_database
        if Action_database is None:
            Action_database = Fixed_Block_helper.Action_database
        if State_machine_action_database is None:
            State_machine_action_database = Fixed_Block_helper.State_machine_action_database
        pattern = r'\{([^{}]*)\}'
    
        def replace_match(match):
            full_match = match.group()
            parts = full_match.strip('{}').split(':')
            if len(parts) == 1:
                return '{' + parts[0] + '}'
    
            variable_name, variable_value = parts
            object_name = None
            try:
                object_name = Object_database[Object_database['Object_id'] == int(variable_value)].iloc[0]['Keyword']
            except ValueError:
                object_name = variable_value
    
            assert object_name is not None, f'value of the following condition variable{(variable_name, variable_value)} not correctlly parsed'
            return '{' + variable_name + ':' + object_name + '}'

        replaced_condition_str = re.sub(pattern, replace_match, condition_str)
        return replaced_condition_str
        
    @staticmethod
    def create_parse_pipeline_dict(Object_input_information:Union[int, Dict[str,Any]], 
                                   Object_database:Optional[pd.DataFrame] = None,
                                   Action_database:Optional[pd.DataFrame] = None)->Dict[str,Dict[str,Any]]:
        if Object_database is None:
            Object_database = Fixed_Block_helper.Object_database
        if Action_database is None:
            Action_database = Fixed_Block_helper.Action_database
        if isinstance(Object_input_information, int):
            Object_id = Object_input_information
        else:
            Object_id = Object_input_information['Object_id']
        all_actions = Action_database[Action_database['Related_object_id'] == Object_id]
        if len(all_actions) == 0: # this object is not interactable, the only Purpose of such object is to provide information about this object
            return {}
        parse_pipeline_dict = {}
        for i in range(len(all_actions)):
            action_information = all_actions.iloc[i]
            Destory_bool = True if action_information['Destory_bool'] else False
            related_object_keyword = f'''<{Object_database[Object_database['Object_id'] == action_information['Related_object_id']].iloc[0]['Keyword']}>'''
            if action_information['Show_condition'] in ['True', 'False']:
                Show_condition = bool(action_information['Show_condition'])
            else:
                Show_condition = Fixed_Block_helper.parse_condition_sentence(action_information['Show_condition'],
                                                                            Object_database = Object_database,
                                                                            Action_database = Action_database)
            if action_information['Success_condition'] in ['True', 'False']:
                Success_condition = ast.literal_eval(action_information['Success_condition'])
            else:
                condition_list = action_information['Success_condition'].split(':')
                if condition_list[0] == 'Object':
                    Success_condition = Object_database[Object_database['Object_id'] == int(condition_list[1])].iloc[0]['Keyword']
                elif condition_list[0] == 'String':
                    Success_condition = condition_list[1]
                else:
                    assert False, f'{condition_list[0]} is not a known type of success condition'
            if action_information['Systemic_parse_id'] == 'None':
                Systemic_parse_id = None
            else:
                Systemic_parse_id = action_information['Systemic_parse_id']

            if action_information['Information_edit'] == 'None':
                Information_edit = None
            elif action_information['Information_edit'].startswith('Object'):
                Information_edit = int(action_information['Information_edit'].split(':')[1])
            else:
                Information_edit = action_information['Information_edit']
    
            if action_information['Action_return_information'] == 'None':
                Action_return_information = None
            else:
                Action_return_information = action_information['Action_return_information']
    
            if action_information['Generated_subparts'] == 'None':
                Generated_subparts = None
            else:
                Generated_subparts = [int(subpart_id) for subpart_id in str(action_information['Generated_subparts']).split(',')]

            if action_information['Linked_objects'] == 'None':
                Linked_objects = []
            else:
                Linked_objects = []
                object_string_list = action_information['Linked_objects'].split(',')
                for object_string in object_string_list:
                    if object_string.startswith('Object'): # we use id to find the name of the object in the database
                        tem_object_id = int(object_string.split(':')[1])
                        tem_object_keyword = Object_database[Object_database['Object_id'] == tem_object_id].iloc[0]['Keyword']
                        Linked_objects.append(tem_object_keyword)
                    elif object_string.startswith('String'):
                        tem_object_keyword = object_string.split(':')[1]
                        Linked_objects.append(tem_object_keyword)
                    elif object_string.startswith('Object_index'):
                        Linked_objects.append(int(object_string.split(':')[1]))
                    

            if isinstance(action_information['Repeat_action'], str):
                Repeat_action = True if ast.literal_eval(action_information['Repeat_action']) else False
            else:
                Repeat_action = True if action_information['Repeat_action'] else False
                    
            cur_pipeline_dict = {'Action_str':action_information['Action_str'],
                                 'Destory_bool':Destory_bool,
                                 'Show_condition':Show_condition,
                                 'Success_condition':Success_condition,
                                 'Systemic_parse_id':Systemic_parse_id,
                                 'Information_edit':Information_edit,
                                 'Action_return_information':Action_return_information,
                                 'Generated_subparts':Generated_subparts,
                                 'Linked_objects':Linked_objects,
                                 'Repeat_action':Repeat_action,
                                 'Marker':action_information['Marker'],
                                }

            action_string = action_information['Action_str']
            parse_pipeline_dict.update({action_string:cur_pipeline_dict})
        return parse_pipeline_dict

    
    @staticmethod
    def create_state_machine_parse_dict(Object_id:int, 
                                        Object_database:Optional[pd.DataFrame] = None,
                                        Action_database:Optional[pd.DataFrame] = None,
                                        State_machine_action_database:Optional[pd.DataFrame] = None,)->Dict[str,Dict[str,Any]]:
        if Object_database is None:
            Object_database = Fixed_Block_helper.Object_database
        if Action_database is None:
            Action_database = Fixed_Block_helper.Action_database
        if State_machine_action_database is None:
            State_machine_action_database = Fixed_Block_helper.State_machine_action_database
        all_actions = State_machine_action_database[State_machine_action_database['Related_object_id'] == Object_id]
        auto_call_actions = []
        assert len(all_actions) > 0, f'''The obejct: {Object_id} do not have statemachine actions defined in the database, this object should not be a statemachine object'''
        for i in range(len(all_actions)):
            action_information = all_actions.iloc[i]
            Destory_bool = True if action_information['Destory_bool'] else False
            Auto_call_condition = Fixed_Block_helper.parse_condition_sentence(action_information['Auto_call_condition'],
                                                                              Object_database = Object_database,
                                                                              Action_database = Action_database,
                                                                              State_machine_action_database = State_machine_action_database)
            Systemic_parse_id = action_information['Systemic_parse_id']
            
            if action_information['Information_edit'] == 'None':
                Information_edit = None
            elif action_information['Information_edit'].startswith('Object'):
                Information_edit = int(action_information['Information_edit'].split(':')[1])
            else:
                Information_edit = action_information['Information_edit']

            if action_information['Broadcast_information'] == 'None':
                Broadcast_information = None
            else:
                Broadcast_information = action_information['Broadcast_information']

            if action_information['Generated_subparts'] == 'None':
                Generated_subparts = None
            else:
                Generated_subparts = [int(subpart_id) for subpart_id in str(action_information['Generated_subparts']).split(',')]

            if action_information['Linked_objects'] == 'None':
                Linked_objects = []
            else:
                Linked_objects = []
                object_string_list = action_information['Linked_objects'].split(',')
                for object_string in object_string_list:
                    if object_string.startswith('Object'): # we use id to find the name of the object in the database
                        tem_object_id = int(object_string.split(':')[1])
                        tem_object_keyword = Object_database[Object_database['Object_id'] == tem_object_id].iloc[0]['Keyword']
                        Linked_objects.append(tem_object_keyword)
                    elif object_string.startswith('String'):
                        tem_object_keyword = object_string.split(':')[1]
                        Linked_objects.append(tem_object_keyword)

            if isinstance(action_information['Repeat_action'], bool):
                Repeat_action = True if action_information['Repeat_action'] else False
            else:
                Repeat_action = True if ast.literal_eval(action_information['Repeat_action']) else False
            cur_pipeline_dict = {
                                 'Destory_bool':Destory_bool,
                                 'Auto_call_condition':Auto_call_condition,
                                 'Systemic_parse_id':Systemic_parse_id,
                                 'Information_edit':Information_edit,
                                 'Broadcast_information':Broadcast_information,
                                 'Generated_subparts':Generated_subparts,
                                 'Linked_objects':Linked_objects,
                                 'Repeat_action':Repeat_action,
                                 'Marker':action_information['Marker'],
                                }
            auto_call_actions.append(cur_pipeline_dict)
        return auto_call_actions
            
        